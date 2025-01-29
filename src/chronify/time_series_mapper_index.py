import logging
import abc
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import Engine, MetaData, Table, select


from chronify.datetime_range_generator import DatetimeRangeGenerator
from chronify.models import TableSchema, MappingTableSchema
from chronify.sqlalchemy.functions import read_database
from chronify.exceptions import (
    InvalidParameter,
)
from chronify.time import DaylightSavingsDataAdjustment, TimeType
from chronify.time_utils import shift_and_wrap_time_intervals
from chronify.time_series_mapper_base import TimeSeriesMapperBase, apply_mapping
from chronify.index_time_range_generator import IndexTimeRangeGenerator
from chronify.time_configs import (
    DatetimeRange,
    IndexTimeRange,
    TimeBasedDataAdjustment,
)


logger = logging.getLogger(__name__)


class MapperIndexTimeToDatetime(TimeSeriesMapperBase):
    def __init__(
        self,
        engine: Engine,
        metadata: MetaData,
        from_schema: TableSchema,
        to_schema: TableSchema,
        time_based_data_adjustment: TimeBasedDataAdjustment | None = None,
    ) -> None:
        """
        Parameters
        ----------

        engine : Engine
            sqlalchemy engine to connect to database

        metadata : MetaData
            metadata for the database

        from_schema : TableSchema
            schema mapping from the input dataset

        to_schema : TableSchema
            schema mapping to the project

        time_based_data_adjustment : TimeBasedDataAdjustment
            enumerates actions to take that will duplicate, drop or adjust data based on time mapping
        """
        if not isinstance(from_schema.time_config, IndexTimeRange):
            msg = "source schema does not have IndexTimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        if not isinstance(to_schema.time_config, DatetimeRange):
            msg = "destination schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)

        super().__init__(engine, metadata, from_schema, to_schema)

        self._from_time_config: IndexTimeRange = from_schema.time_config
        self._to_time_config: DatetimeRange = to_schema.time_config
        self._from_schema = from_schema

        self._to_schema = to_schema

        self._engine = engine
        self._metadata = metadata
        self._time_based_data_adjustment = time_based_data_adjustment or TimeBasedDataAdjustment()

        if "time_zone" in self._from_schema.time_array_id_columns:
            if not self._from_time_config.is_time_zone_naive():
                msg = "Can't map multiple time-zones with a start timestamp with a time zone"
                raise InvalidParameter(msg)
            time_zones = self._list_time_zones()

            self._mapping_generator = MultipleLocalMappingGenerator(
                to_time_config=self._to_time_config,
                from_time_config=self._from_time_config,
                to_time_col=self._to_time_config.time_column,
                from_time_col="from_" + self._from_time_config.time_column,
                time_based_data_adjustment=self._time_based_data_adjustment,
                time_zones=time_zones,
            )
        else:
            self._mapping_generator = SimpleMappingGenerator(
                to_time_config=self._to_time_config,
                from_time_config=self._from_time_config,
                to_time_col=self._to_time_config.time_column,
                from_time_col="from_" + self._from_time_config.time_column,
                time_based_data_adjustment=self._time_based_data_adjustment,
            )

        # self._not_supported_mappings()

    def check_schema_consistency(self) -> None:
        # TODO: fail for interpolate fall_back_hour
        pass

    def map_time(
        self,
        wrap_time_allowed: bool = False,
        **kwargs,
    ) -> None:
        """Convert time columns with from_schema to to_schema configuration."""
        dfm = self._create_mapping(wrap_time_allowed)
        self._map_time(dfm, **kwargs)

    def _map_time(self, dfm: pd.DataFrame, **kwargs):
        self.check_schema_consistency()

        map_table_name = "map_table"
        mapping_schema = self._create_mapping_schema(map_table_name)

        apply_mapping(
            dfm,
            mapping_schema,
            self._from_schema,
            self._to_schema,
            self._engine,
            self._metadata,
            **kwargs,
        )

    def _list_time_zones(self):
        # Could be cached?
        self._metadata.reflect(self._engine)
        with self._engine.connect() as conn:
            table = Table(self._from_schema.name, self._metadata)
            stmt = select(table.c["time_zone"]).distinct().where(table.c["time_zone"].is_not(None))
            time_zones = read_database(stmt, conn, self._from_time_config)["time_zone"].to_list()
        return time_zones

    def _create_mapping(self, wrap_time_allowed: bool = False) -> pd.DataFrame:
        """Create mapping dataframe"""

        return self._mapping_generator.create_mapping(wrap_time_allowed)

    def _create_mapping_schema(self, table_name: str):
        return MappingTableSchema(name=table_name, time_configs=[self._to_time_config])


class BaseMappingGenerator:
    def __init__(
        self,
        to_time_config: DatetimeRange,
        from_time_config: IndexTimeRange,
        to_time_col: str,
        from_time_col: str,
        time_based_data_adjustment: TimeBasedDataAdjustment,
    ) -> None:
        self._to_time_config = to_time_config
        self._from_time_config = from_time_config
        self._to_time_col = to_time_col
        self._intermediate_time_col = "intermediate_timestamp"
        self._from_time_col = from_time_col
        self._time_based_data_adjustment = time_based_data_adjustment

        self._index_generator = IndexTimeRangeGenerator(self._from_time_config)

        self._intermediate_time_config = self.create_intermediate_schema()
        self._intermediate_datetime_generator = DatetimeRangeGenerator(
            self._intermediate_time_config
        )

        # adjusts for leapyear, but no dst
        self._datetime_generator = DatetimeRangeGenerator(
            self._to_time_config, self._time_based_data_adjustment
        )

    @property
    def _align_to_local_clocktime(self):
        # TODO: doesn't handle non dst timezone from configs
        return (
            self._from_time_config.is_time_zone_naive()
            and self._time_based_data_adjustment.daylight_saving_adjustment
            != DaylightSavingsDataAdjustment.NONE
        )

    @abc.abstractmethod
    def _create_mapping(self) -> pd.DataFrame:
        """creates a mapping for a given mapping generator."""

    def create_mapping(self, wrap_time_allowed: bool = False):
        dfm = self._create_mapping()
        if self._intermediate_time_config.is_time_zone_naive():
            tz_converted_timestamps = dfm[self._intermediate_time_col].dt.tz_localize(
                self._to_time_config.start.tzinfo
            )
        else:
            tz_converted_timestamps = dfm[self._intermediate_time_col].dt.tz_convert(
                self._to_time_config.start.tzinfo
            )

        dfm[self._to_time_col] = shift_and_wrap_time_intervals(
            self._datetime_generator.list_timestamps(),
            tz_converted_timestamps,
            self._intermediate_time_config,
            self._to_time_config,
            wrap_time_allowed,
        )
        dfm = dfm.drop(columns=[self._intermediate_time_col])
        if self._align_to_local_clocktime:
            dfm = self._adjust_mapping_local_clock_time_to_dst(dfm)
        return dfm

    def create_intermediate_schema(self) -> DatetimeRange:
        time_kwargs = self._from_time_config.model_dump()
        time_kwargs = dict(
            filter(lambda k_v: k_v[0] in DatetimeRange.model_fields, time_kwargs.items())
        )
        time_kwargs["start"] = self._from_time_config.start_timestamp
        time_kwargs["time_type"] = TimeType.DATETIME
        time_config = DatetimeRange(**time_kwargs)

        return time_config

    def _adjust_mapping_local_clock_time_to_dst(self, dfm):
        df_idx_clock_time = pd.DataFrame(
            {
                self._from_time_col: self._index_generator.list_timestamps(),
                "clock_time": pd.date_range(
                    start=self._from_time_config.start_timestamp,  # should be naive
                    periods=self._from_time_config.length,
                    freq=self._from_time_config.resolution,
                ),
            }
        )

        dfm["clock_time"] = dfm[self._to_time_col].dt.tz_localize(None)
        dfm = dfm.drop(self._from_time_col, axis=1)
        dfm = dfm.merge(df_idx_clock_time, on="clock_time")
        return dfm


class SimpleMappingGenerator(BaseMappingGenerator):
    """
    Creates a mapping dataframe for a mapping with a single to/from time zone,
    None is considered an accpetable time zone for this mapper
    """

    def _simple_mapping(self) -> pd.DataFrame:
        dfm = pd.DataFrame(
            {
                self._from_time_col: self._index_generator.list_timestamps(),
                self._intermediate_time_col: self._intermediate_datetime_generator.list_timestamps(),
            }
        )
        return dfm

    def _create_mapping(self) -> pd.DataFrame:
        dfm = self._simple_mapping()

        # TODO: Not tested, is this a case we support?
        if self._align_to_local_clocktime:
            df_idx_clock_time = pd.DataFrame(
                {
                    self._from_time_col: self._index_generator.list_timestamps(),
                    "clock_time": pd.date_range(
                        start=self._from_time_config.start_timestamp,  # should be naive
                        periods=self._from_time_config.length,
                        freq=self._from_time_config.resolution,
                    ),
                }
            )

            dfm["clock_time"] = dfm[self._to_time_col].dt.tz_localize(None)
            dfm = dfm.drop(self._from_time_col, axis=1)
            dfm = dfm.merge(df_idx_clock_time, on="clock_time")

        return dfm


class MultipleLocalMappingGenerator(BaseMappingGenerator):
    """
    Creates a mapping dataframe for input data that contains a time_zone column
    """

    def __init__(
        self,
        to_time_config: DatetimeRange,
        from_time_config: IndexTimeRange,
        to_time_col: str,
        from_time_col: str,
        time_based_data_adjustment: TimeBasedDataAdjustment,
        time_zones: list[str],
    ) -> None:
        super().__init__(
            to_time_config,
            from_time_config,
            to_time_col,
            from_time_col,
            time_based_data_adjustment,
        )
        self._time_zones = time_zones

    def _tz_adjusted_mapping(self, tzinfo: ZoneInfo) -> pd.DataFrame:
        self._intermediate_datetime_generator.set_tzinfo(tzinfo)
        dfm = pd.DataFrame(
            {
                self._from_time_col: self._index_generator.list_timestamps(),
                self._intermediate_time_col: self._intermediate_datetime_generator.list_timestamps(),
            }
        )
        return dfm

    def _create_mapping(self) -> pd.DataFrame:
        df_mtz = []
        for tz in self._time_zones:
            dfc = self._tz_adjusted_mapping(ZoneInfo(tz))
            # TODO is tz ever none in a the time_zone col?
            dfc[self._intermediate_time_col] = dfc[self._intermediate_time_col].dt.tz_convert(
                self._to_time_config.start.tzinfo
            )
            dfc["from_time_zone"] = tz
            df_mtz.append(dfc)

        dfm = pd.concat(df_mtz, ignore_index=True)
        return dfm


class MultipleLocalClocktimeMappingGenerator(BaseMappingGenerator):
    """Often refered to as Industrial Time."""

    def __init__(
        self,
        to_time_config: DatetimeRange,
        from_time_config: IndexTimeRange,
        to_time_col: str,
        from_time_col: str,
        time_based_data_adjustment: TimeBasedDataAdjustment,
        time_zones: list[str],
    ) -> None:
        super().__init__(
            to_time_config,
            from_time_config,
            to_time_col,
            from_time_col,
            time_based_data_adjustment,
        )
        self._time_zones = time_zones

    def _create_mapping(self) -> pd.DataFrame:
        return pd.DataFrame()


# class MapperIndexTimetoDatetimeIntermediate(MapperIndexTimeToDatetime):
#     """Maps to an intermediate schema that matches the from_schema, so all time points are represented"""
#
#     def __init__(
#         self,
#         engine: Engine,
#         metadata: MetaData,
#         from_schema: TableSchema,
#         to_schema: TableSchema,
#         time_based_data_adjustment: TimeBasedDataAdjustment | None = None,
#     ) -> None:
#         to_schema = self.create_intermediate_schema()
#
#         super().__init__(engine, metadata, from_schema, to_schema, time_based_data_adjustment)
#
#     def create_intermediate_schema(self) -> TableSchema:
#         time_kwargs = self._from_time_config.model_dump()
#         time_kwargs = dict(
#             filter(lambda k_v: k_v[0] in DatetimeRange.model_fields, time_kwargs.items())
#         )
#         time_kwargs["start"] = self._from_time_config.start_timestamp
#         time_kwargs["time_type"] = TimeType.DATETIME
#
#         time_config = DatetimeRange(**time_kwargs)
#
#         from_schema_kwargs = self._from_schema.model_dump()
#         from_schema_kwargs["name"] = "intermediate_mapping"
#         from_schema_kwargs["time_config"] = time_config
#
#         int_schema = TableSchema(**from_schema_kwargs)
#         return int_schema
#
#     @property
#     def is_unaligned(self):
#         """if time_zone is a time_array_id and start_timestamp is tz_naive then the timestamps are."""
#         return (
#             self._to_time_config.is_time_zone_naive()
#             and "time_zone" in self._to_schema.time_array_id_columns
#         )
#
#     def _create_multi_tz_unaligned_mapping_dataframe(
#         self, to_time_col: str, from_time_col: str, time_zones: list[str]
#     ) -> pd.DataFrame:
#         assert (
#             self._to_time_config.is_time_zone_naive()
#         ), "Can't create unaligned mapping with start timezone"
#         dfm = []
#         for tz in time_zones:
#             df_tz = pd.DataFrame(
#                 {
#                     to_time_col: pd.date_range(
#                         start=self._to_time_config.start.replace(tzinfo=ZoneInfo(tz)),
#                         periods=self._to_time_config.length,
#                         freq=self._to_time_config.resolution,
#                     ),
#                     from_time_col: self._index_generator.list_timestamps(),
#                     "from_time_zone": tz,
#                 }
#             )
#             dfm.append(df_tz)
#
#         return pd.concat(dfm, ignore_index=True)
#
#     def map_time(
#         self,
#         **kwargs,
#     ) -> None:
#         """Convert time columns with from_schema to to_schema configuration."""
#         if self.is_unaligned:
#             time_zones = self._list_time_zones()
#             to_time_col = self._to_time_config.time_column
#             from_time_col = "from_" + self._from_time_config.time_column
#             dfm = self._create_multi_tz_unaligned_mapping_dataframe(
#                 to_time_col, from_time_col, time_zones
#             )
#
#         else:
#             dfm = self._create_mapping()
#
#         self._map_time(dfm, **kwargs)
