import logging
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import Engine, MetaData, Table, select


from chronify.datetime_range_generator import DatetimeRangeGenerator
from chronify.models import TableSchema, MappingTableSchema
from chronify.sqlalchemy.functions import read_database
from chronify.exceptions import (
    InvalidParameter,
)
from chronify.time import TimeType
from chronify.time_series_mapper_base import TimeSeriesMapperBase, apply_mapping
from chronify.index_time_range_generator import IndexTimeRangeGenerator
from chronify.time_configs import (
    DatetimeRange,
    DaylightSavingAdjustment,
    IndexTimeRange,
    TimeBasedDataAdjustment,
)
from chronify.time_series_mapper_datetime import MapperDatetimeToDatetime


logger = logging.getLogger(__name__)


class MapperIndexTimeToDatetime(TimeSeriesMapperBase):
    def __init__(
        self,
        engine: Engine,
        metadata: MetaData,
        from_schema: TableSchema,
        to_schema: TableSchema,
        time_based_data_adjustment: TimeBasedDataAdjustment | None = None,
        aligned: bool = True,
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

        aligned : bool
            if True, assume the data is aligned in absolute time,
            if False, use DatatimeToDatetime mapping to align time with shifting and wrapping
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

        # if not aligned, map to intermediate schema and then use Datetime to Datetime
        if aligned:
            self._to_schema = to_schema
            self._datetime_mapper = None
        else:
            self._to_schema = self.create_intermediate_schema()
            self._to_time_config: DatetimeRange = self._to_schema.time_config  # type: ignore
            self._datetime_mapper = MapperDatetimeToDatetime(
                engine, metadata, self._to_schema, to_schema
            )

        self._engine = engine
        self._metadata = metadata
        self._time_based_data_adjustment = time_based_data_adjustment or TimeBasedDataAdjustment()
        self._aligned = aligned

        self._index_generator = IndexTimeRangeGenerator(self._from_time_config)

        # adjusts for leapyear, but no dst
        self._datetime_generator = DatetimeRangeGenerator(
            self._to_time_config, self._time_based_data_adjustment
        )

        self._tz_naive: tuple[bool, bool] = (
            self._from_time_config.is_time_zone_naive(),
            self._to_time_config.is_time_zone_naive(),
        )

        # self._not_supported_mappings()

    def check_schema_consistency(self) -> None:
        # TODO: fail for interpolate fall_back_hour
        pass

    @property
    def _align_local_clocktime(self) -> bool:
        """Formerly known as Industrial time, flag indicating index time should map to a local clock time"""
        return (
            self._from_time_config.is_time_zone_naive()
            and self._time_based_data_adjustment.daylight_saving_adjustment
            == DaylightSavingAdjustment(spring_forward_hour="drop", fall_back_hour="duplicate")
        )

    def create_intermediate_schema(self) -> TableSchema:
        time_kwargs = self._from_time_config.model_dump()
        time_kwargs = dict(
            filter(lambda k_v: k_v[0] in DatetimeRange.model_fields, time_kwargs.items())
        )
        time_kwargs["start"] = self._from_time_config.start_timestamp
        time_kwargs["time_type"] = TimeType.DATETIME

        time_config = DatetimeRange(**time_kwargs)

        from_schema_kwargs = self._from_schema.model_dump()
        from_schema_kwargs["name"] = "intermediate_mapping"
        from_schema_kwargs["time_config"] = time_config

        int_schema = TableSchema(**from_schema_kwargs)
        return int_schema

    def map_time(
        self,
        check_mapped_timestamps: bool = False,
        **kwargs,
    ) -> None:
        """Convert time columns with from_schema to to_schema configuration."""
        self.check_schema_consistency()

        map_table_name = "map_table"
        dfm = self._create_mapping()
        mapping_schema = self._create_mapping_schema(map_table_name)

        # if using datetime mapper, check timestamps in that mapper
        tmp_check_mapped_timestamps = self._datetime_mapper is None

        apply_mapping(
            dfm,
            mapping_schema,
            self._from_schema,
            self._to_schema,
            self._engine,
            self._metadata,
            check_mapped_timestamps=tmp_check_mapped_timestamps,
            **kwargs,
        )

        if self._datetime_mapper is not None:
            self._datetime_mapper.map_time(
                check_mapped_timestamps=check_mapped_timestamps, **kwargs
            )

            # TODO drop table with self._to_schema.name

    def create_single_tz_mapping_dataframe(
        self, to_time_col: str, from_time_col: str
    ) -> pd.DataFrame:
        """
        Creates a dataframe mapping by subtracting timestamps from the start_timestamp of
        the index_time schema.

        timestamps and start_timestamp need to either both be timezone aware or both be timezone naive
        """

        dfm = pd.DataFrame(
            {
                from_time_col: self._index_generator.list_timestamps(),
                to_time_col: self._datetime_generator.list_timestamps(),
            }
        )

        if self._align_local_clocktime:
            df_idx_clock_time = pd.DataFrame(
                {
                    from_time_col: self._index_generator.list_timestamps(),
                    "clock_time": pd.date_range(
                        start=self._from_time_config.start_timestamp,  # should be naive
                        periods=self._from_time_config.length,
                        freq=self._from_time_config.resolution,
                    ),
                }
            )

            dfm["clock_time"] = dfm[to_time_col].dt.tz_localize(None)
            dfm = dfm.drop(from_time_col, axis=1)
            dfm = dfm.merge(df_idx_clock_time, on="clock_time")

        return dfm

    def create_multi_tz_aligned_mapping_dataframe(
        self, to_time_col: str, from_time_col: str, time_zones: list[str]
    ) -> pd.DataFrame:
        # Create mapping using from/to schema
        dfm = self.create_single_tz_mapping_dataframe(to_time_col, from_time_col)
        df_mtz = []
        for tz in time_zones:
            dfc = dfm.copy()
            dfc[to_time_col] = dfc[to_time_col].dt.tz_convert(tz)
            dfc["from_time_zone"] = tz
            df_mtz.append(dfc)
        return pd.concat(df_mtz, ignore_index=True)

    def create_multi_tz_unaligned_mapping_dataframe(
        self, to_time_col: str, from_time_col: str, time_zones: list[str]
    ) -> pd.DataFrame:
        dfm = []
        for tz in time_zones:
            df_tz = pd.DataFrame(
                {
                    to_time_col: pd.date_range(
                        start=self._to_time_config.start.replace(tzinfo=ZoneInfo(tz)),
                        periods=self._to_time_config.length,
                        freq=self._to_time_config.resolution,
                    ),
                    from_time_col: self._index_generator.list_timestamps(),
                    "from_time_zone": tz,
                }
            )
            dfm.append(df_tz)

        return pd.concat(dfm, ignore_index=True)

    def _create_mapping(self) -> pd.DataFrame:
        """Create mapping dataframe"""

        to_time_col = self._to_time_config.time_column
        from_time_col = "from_" + self._from_time_config.time_column
        self._metadata.reflect(self._engine)

        if "time_zone" in self._from_schema.time_array_id_columns:
            with self._engine.connect() as conn:
                table = Table(self._from_schema.name, self._metadata)
                stmt = (
                    select(table.c["time_zone"])
                    .distinct()
                    .where(table.c["time_zone"].is_not(None))
                )
                time_zones = read_database(stmt, conn, self._from_time_config)[
                    "time_zone"
                ].to_list()
            if self._from_time_config.is_time_zone_naive():
                dfm = self.create_multi_tz_unaligned_mapping_dataframe(
                    to_time_col, from_time_col, time_zones
                )
            else:
                dfm = self.create_multi_tz_aligned_mapping_dataframe(
                    to_time_col, from_time_col, time_zones
                )
        else:
            dfm = self.create_single_tz_mapping_dataframe(to_time_col, from_time_col)

        return dfm

    def _create_mapping_schema(self, table_name: str):
        return MappingTableSchema(name=table_name, time_configs=[self._to_time_config])
