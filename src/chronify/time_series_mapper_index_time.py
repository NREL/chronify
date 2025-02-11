import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import Engine, MetaData, Table, select

from chronify.models import TableSchema, MappingTableSchema
from chronify.exceptions import InvalidParameter, ConflictingInputsError
from chronify.time_series_mapper_base import TimeSeriesMapperBase, apply_mapping
from chronify.time_configs import (
    DatetimeRange,
    IndexTimeRanges,
    IndexTimeRangeBase,
    TimeBasedDataAdjustment,
)
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time_series_mapper_datetime import MapperDatetimeToDatetime
from chronify.time import TimeType, DaylightSavingAdjustmentType
from chronify.sqlalchemy.functions import read_database

logger = logging.getLogger(__name__)


class MapperIndexTimeToDatetime(TimeSeriesMapperBase):
    def __init__(
        self,
        engine: Engine,
        metadata: MetaData,
        from_schema: TableSchema,
        to_schema: TableSchema,
        data_adjustment: Optional[TimeBasedDataAdjustment] = None,
        wrap_time_allowed: bool = False,
    ) -> None:
        super().__init__(
            engine, metadata, from_schema, to_schema, data_adjustment, wrap_time_allowed
        )
        self._dst_adjustment = self._data_adjustment.daylight_saving_adjustment
        if not isinstance(self._from_schema.time_config, IndexTimeRangeBase):
            msg = "Source schema does not have IndexTimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        if not isinstance(self._to_schema.time_config, DatetimeRange):
            msg = "Destination schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        self._from_time_config: IndexTimeRanges = self._from_schema.time_config
        self._to_time_config: DatetimeRange = self._to_schema.time_config

    def check_schema_consistency(self) -> None:
        """Check that from_schema can produce to_schema."""
        self._check_table_columns_producibility()
        self._check_measurement_type_consistency()
        self._check_time_interval_type()
        self._check_time_resolution()
        self._check_time_length()

    def _check_time_resolution(self) -> None:
        if self._from_time_config.resolution != self._to_time_config.resolution:
            msg = "Handling of changing time resolution is not supported yet."
            raise NotImplementedError(msg)

    def _check_time_length(self) -> None:
        flen, tlen = self._from_time_config.length, self._to_time_config.length
        if flen != tlen and not self._wrap_time_allowed:
            msg = f"DatetimeRange length must match between from_schema and to_schema. {flen} vs. {tlen} OR wrap_time_allowed must be set to True"
            raise ConflictingInputsError(msg)

    def map_time(
        self,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        """Convert from index time to its represented datetime"""
        self.check_schema_consistency()

        # Convert from index time to its represented datetime
        if self._from_time_config.time_type == TimeType.INDEX_LOCAL:
            if (
                self._dst_adjustment
                == DaylightSavingAdjustmentType.DROP_SPRINGFORWARD_DUPLICATE_FALLBACK
            ):
                (
                    df,
                    mapping_schema,
                    mapped_schema,
                ) = self._create_interm_map_with_time_zone_and_dst_adj_drop_dup()
            elif (
                self._dst_adjustment
                == DaylightSavingAdjustmentType.DROP_SPRINGFORWARD_INTERPOLATE_FALLBACK
            ):
                msg = "Cannot use daylight saving adjustment type: DROP_SPRINGFORWARD_INTERPOLATE_FALLBACK for INDEX_LOCAL time"
                raise NotImplementedError(msg)
            else:
                (
                    df,
                    mapping_schema,
                    mapped_schema,
                ) = self._create_interm_map_with_time_zone()
        else:
            df, mapping_schema, mapped_schema = self._create_interm_map()
        apply_mapping(
            df,
            mapping_schema,
            self._from_schema,
            mapped_schema,
            self._engine,
            self._metadata,
            TimeBasedDataAdjustment(),
            scratch_dir=scratch_dir,
            output_file=output_file,
        )

        # Convert from represented datetime to dst time_config
        MapperDatetimeToDatetime(
            self._engine,
            self._metadata,
            mapped_schema,
            self._to_schema,
            self._data_adjustment,
            self._wrap_time_allowed,
        ).map_time(
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

    def _create_intermediate_schema(self) -> TableSchema:
        """Create the intermediate table schema for converting index time to its represented datetime"""
        time_kwargs = self._from_time_config.model_dump()
        time_kwargs = dict(
            filter(lambda k_v: k_v[0] in DatetimeRange.model_fields, time_kwargs.items())
        )
        time_kwargs["time_type"] = TimeType.DATETIME
        time_kwargs["start"] = self._from_time_config.start_timestamp
        time_kwargs["time_column"] = "represented_time"

        schema_kwargs = self._from_schema.model_dump()
        schema_kwargs["name"] += "_intermediate"
        if time_kwargs["time_zone_column"] is not None:
            schema_kwargs["time_array_id_columns"].append(time_kwargs["time_zone_column"])
            time_kwargs["time_zone_column"] = None

        schema_kwargs["time_config"] = DatetimeRange(**time_kwargs)
        schema = TableSchema(**schema_kwargs)

        return schema

    def _create_local_time_config(self, time_zone: str) -> DatetimeRange:
        """Create the intermediate time config for converting index local time to its represented datetime"""
        time_kwargs = self._from_time_config.model_dump()
        time_kwargs = dict(
            filter(lambda k_v: k_v[0] in DatetimeRange.model_fields, time_kwargs.items())
        )
        time_kwargs["time_type"] = TimeType.DATETIME
        time_kwargs["start"] = self._from_time_config.start_timestamp
        time_kwargs["time_column"] = "represented_time"
        time_kwargs["time_zone_column"] = None
        time_config = DatetimeRange(**time_kwargs)
        assert (
            time_config.start.tzinfo is None
        ), "Start time must be tz-naive for local time config"
        time_config.start = time_config.start.tz_localize(time_zone)  # type: ignore[attr-defined]

        return time_config

    def _create_interm_map(self) -> tuple[pd.DataFrame, MappingTableSchema, TableSchema]:
        """Create mapping dataframe for converting INDEX_TZ or INDEX_NTZ time to its represented datetime"""
        mapped_schema = self._create_intermediate_schema()
        assert isinstance(mapped_schema.time_config, DatetimeRange)
        mapped_time_col = mapped_schema.time_config.time_column
        mapped_time_data = make_time_range_generator(mapped_schema.time_config).list_timestamps()

        from_time_col = "from_" + self._from_time_config.time_column
        from_time_data = make_time_range_generator(self._from_time_config).list_timestamps()

        from_time_config = self._from_time_config.model_copy()
        from_time_config.time_column = from_time_col
        mapping_schema = MappingTableSchema(
            name="mapping_table",
            time_configs=[from_time_config, mapped_schema.time_config],
        )

        df = pd.DataFrame(
            {
                from_time_col: from_time_data,
                mapped_time_col: mapped_time_data,
            }
        )
        return df, mapping_schema, mapped_schema

    def _create_interm_map_with_time_zone(
        self,
    ) -> tuple[pd.DataFrame, MappingTableSchema, TableSchema]:
        """Create mapping dataframe for converting INDEX_LOCAL time to its represented datetime"""
        mapped_schema = self._create_intermediate_schema()
        assert isinstance(mapped_schema.time_config, DatetimeRange)
        mapped_time_col = mapped_schema.time_config.time_column

        from_time_col = "from_" + self._from_time_config.time_column
        from_time_data = make_time_range_generator(self._from_time_config).list_timestamps()

        tz_col_list = self._from_time_config.list_time_zone_column()
        assert tz_col_list != [], "Expecting a time zone column for INDEX_LOCAL"
        tz_col = tz_col_list[0]
        from_tz_col = "from_" + tz_col

        with self._engine.connect() as conn:
            table = Table(self._from_schema.name, self._metadata)
            stmt = select(table.c[tz_col]).distinct().where(table.c[tz_col].is_not(None))
            time_zones = read_database(stmt, conn, self._from_time_config)[tz_col].to_list()

        from_time_config = self._from_time_config.model_copy()
        from_time_config.time_column = from_time_col
        from_time_config.time_zone_column = from_tz_col

        df_tz = []
        to_tz = self._to_time_config.start.tzinfo
        for time_zone in time_zones:
            config_tz = self._create_local_time_config(time_zone)
            time_data = make_time_range_generator(config_tz).list_timestamps()
            # Preemptively convert to dst time tzinfo, otherwise pandas treats the col,
            # which consists of the timeseries of different time zones, as an object col
            mapped_time_data = [x.tz_convert(to_tz) for x in time_data]
            df_tz.append(
                pd.DataFrame(
                    {
                        from_time_col: from_time_data,
                        from_tz_col: time_zone,
                        mapped_time_col: mapped_time_data,
                    }
                )
            )
        df = pd.concat(df_tz, ignore_index=True)

        # Update mapped_schema
        mapped_schema.time_config.start = df[mapped_time_col].min()
        mapped_schema.time_config.length = df[mapped_time_col].nunique()

        mapping_schema = MappingTableSchema(
            name="mapping_table",
            time_configs=[from_time_config, mapped_schema.time_config],
        )
        return df, mapping_schema, mapped_schema

    def _create_interm_map_with_time_zone_and_dst_adj_drop_dup(
        self,
    ) -> tuple[pd.DataFrame, MappingTableSchema, TableSchema]:
        """Create mapping dataframe for converting INDEX_LOCAL time to its represented datetime
        with time-based daylight_saving adjustment that
        drops the spring-forward hour and duplicates the fall-back hour
        """
        mapped_schema = self._create_intermediate_schema()
        assert isinstance(mapped_schema.time_config, DatetimeRange)
        mapped_time_col = mapped_schema.time_config.time_column
        mapped_time_data_ntz = make_time_range_generator(
            mapped_schema.time_config
        ).list_timestamps()

        from_time_col = "from_" + self._from_time_config.time_column
        from_time_data = make_time_range_generator(self._from_time_config).list_timestamps()

        df_ntz = pd.DataFrame(
            {
                from_time_col: from_time_data,
                "clock_time": mapped_time_data_ntz,
            }
        )
        df_ntz["clock_time"] = df_ntz["clock_time"].astype(str)

        tz_col_list = self._from_time_config.list_time_zone_column()
        assert tz_col_list != [], "Expecting a time zone column for INDEX_LOCAL"
        tz_col = tz_col_list[0]
        from_tz_col = "from_" + tz_col

        with self._engine.connect() as conn:
            table = Table(self._from_schema.name, self._metadata)
            stmt = select(table.c[tz_col]).distinct().where(table.c[tz_col].is_not(None))
            time_zones = read_database(stmt, conn, self._from_time_config)[tz_col].to_list()

        from_time_config = self._from_time_config.model_copy()
        from_time_config.time_column = from_time_col
        from_time_config.time_zone_column = from_tz_col

        df_tz = []
        to_tz = self._to_time_config.start.tzinfo
        for time_zone in time_zones:
            config_tz = self._create_local_time_config(time_zone)
            time_data = make_time_range_generator(config_tz).list_timestamps()
            # Extract clock time
            clock_time_data = [x.strftime("%Y-%m-%d %H:%M:%S") for x in time_data]
            # Preemptively convert to dst time tzinfo, otherwise pandas treats the col,
            # which consists of the timeseries of different time zones, as an object col
            mapped_time_data = [x.tz_convert(to_tz) for x in time_data]
            df_tz.append(
                pd.DataFrame(
                    {
                        from_tz_col: time_zone,
                        "clock_time": clock_time_data,  # str
                        mapped_time_col: mapped_time_data,
                    }
                )
            )
        df = pd.concat(df_tz, ignore_index=True)
        df = df.merge(df_ntz, on="clock_time").drop(columns=["clock_time"])

        # Update mapped_schema
        mapped_schema.time_config.start = df[mapped_time_col].min()
        mapped_schema.time_config.length = df[mapped_time_col].nunique()

        mapping_schema = MappingTableSchema(
            name="mapping_table",
            time_configs=[from_time_config, mapped_schema.time_config],
        )
        return df, mapping_schema, mapped_schema
