import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import Engine, MetaData, Table, select

from chronify.sqlalchemy.functions import read_database
from chronify.models import TableSchema, MappingTableSchema
from chronify.exceptions import (
    MissingParameter,
    InvalidParameter,
)
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time_series_mapper_base import TimeSeriesMapperBase, apply_mapping
from chronify.representative_time_range_generator import RepresentativePeriodTimeGenerator
from chronify.time_configs import DatetimeRange, RepresentativePeriodTime
from chronify.time_utils import shift_time_interval

logger = logging.getLogger(__name__)


class MapperRepresentativeTimeToDatetime(TimeSeriesMapperBase):
    def __init__(
        self, engine: Engine, metadata: MetaData, from_schema: TableSchema, to_schema: TableSchema
    ) -> None:
        super().__init__(engine, metadata, from_schema, to_schema)
        if not isinstance(from_schema.time_config, RepresentativePeriodTime):
            msg = "source schema does not have RepresentativePeriodTime time config. Use a different mapper."
            raise InvalidParameter(msg)
        if not isinstance(to_schema.time_config, DatetimeRange):
            msg = "destination schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        self._generator = RepresentativePeriodTimeGenerator(from_schema.time_config)
        self._from_time_config = from_schema.time_config
        self._to_time_config = to_schema.time_config

    def check_schema_consistency(self) -> None:
        """Check that from_schema can produce to_schema."""
        self._check_table_columns_producibility()
        self._check_measurement_type_consistency()
        self._check_time_interval_type()

    def _check_source_table_has_time_zone(self) -> None:
        """Check source table has time_zone column."""
        if "time_zone" not in self._from_schema.list_columns():
            msg = f"time_zone is required for tz-aware representative time mapping and must be part of the time_array_id_columns for source table: {self._from_schema.name}"
            raise MissingParameter(msg)

    def map_time(
        self,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        """Convert time columns with from_schema to to_schema configuration."""
        is_tz_naive = self._to_time_config.is_time_zone_naive()
        self.check_schema_consistency()
        if not is_tz_naive:
            self._check_source_table_has_time_zone()

        df, mapping_schema = self._create_mapping(is_tz_naive)
        apply_mapping(
            df,
            mapping_schema,
            self._from_schema,
            self._to_schema,
            self._engine,
            self._metadata,
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

    def _create_mapping(self, is_tz_naive: bool) -> tuple[pd.DataFrame, MappingTableSchema]:
        """Create mapping dataframe
        - Handles time interval type adjustment
        - Columns used to join the from_table are prefixed with "from_"
        """
        timestamp_generator = make_time_range_generator(self._to_time_config)

        to_time_col = self._to_time_config.time_column
        dft = pd.Series(timestamp_generator.list_timestamps()).rename(to_time_col).to_frame()

        if self._from_time_config.interval_type != self._to_time_config.interval_type:
            time_col = "to_" + to_time_col
            # Mapping works backward for representative time by
            # shifting interval type of to_time_config to match
            # from_time_config before extracting time info
            dft[time_col] = shift_time_interval(
                dft[to_time_col],
                self._to_time_config.interval_type,
                self._from_time_config.interval_type,
            )
        else:
            time_col = to_time_col

        from_columns = self._from_time_config.list_time_columns()
        if is_tz_naive:
            df = self._generator.create_tz_naive_mapping_dataframe(dft, time_col)
        else:
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
            df = self._generator.create_tz_aware_mapping_dataframe(dft, time_col, time_zones)
            from_columns.append("time_zone")

        prefixed_from_columns = ["from_" + x for x in from_columns]
        df = df.rename(columns=dict(zip(from_columns, prefixed_from_columns)))

        if time_col != to_time_col:
            df.drop(time_col, axis=1, inplace=True)

        mapping_schema = MappingTableSchema(
            name="mapping_table",
            time_configs=[
                self._to_time_config,  # only DatetimeRange
            ],
        )
        return df, mapping_schema
