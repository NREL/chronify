import logging

import pandas as pd
from sqlalchemy import Engine, MetaData

from chronify.models import TableSchema
from chronify.exceptions import (
    ConflictingInputsError,
    InvalidParameter,
)
from chronify.time_series_mapper_base import TimeSeriesMapperBase, apply_mapping
from chronify.time_configs import DatetimeRange
from chronify.time import TimeIntervalType
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time_utils import roll_time_interval

logger = logging.getLogger(__name__)


class MapperDatetimeToDatetime(TimeSeriesMapperBase):
    def __init__(
        self, engine: Engine, metadata: MetaData, from_schema: TableSchema, to_schema: TableSchema
    ) -> None:
        if not isinstance(from_schema.time_config, DatetimeRange):
            msg = "source schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        if not isinstance(to_schema.time_config, DatetimeRange):
            msg = "destination schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        self._from_time_config = from_schema.time_config
        self._to_time_config = to_schema.time_config
        self._from_schema = from_schema
        self._to_schema = to_schema
        self._engine = engine
        self._metadata = metadata

    def check_schema_consistency(self) -> None:
        """Check that from_schema can produce to_schema."""
        self._check_table_columns_producibility()
        self._check_measurement_type_consistency()
        self._check_time_interval_type()

    def _check_table_columns_producibility(self) -> None:
        """Check columns in destination table can be produced by source table."""
        available_cols = self._from_schema.list_columns() + [self._to_time_config.time_column]
        final_cols = self._to_schema.list_columns()
        if diff := set(final_cols) - set(available_cols):
            msg = f"Source table {self._from_schema.name} cannot produce the columns: {diff}"
            raise ConflictingInputsError(msg)

    def _check_measurement_type_consistency(self) -> None:
        """Check that measurement_type is the same between schema."""
        from_mt = self._from_schema.time_config.measurement_type
        to_mt = self._to_schema.time_config.measurement_type
        if from_mt != to_mt:
            msg = f"Inconsistent measurement_types {from_mt=} vs. {to_mt=}"
            raise ConflictingInputsError(msg)

    def _check_time_interval_type(self) -> None:
        """Check time interval type consistency."""
        from_interval = self._from_time_config.interval_type
        to_interval = self._from_time_config.interval_type
        if TimeIntervalType.INSTANTANEOUS in (from_interval, to_interval) and (
            from_interval != to_interval
        ):
            msg = "If instantaneous time interval is used, it must exist in both from_scheme and to_schema."
            raise ConflictingInputsError(msg)

    def _check_time_resolution(self) -> None:
        if self._from_time_config.resolution != self._to_time_config.resolution:
            msg = "Handling of changing time resolution is not supported yet."
            raise NotImplementedError(msg)

    def map_time(self) -> None:
        """Convert time columns with from_schema to to_schema configuration."""
        if self._from_schema == self._to_schema:
            msg = "From table schema is the same as to table schema. Nothing to do.\n{self._from_schema}"
            logger.info(msg)
            return
        self.check_schema_consistency()

        map_table_name = "mapping_table"
        df = self._create_mapping()
        apply_mapping(
            df, map_table_name, self._from_schema, self._to_schema, self._engine, self._metadata
        )
        # TODO - add handling for changing resolution

    def _create_mapping(self) -> pd.DataFrame:
        """Create mapping dataframe
        Handles time interval type
        """
        from_time_col = "from_" + self._from_time_config.time_column
        to_time_col = self._to_time_config.time_column
        to_time_data = make_time_range_generator(self._to_time_config).list_timestamps()
        df = pd.DataFrame(
            {
                from_time_col: make_time_range_generator(self._from_time_config).list_timestamps(),
                to_time_col: to_time_data,
            }
        )
        if self._from_time_config.interval_type != self._to_time_config.interval_type:
            df[to_time_col] = roll_time_interval(
                df[to_time_col],
                self._from_time_config.interval_type,
                self._to_time_config.interval_type,
            )
        return df
