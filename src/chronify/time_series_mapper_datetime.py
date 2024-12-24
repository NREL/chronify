import logging

import pandas as pd
from sqlalchemy import Engine, MetaData

from chronify.models import TableSchema, MappingTableSchema
from chronify.exceptions import (
    InvalidParameter,
)
from chronify.time_series_mapper_base import TimeSeriesMapperBase, apply_mapping
from chronify.time_configs import DatetimeRange
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time_utils import roll_time_interval

logger = logging.getLogger(__name__)


class MapperDatetimeToDatetime(TimeSeriesMapperBase):
    def __init__(
        self, engine: Engine, metadata: MetaData, from_schema: TableSchema, to_schema: TableSchema
    ) -> None:
        super().__init__(engine, metadata, from_schema, to_schema)
        if from_schema == to_schema:
            msg = (
                f"From table schema is the same as to table schema. Nothing to do.\n{from_schema}"
            )
            logger.info(msg)
            return
        if not isinstance(from_schema.time_config, DatetimeRange):
            msg = "Source schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        if not isinstance(to_schema.time_config, DatetimeRange):
            msg = "Destination schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        self._from_time_config = from_schema.time_config
        self._to_time_config = to_schema.time_config

    def _check_time_resolution(self) -> None:
        if self._from_time_config.resolution != self._to_time_config.resolution:
            msg = "Handling of changing time resolution is not supported yet."
            raise NotImplementedError(msg)

    def map_time(self) -> None:
        """Convert time columns with from_schema to to_schema configuration."""
        self.check_schema_consistency()
        df, mapping_schema = self._create_mapping()
        apply_mapping(
            df, mapping_schema, self._from_schema, self._to_schema, self._engine, self._metadata
        )
        # TODO - add handling for changing resolution - Issue #30

    def _create_mapping(self) -> tuple[pd.DataFrame, MappingTableSchema]:
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

        from_time_config = self._from_time_config.model_copy()
        from_time_config.time_column = from_time_col
        mapping_schema = MappingTableSchema(
            name="mapping_table",
            time_configs=[
                from_time_config,
                self._to_time_config,
            ],
            other_columns=[],
        )
        return df, mapping_schema
