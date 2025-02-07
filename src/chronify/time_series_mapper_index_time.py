import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import Engine, MetaData

from chronify.models import TableSchema, MappingTableSchema
from chronify.exceptions import InvalidParameter, ConflictingInputsError
from chronify.time_series_mapper_base import TimeSeriesMapperBase, apply_mapping
from chronify.time_configs import DatetimeRange, IndexTimeRangeBase, TimeBasedDataAdjustment
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time_series_mapper_datetime import MapperDatetimeToDatetime
from chronify.time import TimeType

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
        if not isinstance(self._from_time_config, IndexTimeRangeBase):
            msg = "Source schema does not have IndexTimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        if not isinstance(self._to_time_config, DatetimeRange):
            msg = "Destination schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)

    def check_schema_consistency(self) -> None:
        """Check that from_schema can produce to_schema."""
        self._check_table_columns_producibility()
        self._check_measurement_type_consistency()
        self._check_time_interval_type()
        self._check_time_resolution_and_length()

    def _check_time_resolution_and_length(self) -> None:
        if self._from_time_config.resolution != self._to_time_config.resolution:
            msg = "Handling of changing time resolution is not supported yet."
            raise NotImplementedError(msg)

        flen, tlen = self._from_time_config.length, self._to_time_config.length
        if flen != tlen:
            msg = f"DatetimeRange length must match between from_schema and to_schema. {flen} vs. {tlen}"
            raise ConflictingInputsError(msg)

    def map_time(
        self,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        """Convert from index time to its represented datetime"""
        # self.check_schema_consistency()

        # Convert from index time to its represented datetime
        df, mapping_schema, mapped_schema = self._create_intermediate_mapping()
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
        """Create the intermediate schema for converting index time to its represented datetime"""
        time_kwargs = self._from_time_config.model_dump()
        time_kwargs = dict(
            filter(lambda k_v: k_v[0] in DatetimeRange.model_fields, time_kwargs.items())
        )
        time_kwargs["time_type"] = TimeType.DATETIME
        time_kwargs["start"] = self._from_time_config.start_timestamp
        time_kwargs["time_column"] = "represented_time"
        time_config = DatetimeRange(**time_kwargs)

        schema_kwargs = self._from_schema.model_dump()
        schema_kwargs["name"] += "_intermediate"
        schema_kwargs["time_config"] = time_config
        schema = TableSchema(**schema_kwargs)

        return schema

    def _create_intermediate_mapping(self) -> tuple[pd.DataFrame, MappingTableSchema, TableSchema]:
        """Create mapping dataframe for converting index time to its represented datetime"""
        from_time_col = "from_" + self._from_time_config.time_column
        from_time_data = make_time_range_generator(self._from_time_config).list_timestamps()

        mapped_schema = self._create_intermediate_schema()
        mapped_time_col = mapped_schema.time_config.time_column
        mapped_time_data = make_time_range_generator(mapped_schema.time_config).list_timestamps()

        df = pd.DataFrame(
            {
                from_time_col: from_time_data,
                mapped_time_col: mapped_time_data,
            }
        )

        from_time_config = self._from_time_config.model_copy()
        from_time_config.time_column = from_time_col
        mapping_schema = MappingTableSchema(
            name="mapping_table",
            time_configs=[from_time_config, mapped_schema.time_config],
        )
        return df, mapping_schema, mapped_schema
