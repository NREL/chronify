import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import Engine, MetaData

from chronify.models import TableSchema, MappingTableSchema
from chronify.exceptions import InvalidParameter, ConflictingInputsError
from chronify.time_series_mapper_base import TimeSeriesMapperBase, apply_mapping
from chronify.time_configs import DatetimeRange, TimeBasedDataAdjustment
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time_utils import roll_time_interval

logger = logging.getLogger(__name__)


class MapperDatetimeToDatetime(TimeSeriesMapperBase):
    def __init__(
        self,
        engine: Engine,
        metadata: MetaData,
        from_schema: TableSchema,
        to_schema: TableSchema,
        data_adjustment: Optional[TimeBasedDataAdjustment] = None,
    ) -> None:
        super().__init__(engine, metadata, from_schema, to_schema, data_adjustment)
        if self._from_schema == self._to_schema and self._data_adjustment is None:
            msg = f"from_schema is the same as to_schema and no data_adjustment, nothing to do.\n{self._from_schema}"
            logger.info(msg)
            return
        if not isinstance(self._from_time_config, DatetimeRange):
            msg = "Source schema does not have DatetimeRange time config. Use a different mapper."
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
        """Convert time columns with from_schema to to_schema configuration."""
        self.check_schema_consistency()
        df, mapping_schema = self._create_mapping()
        apply_mapping(
            df,
            mapping_schema,
            self._from_schema,
            self._to_schema,
            self._engine,
            self._metadata,
            self._data_adjustment,
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )
        # TODO - add handling for changing resolution - Issue #30

    def _create_mapping(self) -> tuple[pd.DataFrame, MappingTableSchema]:
        """Create mapping dataframe
        Handles time interval type
        """
        from_time_col = "from_" + self._from_time_config.time_column
        to_time_col = self._to_time_config.time_column
        to_time_data = make_time_range_generator(
            self._to_time_config, leap_day_adjustment=self._data_adjustment.leap_day_adjustment
        ).list_timestamps()
        df = pd.DataFrame(
            {
                from_time_col: make_time_range_generator(self._from_time_config).list_timestamps(),
            }
        )
        if self._from_time_config.interval_type != self._to_time_config.interval_type:
            # If from_tz or to_tz is naive, use tz_localize
            fm_tz = self._from_time_config.start.tzinfo
            to_tz = self._to_time_config.start.tzinfo
            from_timestamps = df[from_time_col]
            if None in (fm_tz, to_tz) and (fm_tz, to_tz) != (None, None):
                from_timestamps = from_timestamps.dt.tz_localize(to_tz)

            df[to_time_col] = roll_time_interval(
                from_timestamps,
                self._from_time_config.interval_type,
                self._to_time_config.interval_type,
                to_time_data,
            )
        else:
            df[to_time_col] = to_time_data

        from_time_config = self._from_time_config.model_copy()
        from_time_config.time_column = from_time_col
        mapping_schema = MappingTableSchema(
            name="mapping_table",
            time_configs=[
                from_time_config,
                self._to_time_config,
            ],
        )
        return df, mapping_schema
