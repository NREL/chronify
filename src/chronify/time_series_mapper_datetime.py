import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
from sqlalchemy import Engine, MetaData

from chronify.models import TableSchema, MappingTableSchema
from chronify.exceptions import InvalidParameter, ConflictingInputsError
from chronify.time_series_mapper_base import TimeSeriesMapperBase, apply_mapping
from chronify.time_configs import DatetimeRange, TimeBasedDataAdjustment
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time_utils import roll_time_interval, wrap_timestamps
from chronify.time import ResamplingOperationType, AggregationType, DisaggregationType

logger = logging.getLogger(__name__)


class MapperDatetimeToDatetime(TimeSeriesMapperBase):
    def __init__(
        self,
        engine: Engine,
        metadata: MetaData,
        from_schema: TableSchema,
        to_schema: TableSchema,
        data_adjustment: Optional[TimeBasedDataAdjustment] = None,
        wrap_time_allowed: bool = False,
        resampling_operation: Optional[ResamplingOperationType] = None,
    ) -> None:
        super().__init__(
            engine,
            metadata,
            from_schema,
            to_schema,
            data_adjustment,
            wrap_time_allowed,
            resampling_operation,
        )
        if self._from_schema == self._to_schema and self._data_adjustment is None:
            msg = f"from_schema is the same as to_schema and no data_adjustment, nothing to do.\n{self._from_schema}"
            logger.info(msg)
        if not isinstance(self._from_schema.time_config, DatetimeRange):
            msg = "Source schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        if not isinstance(self._to_schema.time_config, DatetimeRange):
            msg = "Destination schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        self._from_time_config: DatetimeRange = self._from_schema.time_config
        self._to_time_config: DatetimeRange = self._to_schema.time_config
        self._resampling_type = self.get_resampling_type()

    def get_resampling_type(self) -> Optional[str]:
        if self._from_time_config.resolution < self._to_time_config.resolution:
            return "aggregation"
        if self._from_time_config.resolution > self._to_time_config.resolution:
            return "disaggregation"
        return None

    def check_schema_consistency(self) -> None:
        """Check that from_schema can produce to_schema."""
        self._check_table_columns_producibility()
        self._check_measurement_type_consistency()
        self._check_time_interval_type()
        self._check_time_resolution_and_length()
        self._check_resampling_consistency()

    def _check_time_resolution_and_length(self) -> None:
        """Check that from_schema time resolution and length can produce to_schema counterparts."""
        flen, tlen = self._from_time_config.length, self._to_time_config.length
        fres, tres = self._from_time_config.resolution, self._to_time_config.resolution
        if fres == tres and flen != tlen and not self._wrap_time_allowed:
            msg = f"For unchanging time resolution, length must match between {self._from_schema.__class__} from_schema and {self._to_schema.__class__} to_schema. {flen} vs. {tlen} OR wrap_time_allowed must be set to True"
            raise ConflictingInputsError(msg)

        # Resolution must be mutiples of each other
        if fres < tres and tres % fres > 0:
            msg = f"For aggregation, the time resolution in {self._to_schema.__class__} to_schema must be a multiple of that in {self._from_schema.__class__} from_schema. {tres} vs {fres}"
        if fres > tres and fres % tres > 0:
            msg = f"For disaggregation, the time resolution in {self._from_schema.__class__} from_schema must be a multiple of that in {self._to_schema.__class__} to_schema. {fres} vs {tres}"

        # No extrapolation allowed
        if flen * fres < tlen * tres:
            msg = f"The product of time resolution and length in {self._from_schema.__class__} from_schema cannot be greater than that in {self._to_schema.__class__} to_schema."
            raise ConflictingInputsError(msg)

    def _check_resampling_consistency(self) -> None:
        """Check resampling operation type is consistent with time resolution inputs."""
        if self._resampling_type is None and self._resampling_operation is not None:
            msg = f"For unchanging time resolution, {self._resampling_operation=} must be set to None."
            raise ConflictingInputsError(msg)

        for typ, opt in [("aggregation", AggregationType), ("disaggregation", DisaggregationType)]:
            if self._resampling_type == typ and not isinstance(self._resampling_operation, opt):
                msg = f"{typ} detected, {self._resampling_operation=} must be set to an option from {opt}"
                raise ConflictingInputsError(msg)

    def map_time(
        self,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        """Convert time columns with from_schema to to_schema configuration."""
        if self._resampling_type:
            to_schema = self._to_schema.model_copy(deep=True)
            to_schema.time_config.resolution = self._from_time_config.resolution
            to_schema.time_config.length = self._from_time_config.length
            to_time_config = to_schema.time_config
        else:
            to_schema = self._to_schema
            to_time_config = self._to_time_config

        self.check_schema_consistency()
        df, mapping_schema = self._create_mapping(to_time_config)
        apply_mapping(
            df,
            mapping_schema,
            self._from_schema,
            to_schema,
            self._engine,
            self._metadata,
            self._data_adjustment,
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )
        if self._resampling_type == "aggregation":
            df, mapping_schema = self._create_aggregation_mapping(
                to_time_config, self._to_time_config
            )
        # elif self._resampling_type == "disaggregation":
        #     df, mapping_schema = self._create_disaggregation_mapping(to_time_config, self._to_time_config)
        # TODO - add handling for changing resolution - Issue #30

    def _create_mapping(
        self, to_time_config: DatetimeRange
    ) -> tuple[pd.DataFrame, MappingTableSchema]:
        """Create mapping dataframe
        Handles time interval type but not resolution
        """
        from_time_col = "from_" + self._from_time_config.time_column
        to_time_col = to_time_config.time_column
        from_time_data = make_time_range_generator(self._from_time_config).list_timestamps()
        to_time_data = make_time_range_generator(
            to_time_config, leap_day_adjustment=self._data_adjustment.leap_day_adjustment
        ).list_timestamps()

        ser_from = pd.Series(from_time_data)
        # If from_tz or to_tz is naive, use tz_localize
        fm_tz = self._from_time_config.start.tzinfo
        to_tz = to_time_config.start.tzinfo
        match (fm_tz is None, to_tz is None):
            case (True, False):
                # get standard time zone of to_tz
                year = to_time_config.start.year
                to_tz_std = datetime(year=year, month=1, day=1, tzinfo=to_tz).tzname()
                # tz-naive time does not have skips/dups, so always localize in std tz first
                ser_from = ser_from.dt.tz_localize(to_tz_std).dt.tz_convert(to_tz)
            case (False, True):
                # get standard time zone of fm_tz
                year = self._from_time_config.start.year
                fm_tz_std = datetime(year=year, month=1, day=1, tzinfo=fm_tz).tzname()
                # convert to standard time zone of fm_tz before making it tz-naive
                ser_from = ser_from.dt.tz_convert(fm_tz_std).dt.tz_localize(to_tz)
        match (self._adjust_interval, self._wrap_time_allowed):
            case (True, _):
                ser = roll_time_interval(
                    ser_from,
                    self._from_time_config.interval_type,
                    to_time_config.interval_type,
                    to_time_data,
                )
            case (False, True):
                ser = wrap_timestamps(ser_from, to_time_data)
            case (False, False):
                ser = pd.Series(to_time_data)

        df = pd.DataFrame(
            {
                from_time_col: from_time_data,
                to_time_col: ser,
            }
        )
        assert (
            df[to_time_col].nunique() == to_time_config.length
        ), "to_time_col does not have the right number of timestamps"
        from_time_config = self._from_time_config.model_copy()
        from_time_config.time_column = from_time_col
        mapping_schema = MappingTableSchema(
            name="mapping_table",
            time_configs=[
                from_time_config,
                to_time_config,
            ],
        )
        return df, mapping_schema

    def _create_aggregation_mapping(
        self, from_time_config: DatetimeRange, to_time_config: DatetimeRange
    ) -> tuple[pd.DataFrame, MappingTableSchema]:
        """Create mapping dataframe for aggregation"""
        from_time_col = "from_" + from_time_config.time_column
        to_time_col = to_time_config.time_column
        from_time_data = make_time_range_generator(
            from_time_config, leap_day_adjustment=self._data_adjustment.leap_day_adjustment
        ).list_timestamps()
        to_time_data = make_time_range_generator(
            to_time_config, leap_day_adjustment=self._data_adjustment.leap_day_adjustment
        ).list_timestamps()
        df = pd.Series(from_time_data).rename(from_time_col).to_frame()
        df = df.join(
            pd.Series(to_time_data, index=to_time_data).rename(to_time_col), on=from_time_col
        )
        limit = to_time_config.resolution / from_time_config.resolution - 1
        assert limit % 1 == 0, f"limit is not a whole number {limit}"
        df[to_time_col].ffill(limit=int(limit), inplace=True)

        # mapping schema
        from_time_config = from_time_config.model_copy()
        from_time_config.time_column = from_time_col
        mapping_schema = MappingTableSchema(
            name="mapping_table",
            time_configs=[
                from_time_config,
                to_time_config,
            ],
        )
        return df, mapping_schema
