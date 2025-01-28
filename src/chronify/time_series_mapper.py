from pathlib import Path
from typing import Optional

from sqlalchemy import Engine, MetaData
from chronify.models import TableSchema

from chronify.time_series_mapper_representative import MapperRepresentativeTimeToDatetime
from chronify.time_series_mapper_datetime import MapperDatetimeToDatetime
from chronify.time_series_mapper_index import (
    MapperIndexTimeToDatetime,
)
from chronify.time_configs import (
    RepresentativePeriodTime,
    DatetimeRange,
    IndexTimeRange,
    TimeBasedDataAdjustment,
)


def map_time(
    engine: Engine,
    metadata: MetaData,
    from_schema: TableSchema,
    to_schema: TableSchema,
    time_based_data_adjustment: TimeBasedDataAdjustment | None = None,
    wrap_time_allowed: bool = False,
    scratch_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    check_mapped_timestamps: bool = False,
) -> None:
    """Function to map time using the appropriate TimeSeriesMapper model."""

    if isinstance(from_schema.time_config, RepresentativePeriodTime) and isinstance(
        to_schema.time_config, DatetimeRange
    ):
        MapperRepresentativeTimeToDatetime(engine, metadata, from_schema, to_schema).map_time(
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )
    elif isinstance(from_schema.time_config, DatetimeRange) and isinstance(
        to_schema.time_config, DatetimeRange
    ):
        MapperDatetimeToDatetime(engine, metadata, from_schema, to_schema).map_time(
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )
    elif isinstance(from_schema.time_config, IndexTimeRange) and isinstance(
        to_schema.time_config, DatetimeRange
    ):
        MapperIndexTimeToDatetime(
            engine, metadata, from_schema, to_schema, time_based_data_adjustment
        ).map_time(
            wrap_time_allowed=wrap_time_allowed,
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )
    else:
        msg = f"No mapping function for {from_schema.time_config.__class__=} >> {to_schema.time_config.__class__=}"
        raise NotImplementedError(msg)
