from pathlib import Path
from typing import Optional

from sqlalchemy import Engine, MetaData
from chronify.models import TableSchema

from chronify.time_series_mapper_representative import MapperRepresentativeTimeToDatetime
from chronify.time_series_mapper_datetime import MapperDatetimeToDatetime
from chronify.time_configs import RepresentativePeriodTime, DatetimeRange


def map_time(
    engine: Engine,
    metadata: MetaData,
    from_schema: TableSchema,
    to_schema: TableSchema,
    scratch_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    check_mapped_timestamps: bool = True,
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
    else:
        msg = f"No mapping function for {from_schema.time_config.__class__=} >> {to_schema.time_config.__class__=}"
        raise NotImplementedError(msg)
