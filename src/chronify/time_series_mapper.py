from sqlalchemy import Engine, MetaData
from chronify.models import TableSchema

from chronify.time_series_mapper_representative import MapperRepresentativeTimeToDatetime
from chronify.time_configs import RepresentativePeriodTime, DatetimeRange


def map_time(
    engine: Engine, metadata: MetaData, from_schema: TableSchema, to_schema: TableSchema
) -> None:
    """Function to map time using the appropriate TimeSeriesMapper model."""

    if isinstance(from_schema.time_config, RepresentativePeriodTime) and isinstance(
        to_schema.time_config, DatetimeRange
    ):
        MapperRepresentativeTimeToDatetime(engine, metadata, from_schema, to_schema).map_time()
    else:
        msg = f"No mapping function for {from_schema.time_config.__class__=} >> {to_schema.time_config.__class__=}"
        raise NotImplementedError(msg)
