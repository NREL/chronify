from typing import Type
from sqlalchemy import Engine, MetaData
from chronify.models import TableSchema

from chronify.time_series_mapper_representative import MapperRepresentativeTimeToDatetime
from chronify.time_series_mapper_index import MapperIndexTimeToDatetime
from chronify.time_series_mapper_datetime import MapperDatetimeToDatetime
from chronify.time_configs import IndexTimeRange, RepresentativePeriodTime, DatetimeRange


def check_mapping(from_schema: TableSchema, to_schema: TableSchema, from_config_type: Type, to_config_type: Type):
    if isinstance(from_schema.time_config, from_config_type) and isinstance(to_schema.time_config, to_config_type):
        return True
    return False

def map_time(
    engine: Engine, metadata: MetaData, from_schema: TableSchema, to_schema: TableSchema
) -> None:
    """Function to map time using the appropriate TimeSeriesMapper model."""

    if check_mapping(from_schema, to_schema, RepresentativePeriodTime, DatetimeRange):
        MapperRepresentativeTimeToDatetime(engine, metadata, from_schema, to_schema).map_time()

    elif check_mapping(from_schema, to_schema, DatetimeRange, DatetimeRange):
        MapperDatetimeToDatetime(engine, metadata, from_schema, to_schema).map_time()

    elif check_mapping(from_schema, to_schema, IndexTimeRange, DatetimeRange):
        MapperIndexTimeToDatetime(engine, metadata, from_schema, to_schema).map_time()

    else:
        msg = f"No mapping function for {from_schema.time_config.__class__=} >> {to_schema.time_config.__class__=}"
        raise NotImplementedError(msg)
