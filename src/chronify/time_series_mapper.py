from chronify.time_series_mapper_representative import MapperRepresentativeTimeToDatetime
from chronify.time_configs import RepresentativePeriodTimeRange, DatetimeRange


def map_time(engine, metadata, from_schema, to_schema):
    """Factory function to map time using the appropriate TimeSeriesMapper model."""

    # TODO: Different mapper based on from_schema only or from_ and to_schema?
    if isinstance(from_schema.time_config, RepresentativePeriodTimeRange) and isinstance(
        to_schema.time_config, DatetimeRange
    ):
        return MapperRepresentativeTimeToDatetime(
            engine, metadata, from_schema, to_schema
        ).map_time()
    else:
        msg = f"No mapping function for {from_schema.time_config.__class__=} >> {to_schema.time_config.__class__=}"
        raise NotImplementedError(msg)
        # TODO use class if more than one method, func > use class as needed
