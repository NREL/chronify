from chronify.time_configs import (
    AnnualTimeRange,
    DatetimeRange,
    IndexTimeRange,
    RepresentativePeriodTime,
    TimeBaseModel,
)
from chronify.annual_time_range_generator import AnnualTimeRangeGenerator
from chronify.datetime_range_generator import DatetimeRangeGenerator
from chronify.index_time_range_generator import IndexTimeRangeGenerator
from chronify.representative_time_range_generator import RepresentativePeriodTimeGenerator


def make_time_range_generator(model: TimeBaseModel):
    match model:
        case DatetimeRange():
            return DatetimeRangeGenerator(model)
        case AnnualTimeRange():
            return AnnualTimeRangeGenerator(model)
        case IndexTimeRange():
            return IndexTimeRangeGenerator(model)
        case RepresentativePeriodTime():
            return RepresentativePeriodTimeGenerator(model)
        case _:
            msg = str(type(model))
            raise NotImplementedError(msg)
