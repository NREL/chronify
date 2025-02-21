from typing import Optional

from chronify.time_configs import (
    AnnualTimeRange,
    DatetimeRange,
    IndexTimeRangeBase,
    RepresentativePeriodTimeBase,
    TimeBaseModel,
    ColumnRepresentativeBase,
)
from chronify.time import LeapDayAdjustmentType
from chronify.annual_time_range_generator import AnnualTimeRangeGenerator
from chronify.datetime_range_generator import DatetimeRangeGenerator
from chronify.index_time_range_generator import IndexTimeRangeGenerator
from chronify.representative_time_range_generator import RepresentativePeriodTimeGenerator
from chronify.time_range_generator_base import TimeRangeGeneratorBase
from chronify.column_representative_time_range_generator import ColumnRepresentativeTimeGenerator


def make_time_range_generator(
    model: TimeBaseModel,
    leap_day_adjustment: Optional[LeapDayAdjustmentType] = None,
) -> TimeRangeGeneratorBase:
    match model:
        case DatetimeRange():
            return DatetimeRangeGenerator(model, leap_day_adjustment=leap_day_adjustment)
        case AnnualTimeRange():
            return AnnualTimeRangeGenerator(model)
        case IndexTimeRangeBase():
            return IndexTimeRangeGenerator(model)
        case RepresentativePeriodTimeBase():
            return RepresentativePeriodTimeGenerator(model)
        case ColumnRepresentativeBase():
            return ColumnRepresentativeTimeGenerator(model)
        case _:
            msg = str(type(model))
            raise NotImplementedError(msg)
