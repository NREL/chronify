import abc
from collections.abc import Generator
import logging
from datetime import datetime, timedelta
from typing import Any, Union, Literal
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import (
    Field,
)
from typing_extensions import Annotated

from chronify.base_models import ChronifyBaseModel
from chronify.time import (
    DatetimeFormat,
    DaylightSavingFallBackType,
    DaylightSavingSpringForwardType,
    LeapDayAdjustmentType,
    MeasurementType,
    TimeIntervalType,
    TimeType,
    TimeZone,
    RepresentativePeriodFormat,
    representative_period_columns,
)

# from chronify.time_utils import (
#    build_time_ranges,
#    filter_to_project_timestamps,
#    shift_time_interval,
#    time_difference,
#    apply_time_wrap,
# )


logger = logging.getLogger(__name__)


class AlignedTime(ChronifyBaseModel):
    """Data has absolute timestamps that are aligned with the same start and end
    for each geography."""

    format_type: Literal[DatetimeFormat.ALIGNED] = DatetimeFormat.ALIGNED
    time_zone: Annotated[
        TimeZone,
        Field(
            title="timezone",
            description="Time zone of data",
        ),
    ]


class LocalTimeAsStrings(ChronifyBaseModel):
    """Data has absolute timestamps formatted as strings with offsets from UTC.
    They are aligned for each geography when adjusted for time zone but staggered
    in an absolute time scale."""

    format_type: Literal[DatetimeFormat.LOCAL_AS_STRINGS] = DatetimeFormat.LOCAL_AS_STRINGS

    data_str_format: Annotated[
        str,
        Field(
            title="data_str_format",
            description="Timestamp string format (for parsing the time column of the dataframe)",
        ),
    ] = "yyyy-MM-dd HH:mm:ssZZZZZ"

    # @field_validator("data_str_format")
    # @classmethod
    # def check_data_str_format(cls, data_str_format):
    #    raise NotImplementedError("DatetimeFormat.LOCAL_AS_STRINGS is not fully implemented.")
    #    dsf = data_str_format
    #    if (
    #        "x" not in dsf
    #        and "X" not in dsf
    #        and "Z" not in dsf
    #        and "z" not in dsf
    #        and "V" not in dsf
    #        and "O" not in dsf
    #    ):
    #        raise ValueError("data_str_format must provide time zone or zone offset.")
    #    return data_str_format


class DaylightSavingAdjustment(ChronifyBaseModel):
    """Defines how to drop and add data along with timestamps to convert standard time
    load profiles to clock time"""

    spring_forward_hour: Annotated[
        DaylightSavingSpringForwardType,
        Field(
            title="spring_forward_hour",
            description="Data adjustment for spring forward hour (a 2AM in March)",
        ),
    ] = DaylightSavingSpringForwardType.NONE

    fall_back_hour: Annotated[
        DaylightSavingFallBackType,
        Field(
            title="fall_back_hour",
            description="Data adjustment for spring forward hour (a 2AM in November)",
        ),
    ] = DaylightSavingFallBackType.NONE


class TimeBasedDataAdjustment(ChronifyBaseModel):
    """Defines how data needs to be adjusted with respect to time.
    For leap day adjustment, up to one full day of timestamps and data are dropped.
    For daylight savings, the dataframe is adjusted alongside the timestamps.
    This is useful when the load profiles are modeled in standard time and
    need to be converted to get clock time load profiles.
    """

    leap_day_adjustment: Annotated[
        LeapDayAdjustmentType,
        Field(
            title="leap_day_adjustment",
            description="Leap day adjustment method applied to time data",
        ),
    ] = LeapDayAdjustmentType.NONE
    daylight_saving_adjustment: Annotated[
        DaylightSavingAdjustment,
        Field(
            title="daylight_saving_adjustment",
            description="Daylight saving adjustment method applied to time data",
        ),
    ] = DaylightSavingAdjustment()


class TimeBaseModel(ChronifyBaseModel, abc.ABC):
    """Defines a base model common to all time dimensions."""

    measurement_type: MeasurementType = MeasurementType.TOTAL

    def list_timestamps(self) -> list[Any]:
        """Return a list of timestamps for a time range.
        Type of the timestamps depends on the class.

        Returns
        -------
        list[Any]
        """
        return list(self.iter_timestamps())

    @abc.abstractmethod
    def list_distinct_timestamps_from_dataframe(self, df: pd.DataFrame) -> list[Any]:
        """Return a list of distinct timestamps present in DataFrame.
        Type of the timestamps depends on the class.

        Returns
        -------
        list[Any]
        """

    @abc.abstractmethod
    def iter_timestamps(self) -> Generator[Any, None, None]:
        """Return an iterator over all time indexes in the table.
        Type of the time is dependent on the class.
        """

    @abc.abstractmethod
    def list_time_columns(self) -> list[str]:
        """Return the columns in the table that represent time."""


class DatetimeRange(TimeBaseModel):
    """Defines a time range that uses Python datetime instances."""

    time_column: str = Field(description="Column in the table that represents time.")
    time_type: Literal[TimeType.DATETIME] = TimeType.DATETIME
    start: datetime = Field(
        description="Start time of the range. If it includes a time zone, the timestamps in "
        "the data must also include time zones."
    )
    length: int
    resolution: timedelta
    time_based_data_adjustment: TimeBasedDataAdjustment = TimeBasedDataAdjustment()
    interval_type: TimeIntervalType = TimeIntervalType.PERIOD_ENDING

    def is_time_zone_naive(self) -> bool:
        """Return True if the timestamps in the range do not have time zones."""
        return self.start.tzinfo is None

    def list_distinct_timestamps_from_dataframe(self, df: pd.DataFrame) -> list[datetime]:
        return sorted(df[self.time_column].unique())

    def list_time_columns(self) -> list[str]:
        return [self.time_column]

    def iter_timestamps(self) -> Generator[datetime, None, None]:
        for i in range(self.length):
            if self.is_time_zone_naive():
                cur = adjust_timestamp_by_dst_offset(
                    self.start + i * self.resolution, self.resolution
                )
            else:
                tz = self.start.tzinfo
                # always step in standard time
                cur_utc = self.start.astimezone(ZoneInfo("UTC")) + i * self.resolution
                cur = adjust_timestamp_by_dst_offset(cur_utc.astimezone(tz), self.resolution)
            month = cur.month
            day = cur.day
            if not (
                self.time_based_data_adjustment.leap_day_adjustment
                == LeapDayAdjustmentType.DROP_FEB29
                and month == 2
                and day == 29
            ):
                if not (
                    self.time_based_data_adjustment.leap_day_adjustment
                    == LeapDayAdjustmentType.DROP_DEC31
                    and month == 12
                    and day == 31
                ):
                    if not (
                        self.time_based_data_adjustment.leap_day_adjustment
                        == LeapDayAdjustmentType.DROP_JAN1
                        and month == 1
                        and day == 1
                    ):
                        yield cur


class AnnualTimeRange(TimeBaseModel):
    """Defines a time range that uses years as integers."""

    time_column: str = Field(description="Column in the table that represents time.")
    time_type: Literal[TimeType.ANNUAL] = TimeType.ANNUAL
    start: int
    length: int
    # TODO: measurement_type must be TOTAL, not necessarily right?

    def iter_timestamps(self) -> Generator[int, None, None]:
        for i in range(1, self.length + 1):
            yield i

    def list_time_columns(self) -> list[str]:
        return [self.time_column]


class IndexTimeRange(TimeBaseModel):
    time_type: Literal[TimeType.INDEX] = TimeType.INDEX
    start: int
    length: int
    resolution: timedelta
    time_zone: TimeZone
    time_based_data_adjustment: TimeBasedDataAdjustment
    interval_type: TimeIntervalType = TimeIntervalType.PERIOD_ENDING

    # TODO DT: totally wrong
    # def iter_timestamps(self) -> Generator[datetime, None, None]:
    #    cur = self.start.to_pydatetime().astimezone(ZoneInfo("UTC"))
    #    cur_idx = self.start_index
    #    end = (
    #        self.end.to_pydatetime().astimezone(ZoneInfo("UTC")) + self.resolution
    #    )  # to make end time inclusive

    #    while cur < end:
    #        cur_tz = cur.astimezone(self.tzinfo)
    #        cur_tz = adjust_timestamp_by_dst_offset(cur_tz, self.resolution)
    #        month = cur_tz.month
    #        day = cur_tz.day
    #        if not (
    #            self.time_based_data_adjustment.leap_day_adjustment
    #            == LeapDayAdjustmentType.DROP_FEB29
    #            and month == 2
    #            and day == 29
    #        ):
    #            if not (
    #                self.time_based_data_adjustment.leap_day_adjustment
    #                == LeapDayAdjustmentType.DROP_DEC31
    #                and month == 12
    #                and day == 31
    #            ):
    #                if not (
    #                    self.time_based_data_adjustment.leap_day_adjustment
    #                    == LeapDayAdjustmentType.DROP_JAN1
    #                    and month == 1
    #                    and day == 1
    #                ):
    #                    yield cur_idx
    #        cur += self.resolution
    #        cur_idx += 1


class RepresentativePeriodTimeRange(TimeBaseModel):
    """Defines a representative time dimension."""

    time_type: Literal[TimeType.REPRESENTATIVE_PERIOD] = TimeType.REPRESENTATIVE_PERIOD
    time_format: RepresentativePeriodFormat
    interval_type: TimeIntervalType = TimeIntervalType.PERIOD_ENDING

    def list_time_columns(self) -> list[str]:
        match self.time_format:
            case RepresentativePeriodFormat.ONE_WEEK_PER_MONTH_BY_HOUR:
                return OneWeekPerMonthByHourHandler().list_time_columns()
            case RepresentativePeriodFormat.ONE_WEEKDAY_DAY_AND_ONE_WEEKEND_DAY_PER_MONTH_BY_HOUR:
                return OneWeekdayDayAndWeekendDayPerMonthByHourHandler().list_time_columns()

    def iter_timestamps(self) -> Generator[int, None, None]:
        match self.time_format:
            case RepresentativePeriodFormat.ONE_WEEK_PER_MONTH_BY_HOUR:
                return OneWeekPerMonthByHourHandler().iter_timestamps()
            case RepresentativePeriodFormat.ONE_WEEKDAY_DAY_AND_ONE_WEEKEND_DAY_PER_MONTH_BY_HOUR:
                return OneWeekdayDayAndWeekendDayPerMonthByHourHandler().iter_timestamps()

    def list_distinct_timestamps_from_dataframe(self, df: pd.DataFrame) -> list[Any]:
        return df[self.list_time_columns()].drop_duplicates().apply(tuple, axis=1).to_list()


class RepresentativeTimeFormatHandlerBase(abc.ABC):
    """Provides implementations for different representative time formats."""

    @staticmethod
    @abc.abstractmethod
    def list_time_columns() -> list[str]:
        """Return the columns in the table that represent time."""

    @staticmethod
    @abc.abstractmethod
    def iter_timestamps() -> Generator[Any, None, None]:
        """Return an iterator over all time indexes in the table.
        Type of the time is dependent on the class.
        """


class OneWeekPerMonthByHourHandler(RepresentativeTimeFormatHandlerBase):
    """Handler for format with hourly data that includes one week per month."""

    @staticmethod
    def list_time_columns() -> list[str]:
        return representative_period_columns[RepresentativePeriodFormat.ONE_WEEK_PER_MONTH_BY_HOUR]

    @staticmethod
    def iter_timestamps() -> Generator[Any, None, None]:
        for month in range(1, 13):
            for dow in range(7):
                for hour in range(24):
                    yield (month, dow, hour)


class OneWeekdayDayAndWeekendDayPerMonthByHourHandler(RepresentativeTimeFormatHandlerBase):
    """Handler for format with hourly data that includes one weekday day and one weekend day
    per month.
    """

    @staticmethod
    def list_time_columns() -> list[str]:
        return representative_period_columns[
            RepresentativePeriodFormat.ONE_WEEKDAY_DAY_AND_ONE_WEEKEND_DAY_PER_MONTH_BY_HOUR
        ]

    @staticmethod
    def iter_timestamps() -> Generator[Any, None, None]:
        for month in range(1, 13):
            for is_weekday in sorted([False, True]):
                for hour in range(24):
                    yield (month, is_weekday, hour)


TimeConfig = Annotated[
    Union[AnnualTimeRange, DatetimeRange, IndexTimeRange, RepresentativePeriodTimeRange],
    Field(
        description="Defines the times in a time series table.",
        discriminator="time_type",
    ),
]


def adjust_timestamp_by_dst_offset(timestamp: datetime, resolution: timedelta) -> datetime:
    """Reduce the timestamps within the daylight saving range by 1 hour.
    Used to ensure that a time series at daily (or lower) resolution returns each day at the
    same timestamp in prevailing time, an expected behavior in most standard libraries.
    (e.g., ensure a time series can return 2018-03-11 00:00, 2018-03-12 00:00...
    instead of 2018-03-11 00:00, 2018-03-12 01:00...)
    """
    if resolution < timedelta(hours=24):
        return timestamp

    offset = timestamp.dst() or timedelta(hours=0)
    return timestamp - offset
