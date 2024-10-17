import abc
from collections.abc import Generator
import logging
from datetime import datetime, timedelta
from typing import Any, Optional, Union, Literal
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated

from chronify.time import (
    DatetimeFormat,
    DaylightSavingFallBackType,
    DaylightSavingSpringForwardType,
    LeapDayAdjustmentType,
    MeasurementType,
    TimeIntervalType,
    TimeType,
    TimeZone,
    get_zone_info,
)

# from chronify.time_utils import (
#    build_time_ranges,
#    filter_to_project_timestamps,
#    shift_time_interval,
#    time_difference,
#    apply_time_wrap,
# )


logger = logging.getLogger(__name__)


class AlignedTime(BaseModel):
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


class LocalTimeAsStrings(BaseModel):
    """Data has absolute timestamps formatted as strings with offsets from UTC.
    They are aligned for each geography when adjusted for time zone but staggered
    in an absolute time scale."""

    format_type: Literal[DatetimeFormat.LOCAL_AS_STRINGS] = DatetimeFormat.LOCAL_AS_STRINGS

    data_str_format: Annotated[
        str,
        Field(
            title="data_str_format",
            default="yyyy-MM-dd HH:mm:ssZZZZZ",
            description="Timestamp string format (for parsing the time column of the dataframe)",
        ),
    ]

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


class DaylightSavingAdjustment(BaseModel):
    """Defines how to drop and add data along with timestamps to convert standard time
    load profiles to clock time"""

    spring_forward_hour: Annotated[
        DaylightSavingSpringForwardType,
        Field(
            title="spring_forward_hour",
            description="Data adjustment for spring forward hour (a 2AM in March)",
            default=DaylightSavingSpringForwardType.NONE,
        ),
    ]

    fall_back_hour: Annotated[
        DaylightSavingFallBackType,
        Field(
            title="fall_back_hour",
            description="Data adjustment for spring forward hour (a 2AM in November)",
            default=DaylightSavingFallBackType.NONE,
        ),
    ]


class TimeBasedDataAdjustment(BaseModel):
    """Defines how data needs to be adjusted with respect to time.
    For leap day adjustment, up to one full day of timestamps and data are dropped.
    For daylight savings, the dataframe is adjusted alongside the timestamps.
    This is useful when the load profiles are modeled in standard time and
    need to be converted to get clock time load profiles.
    """

    leap_day_adjustment: Annotated[
        LeapDayAdjustmentType,
        Field(
            default=LeapDayAdjustmentType.NONE,
            title="leap_day_adjustment",
            description="Leap day adjustment method applied to time data",
        ),
    ]
    daylight_saving_adjustment: Annotated[
        DaylightSavingAdjustment,
        Field(
            default={
                "spring_forward_hour": DaylightSavingSpringForwardType.NONE,
                "fall_back_hour": DaylightSavingFallBackType.NONE,
            },
            title="daylight_saving_adjustment",
            description="Daylight saving adjustment method applied to time data",
        ),
    ]


class TimeBaseModel(BaseModel, abc.ABC):
    """Defines a base model common to all time dimensions."""

    time_columns: Annotated[
        list[str],
        Field(description="Columns in the table that represent time."),
    ]
    length: int

    def list_timestamps(self) -> list[Any]:
        """Return a list of timestamps for a time range.
        Type of the timestamps depends on the class.

        Returns
        -------
        list[datetime]
        """
        return list(self.iter_timestamps())

    @abc.abstractmethod
    def iter_timestamps(self) -> Generator[Any, None, None]:
        """Return an iterator over all time indexes in the table.
        Type of the time is dependent on the class.
        """

    @abc.abstractmethod
    def convert_database_timestamps(self, df: pd.DataFrame) -> list[Any]:
        """Convert timestamps from the database."""


class DatetimeRange(TimeBaseModel):
    """Defines a time range that uses Python datetime instances."""

    time_type: Literal[TimeType.DATETIME] = TimeType.DATETIME
    time_zone: Annotated[
        Optional[TimeZone],
        Field(
            default=None,
            description="Time zone if the timestamps are timezone-aware. "
            "If None, timestamps are timezone-naive.",
        ),
    ]
    start: datetime  # TODO: what if the time zone is specified here?
    resolution: timedelta
    time_based_data_adjustment: TimeBasedDataAdjustment = TimeBasedDataAdjustment()
    interval_type: TimeIntervalType = TimeIntervalType.PERIOD_ENDING
    measurement_type: MeasurementType = MeasurementType.TOTAL

    @model_validator(mode="after")
    def check_time_columns(self) -> "DatetimeRange":
        if len(self.time_columns) != 1:
            msg = f"{self.time_columns=} must have one column"
            raise ValueError(msg)
        return self

    @field_validator("start")
    @classmethod
    def fix_time_zone(cls, start: datetime, info: ValidationInfo) -> datetime:
        if "time_zone" not in info.data:
            return start
        if start.tzinfo is not None:
            return start
        if info.data["time_zone"] is not None:
            zone_info = get_zone_info(info.data["time_zone"])
            return start.replace(tzinfo=zone_info)
        return start

    def convert_database_timestamps(self, df: pd.DataFrame) -> list[datetime]:
        assert self.time_zone is not None
        tzinfo = get_zone_info(self.time_zone)
        time_column = self.get_time_column()
        return df[time_column].apply(lambda x: x.astimezone(tzinfo)).to_list()

    def get_time_column(self) -> str:
        """Return the time column."""
        return self.time_columns[0]

    def iter_timestamps(self) -> Generator[datetime, None, None]:
        tz_info = self.start.tzinfo
        for i in range(self.length):
            cur = self.start.astimezone(ZoneInfo("UTC")) + i * self.resolution
            cur = adjust_timestamp_by_dst_offset(cur.astimezone(tz_info), self.resolution)
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

    time_type: Literal[TimeType.ANNUAL] = TimeType.ANNUAL
    start: int
    # TODO: measurement_type must be TOTAL

    def iter_timestamps(self) -> Generator[int, None, None]:
        for i in range(1, self.length + 1):
            yield i


class IndexTimeRange(TimeBaseModel):
    time_type: Literal[TimeType.INDEX] = TimeType.INDEX
    start: int
    resolution: timedelta
    time_zone: TimeZone
    time_based_data_adjustment: TimeBasedDataAdjustment
    interval_type: TimeIntervalType
    measurement_type: MeasurementType

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
    measurement_type: MeasurementType
    time_interval_type: TimeIntervalType
    # TODO


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
