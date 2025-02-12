import abc
import logging
from datetime import datetime, timedelta
from typing import Union, Literal, Optional
from pydantic import Field, field_validator
from typing_extensions import Annotated

from chronify.base_models import ChronifyBaseModel
from chronify.time import (
    DaylightSavingAdjustmentType,
    LeapDayAdjustmentType,
    MeasurementType,
    TimeIntervalType,
    TimeType,
    RepresentativePeriodFormat,
    list_representative_time_columns,
)


logger = logging.getLogger(__name__)


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
            description="Leap day adjustment method applied to change the table data based on the time column",
        ),
    ] = LeapDayAdjustmentType.NONE
    daylight_saving_adjustment: Annotated[
        DaylightSavingAdjustmentType,
        Field(
            title="daylight_saving_adjustment",
            description="Daylight saving adjustment method applied to change the table data based on the time column",
        ),
    ] = DaylightSavingAdjustmentType.NONE


class TimeBaseModel(ChronifyBaseModel, abc.ABC):
    """Defines a base model common to all time dimensions."""

    measurement_type: MeasurementType = MeasurementType.TOTAL
    interval_type: TimeIntervalType = TimeIntervalType.PERIOD_BEGINNING

    @abc.abstractmethod
    def list_time_columns(self) -> list[str]:
        """Return the columns in the table that represent time."""

    @abc.abstractmethod
    def get_time_zone_column(self) -> Optional[str]:
        """Return the column in the table that contains time zone or offset information."""


class DatetimeRange(TimeBaseModel):
    """Defines a time range that uses Python datetime instances."""

    time_column: str = Field(description="Column in the table that represents time.")
    time_type: Literal[TimeType.DATETIME] = TimeType.DATETIME
    start: datetime = Field(
        description="Start time of the range. If it includes a time zone, the timestamps in "
        "the data must be time zone-aware."
    )
    length: int
    resolution: timedelta

    def start_time_is_tz_naive(self) -> bool:
        """Return True if the timestamps in the range do not have time zones."""
        return self.start.tzinfo is None

    def list_time_columns(self) -> list[str]:
        return [self.time_column]

    def get_time_zone_column(self) -> None:
        return None


class AnnualTimeRange(TimeBaseModel):
    """Defines a time range that uses years as integers."""

    time_column: str = Field(description="Column in the table that represents time.")
    time_type: Literal[TimeType.ANNUAL] = TimeType.ANNUAL
    start: int
    length: int
    # TODO: measurement_type must be TOTAL, not necessarily right?

    def list_time_columns(self) -> list[str]:
        return [self.time_column]

    def get_time_zone_column(self) -> None:
        return None


class IndexTimeRangeBase(TimeBaseModel):
    """Defines a time range in the form of indexes"""

    time_column: str = Field(description="Column in the table that represents index time.")
    start: int = Field(description="starting index")
    length: int
    start_timestamp: datetime = Field(
        description="The timestamp represented by the starting index."
    )
    resolution: timedelta = Field(description="The resolution of time represented by the indexes.")
    time_type: TimeType

    def start_time_is_tz_naive(self) -> bool:
        """Return True if the represented timestamps do not have time zones."""
        return self.start_timestamp.tzinfo is None

    def list_time_columns(self) -> list[str]:
        return [self.time_column]


class IndexTimeRangeNTZ(IndexTimeRangeBase):
    """Index time that represents tz-naive timestamps.
    start_timestamp is tz-naive
    time_zone_column is None
    """

    time_type: Literal[TimeType.INDEX_NTZ] = TimeType.INDEX_NTZ

    @field_validator("start_timestamp")
    @classmethod
    def check_start_timestamp(cls, start_timestamp: datetime) -> datetime:
        if start_timestamp.tzinfo is not None:
            msg = "start_timestamp must be tz-naive for IndexTimeRangeNTZ"
            raise ValueError(msg)
        return start_timestamp

    def get_time_zone_column(self) -> None:
        return None


class IndexTimeRangeTZ(IndexTimeRangeBase):
    """Index time that represents tz-aware timestamps of a single time zone.
    start_timestamp is tz-aware
    time_zone_column is None
    """

    time_type: Literal[TimeType.INDEX_TZ] = TimeType.INDEX_TZ

    @field_validator("start_timestamp")
    @classmethod
    def check_start_timestamp(cls, start_timestamp: datetime) -> datetime:
        if start_timestamp.tzinfo is None:
            msg = "start_timestamp must be tz-aware for IndexTimeRangeTZ"
            raise ValueError(msg)
        return start_timestamp

    def get_time_zone_column(self) -> None:
        return None


class IndexTimeRangeLocalTime(IndexTimeRangeBase):
    """Index time that reprsents local time relative to a time zone column.
    start_timestamp is tz-naive
    time_zone_column is not None
    """

    time_type: Literal[TimeType.INDEX_LOCAL] = TimeType.INDEX_LOCAL
    time_zone_column: str = Field(
        description="Column in the table that has time zone or offset information."
    )

    @field_validator("start_timestamp")
    @classmethod
    def check_start_timestamp(cls, start_timestamp: datetime) -> datetime:
        if start_timestamp.tzinfo is not None:
            msg = "start_timestamp must be tz-naive for IndexTimeRangeLocalTime"
            raise ValueError(msg)
        return start_timestamp

    def get_time_zone_column(self) -> str:
        return self.time_zone_column


IndexTimeRanges = Union[
    IndexTimeRangeNTZ,
    IndexTimeRangeTZ,
    IndexTimeRangeLocalTime,
]


class RepresentativePeriodTimeBase(TimeBaseModel):
    """Defines a representative time dimension that covers one full year of time."""

    time_format: RepresentativePeriodFormat

    def list_time_columns(self) -> list[str]:
        return list_representative_time_columns(self.time_format)


class RepresentativePeriodTimeNTZ(RepresentativePeriodTimeBase):
    """Defines a tz-naive representative time dimension that covers one full year of time."""

    time_type: Literal[TimeType.REPRESENTATIVE_PERIOD_NTZ] = TimeType.REPRESENTATIVE_PERIOD_NTZ

    def get_time_zone_column(self) -> None:
        return None


class RepresentativePeriodTimeTZ(RepresentativePeriodTimeBase):
    """Defines a tz-aware representative time dimension that covers one full year of time."""

    time_type: Literal[TimeType.REPRESENTATIVE_PERIOD_TZ] = TimeType.REPRESENTATIVE_PERIOD_TZ
    time_zone_column: str = Field(
        description="Column in the table that has time zone or offset information.",
        default=None,
    )

    def get_time_zone_column(self) -> str:
        return self.time_zone_column


RepresentativePeriodTimes = Union[
    RepresentativePeriodTimeNTZ,
    RepresentativePeriodTimeTZ,
]

TimeConfig = Annotated[
    Union[
        AnnualTimeRange,
        DatetimeRange,
        IndexTimeRangeNTZ,
        IndexTimeRangeTZ,
        IndexTimeRangeLocalTime,
        RepresentativePeriodTimeNTZ,
        RepresentativePeriodTimeTZ,
    ],
    Field(
        description="Defines the times in a time series table.",
        discriminator="time_type",
    ),
]
