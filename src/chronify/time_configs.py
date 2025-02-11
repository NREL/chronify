import abc
import logging
from datetime import datetime, timedelta
from typing import Union, Literal, Optional
from pydantic import Field, field_validator, model_validator
from typing_extensions import Annotated

from chronify.base_models import ChronifyBaseModel
from chronify.time import (
    DatetimeFormat,
    DaylightSavingAdjustmentType,
    LeapDayAdjustmentType,
    MeasurementType,
    TimeIntervalType,
    TimeType,
    TimeZone,
    RepresentativePeriodFormat,
    list_representative_time_columns,
)
from chronify.exceptions import ConflictingInputsError, InvalidParameter


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
            description="Leap day adjustment method applied to change the dataframe based on the time column",
        ),
    ] = LeapDayAdjustmentType.NONE
    daylight_saving_adjustment: Annotated[
        DaylightSavingAdjustmentType,
        Field(
            title="daylight_saving_adjustment",
            description="Daylight saving adjustment method applied to change the dataframe based on the time column",
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
    def list_time_zone_column(self) -> list[str]:
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
    time_zone_column: Optional[str] = Field(
        description="Column in the table that has time zone or offset information.", default=None
    )

    @model_validator(mode="after")
    def check_time_zone_column(self) -> "DatetimeRange":
        tz_col = self.time_zone_column
        if self.start.tzinfo is not None and tz_col is not None:
            msg = f"{self.start} is tz-aware and {tz_col=} is provided"
            raise ConflictingInputsError(msg)
        return self

    def start_time_is_tz_naive(self) -> bool:
        """Return True if the timestamps in the range do not have time zones."""
        return self.start.tzinfo is None

    def list_time_columns(self) -> list[str]:
        return [self.time_column]

    def list_time_zone_column(self) -> list[str]:
        if self.time_zone_column is None:
            return []
        return [self.time_zone_column]

    # TODO: make different datetime classes


class AnnualTimeRange(TimeBaseModel):
    """Defines a time range that uses years as integers."""

    time_column: str = Field(description="Column in the table that represents time.")
    time_type: Literal[TimeType.ANNUAL] = TimeType.ANNUAL
    start: int
    length: int
    # TODO: measurement_type must be TOTAL, not necessarily right?

    def list_time_columns(self) -> list[str]:
        return [self.time_column]

    def list_time_zone_column(self) -> list[str]:
        return []


class IndexTimeRangeBase(TimeBaseModel):
    """Defines a time range in the form of indexes"""

    time_column: str = Field(description="Column in the table that represents index time.")
    start: int = Field(description="starting index")
    length: int
    start_timestamp: datetime = Field(
        description="The timestamp represented by the starting index."
    )
    resolution: timedelta = Field(description="The resolution of time represented by the indexes.")
    time_zone_column: Optional[str] = Field(
        description="Column in the table that has time zone or offset information.", default=None
    )

    @model_validator(mode="after")
    def check_time_zone_column(self) -> "IndexTimeRangeBase":
        tz_col = self.time_zone_column
        if self.start_timestamp.tzinfo is not None and tz_col is not None:
            msg = f"{self.start} is tz-aware and {tz_col=} is provided"
            raise ConflictingInputsError(msg)
        return self

    def start_time_is_tz_naive(self) -> bool:
        """Return True if the repreentative timestamps do not have time zones."""
        return self.start_timestamp.tzinfo is None

    def list_time_columns(self) -> list[str]:
        return [self.time_column]

    def list_time_zone_column(self) -> list[str]:
        if self.time_zone_column is None:
            return []
        return [self.time_zone_column]


class IndexTimeRangeNTZ(IndexTimeRangeBase):
    """Index time that represents tz-naive timestamps.
    start_timestamp is tz-naive
    time_zone_column is None
    """

    time_type: Literal[TimeType.INDEX_NTZ] = TimeType.INDEX_NTZ
    time_zone_column: None = None

    @field_validator("start_timestamp")
    @classmethod
    def check_start_timestamp(cls, start_timestamp: datetime) -> datetime:
        if start_timestamp.tzinfo is not None:
            msg = "start_timestamp must be tz-naive for IndexTimeRangeNTZ model"
            raise InvalidParameter(msg)
        return start_timestamp


class IndexTimeRangeTZ(IndexTimeRangeBase):
    """Index time that represents tz-aware timestamps of a single time zone.
    start_timestamp is tz-aware
    time_zone_column is None
    """

    time_type: Literal[TimeType.INDEX_TZ] = TimeType.INDEX_TZ
    time_zone_column: None = None

    @field_validator("start_timestamp")
    @classmethod
    def check_start_timestamp(cls, start_timestamp: datetime) -> datetime:
        if start_timestamp.tzinfo is None:
            msg = "start_timestamp must be tz-aware for IndexTimeRangeTZ model"
            raise InvalidParameter(msg)
        return start_timestamp


class IndexTimeRangeLocalTime(IndexTimeRangeBase):
    """Index time that reprsents local time relative to a time zone column.
    start_timestamp is tz-naive
    time_zone_column is not None
    """

    time_type: Literal[TimeType.INDEX_LOCAL] = TimeType.INDEX_LOCAL
    time_zone_column: str

    @field_validator("start_timestamp")
    @classmethod
    def check_start_timestamp(cls, start_timestamp: datetime) -> datetime:
        if start_timestamp.tzinfo is not None:
            msg = "start_timestamp must be tz-naive for IndexTimeRangeLocalTime model"
            raise InvalidParameter(msg)
        return start_timestamp


class RepresentativePeriodTime(TimeBaseModel):
    """Defines a representative time dimension that covers one full year of time."""

    time_type: Literal[TimeType.REPRESENTATIVE_PERIOD] = TimeType.REPRESENTATIVE_PERIOD
    time_format: RepresentativePeriodFormat
    time_zone_column: Optional[str] = Field(
        description="Column in the table that has time zone or offset information.",
        default=None,
    )

    def list_time_columns(self) -> list[str]:
        return list_representative_time_columns(self.time_format)

    def list_time_zone_column(self) -> list[str]:
        if self.time_zone_column is None:
            return []
        return [self.time_zone_column]


TimeConfig = Annotated[
    Union[
        AnnualTimeRange,
        DatetimeRange,
        IndexTimeRangeNTZ,
        IndexTimeRangeTZ,
        IndexTimeRangeLocalTime,
        RepresentativePeriodTime,
    ],
    Field(
        description="Defines the times in a time series table.",
        discriminator="time_type",
    ),
]
