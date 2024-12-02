"""Definitions related to time"""

from enum import StrEnum
from typing import NamedTuple
from zoneinfo import ZoneInfo

from chronify.exceptions import InvalidParameter


class TimeType(StrEnum):
    """Defines the supported time formats in the load data."""

    DATETIME = "datetime"
    ANNUAL = "annual"
    REPRESENTATIVE_PERIOD = "representative_period"
    INDEX = "index"


class DatetimeFormat(StrEnum):
    """Defines the time format of the datetime config model"""

    ALIGNED = "aligned"
    LOCAL = "local"
    LOCAL_AS_STRINGS = "local_as_strings"


class RepresentativePeriodFormat(StrEnum):
    """Defines the supported formats for representative period data."""

    # All instances of this Enum must declare frequency.
    # This Enum may be replaced by a generic implementation in order to support a large
    # number of permutations (seasons, weekend day vs week day, sub-hour time, etc).

    ONE_WEEK_PER_MONTH_BY_HOUR = "one_week_per_month_by_hour"
    ONE_WEEKDAY_DAY_AND_ONE_WEEKEND_DAY_PER_MONTH_BY_HOUR = (
        "one_weekday_day_and_one_weekend_day_per_month_by_hour",
    )


class OneWeekPerMonthByHour(NamedTuple):
    month: int
    day_of_week: int
    hour: int


class OneWeekdayDayOneWeekendDayPerMonthByHour(NamedTuple):
    month: int
    is_weekday: bool
    hour: int


def list_representative_time_columns(format_type: RepresentativePeriodFormat) -> list[str]:
    """Return the time columns for the format."""
    match format_type:
        case RepresentativePeriodFormat.ONE_WEEK_PER_MONTH_BY_HOUR:
            columns = list(OneWeekPerMonthByHour._fields)
        case RepresentativePeriodFormat.ONE_WEEKDAY_DAY_AND_ONE_WEEKEND_DAY_PER_MONTH_BY_HOUR:
            columns = list(OneWeekdayDayOneWeekendDayPerMonthByHour._fields)
        case _:
            msg = str(format_type)
            raise NotImplementedError(msg)
    return list(columns)


class LeapDayAdjustmentType(StrEnum):
    """Leap day adjustment enum types"""

    DROP_DEC31 = "drop_dec31"
    DROP_FEB29 = "drop_feb29"
    DROP_JAN1 = "drop_jan1"
    NONE = "none"


class DaylightSavingSpringForwardType(StrEnum):
    """Daylight saving spring forward adjustment enum types"""

    DROP = "drop"
    NONE = "none"


class DaylightSavingFallBackType(StrEnum):
    """Daylight saving fall back adjustment enum types"""

    INTERPOLATE = "interpolate"
    DUPLICATE = "duplicate"
    NONE = "none"


class TimeIntervalType(StrEnum):
    """Time interval enum types"""

    # TODO: R2PD uses a different set; do we want to align?
    # https://github.com/Smart-DS/R2PD/blob/master/R2PD/tshelpers.py#L15

    PERIOD_ENDING = "period_ending"
    # description="A time interval that is period ending is coded by the end time. E.g., 2pm (with"
    # " freq=1h) represents a period of time between 1-2pm.",
    PERIOD_BEGINNING = "period_beginning"
    # description="A time interval that is period beginning is coded by the beginning time. E.g.,"
    # " 2pm (with freq=01:00:00) represents a period of time between 2-3pm. This is the dsgrid"
    # " default.",
    INSTANTANEOUS = "instantaneous"
    # description="The time record value represents measured, instantaneous time",


class MeasurementType(StrEnum):
    """Time value measurement enum types"""

    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    MEASURED = "measured"
    # description="Data values represent the measured value at that reported time",
    TOTAL = "total"
    # description="Data values represent the sum of values in a time range",


class TimeZone(StrEnum):
    """Time zones"""

    UTC = "UTC"
    HST = "HawaiiAleutianStandard"
    AST = "AlaskaStandard"
    APT = "AlaskaPrevailing"
    PST = "PacificStandard"
    PPT = "PacificPrevailing"
    MST = "MountainStandard"
    MPT = "MountainPrevailing"
    CST = "CentralStandard"
    CPT = "CentralPrevailing"
    EST = "EasternStandard"
    EPT = "EasternPrevailing"
    ARIZONA = "USArizona"


_TIME_ZONE_TO_ZONE_INFO = {
    TimeZone.UTC: ZoneInfo("UTC"),
    TimeZone.HST: ZoneInfo("US/Hawaii"),
    TimeZone.AST: ZoneInfo("Etc/GMT+9"),
    TimeZone.APT: ZoneInfo("US/Alaska"),
    TimeZone.PST: ZoneInfo("Etc/GMT+8"),
    TimeZone.PPT: ZoneInfo("US/Pacific"),
    TimeZone.MST: ZoneInfo("Etc/GMT+7"),
    TimeZone.MPT: ZoneInfo("US/Mountain"),
    TimeZone.CST: ZoneInfo("Etc/GMT+6"),
    TimeZone.CPT: ZoneInfo("US/Central"),
    TimeZone.EST: ZoneInfo("Etc/GMT+5"),
    TimeZone.EPT: ZoneInfo("US/Eastern"),
    TimeZone.ARIZONA: ZoneInfo("US/Arizona"),
}


def get_zone_info(tz: TimeZone) -> ZoneInfo:
    """Return a ZoneInfo instance for the given time zone."""
    return _TIME_ZONE_TO_ZONE_INFO[tz]


_TIME_ZONE_TO_TZ_OFFSET_AS_STR = {
    TimeZone.UTC: "UTC",
    TimeZone.HST: "-10:00",
    TimeZone.AST: "-09:00",
    TimeZone.PST: "-08:00",
    TimeZone.MST: "-07:00",
    TimeZone.CST: "-06:00",
    TimeZone.EST: "-05:00",
}


def get_time_zone_offset(tz: TimeZone) -> str:
    """Return the offset of the time zone from UTC."""
    offset = _TIME_ZONE_TO_TZ_OFFSET_AS_STR.get(tz)
    if offset is None:
        msg = f"Cannot get time zone offset for {tz=}"
        raise InvalidParameter(msg)
    return offset


def get_standard_time(tz: TimeZone) -> TimeZone:
    """Return the equivalent standard time zone."""
    match tz:
        case TimeZone.UTC:
            return TimeZone.UTC
        case TimeZone.HST:
            return TimeZone.HST
        case TimeZone.AST | TimeZone.APT:
            return TimeZone.AST
        case TimeZone.PST | TimeZone.PPT:
            return TimeZone.PST
        case TimeZone.MST | TimeZone.MPT:
            return TimeZone.MST
        case TimeZone.CST | TimeZone.CPT:
            return TimeZone.CST
        case TimeZone.EST | TimeZone.EPT:
            return TimeZone.EST
        case TimeZone.ARIZONA:
            return TimeZone.ARIZONA
        case _:
            msg = f"BUG: case not covered: {tz}"
            raise NotImplementedError(msg)


def get_prevailing_time(tz: TimeZone) -> TimeZone:
    """Return the equivalent prevailing time zone."""
    match tz:
        case TimeZone.UTC:
            return TimeZone.UTC
        case TimeZone.HST:
            return TimeZone.HST
        case TimeZone.AST | TimeZone.APT:
            return TimeZone.APT
        case TimeZone.PST | TimeZone.PPT:
            return TimeZone.PPT
        case TimeZone.MST | TimeZone.MPT:
            return TimeZone.MPT
        case TimeZone.CST | TimeZone.CPT:
            return TimeZone.CPT
        case TimeZone.EST | TimeZone.EPT:
            return TimeZone.EPT
        case TimeZone.ARIZONA:
            return TimeZone.ARIZONA
        case _:
            msg = f"BUG: case not covered: {tz}"
            raise NotImplementedError(msg)


_STANDARD_TIME_ZONES = {
    TimeZone.UTC,
    TimeZone.HST,
    TimeZone.AST,
    TimeZone.PST,
    TimeZone.MST,
    TimeZone.CST,
    TimeZone.EST,
    TimeZone.ARIZONA,
}
_PREVAILING_TIME_ZONES = {
    TimeZone.APT,
    TimeZone.PPT,
    TimeZone.MPT,
    TimeZone.CPT,
    TimeZone.EPT,
    TimeZone.ARIZONA,
}


def is_standard(tz: TimeZone) -> bool:
    """Return True if the time zone is a standard time zone."""
    return tz in _STANDARD_TIME_ZONES


def is_prevailing(tz: TimeZone) -> bool:
    """Return True if the time zone is a prevailing time zone."""
    return tz in _PREVAILING_TIME_ZONES
