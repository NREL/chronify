"""Functions related to time"""

import logging
import numpy as np
from datetime import datetime, timedelta, timezone, tzinfo
import zoneinfo
import pandas as pd

from chronify.time import (
    TimeIntervalType,
)
from chronify.exceptions import InvalidParameter

logger = logging.getLogger(__name__)


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


def shift_time_interval(
    ser: "pd.Series[pd.Timestamp]",
    from_interval_type: TimeIntervalType,
    to_interval_type: TimeIntervalType,
) -> "pd.Series[pd.Timestamp]":
    """Shift pandas timeseries by ONE time interval based on interval type.

    Example:
    >>> ser = pd.Series(pd.date_range("2018-12-31 22:00", periods=4, freq="h"))
    0   2018-12-31 22:00:00
    1   2018-12-31 23:00:00
    2   2019-01-01 00:00:00
    3   2019-01-01 01:00:00
    dtype: datetime64[ns]

    >>> ser2 = shift_time_interval(
    ...     ser, TimeIntervalType.PERIOD_BEGINNING, TimeIntervalType.PERIOD_ENDING
    ... )
    0   2018-12-31 23:00:00
    1   2019-01-01 00:00:00
    2   2019-01-01 01:00:00
    3   2019-01-01 02:00:00
    dtype: datetime64[ns]
    """
    assert (
        from_interval_type != to_interval_type
    ), f"from_ and to_interval_type are the same: {from_interval_type}"
    arr = np.sort(ser)
    freqs = set((np.roll(arr, -1) - arr)[:-1])
    assert len(freqs), f"Timeseries has more than one frequency, {freqs}"
    freq: np.timedelta64 = next(iter(freqs))

    match (from_interval_type, to_interval_type):
        case (TimeIntervalType.PERIOD_BEGINNING, TimeIntervalType.PERIOD_ENDING):
            # shift time forward, 1am pd-beginning >> 2am pd-ending
            mult = 1
        case (TimeIntervalType.PERIOD_ENDING, TimeIntervalType.PERIOD_BEGINNING):
            # shift time backward
            mult = -1
        case _:
            msg = f"Cannot handle from {from_interval_type} to {to_interval_type}"
            raise InvalidParameter(msg)
    return ser + freq * mult


def wrap_timestamps(
    ser: "pd.Series[pd.Timestamp]", to_timestamps: list[pd.Timestamp]
) -> "pd.Series[pd.Timestamp]":
    """Wrap pandas timeseries so it stays within a list of timestamps.

    Example:
    >>> ser = pd.Series(pd.date_range("2018-12-31 22:00", periods=4, freq="h"))
    0   2018-12-31 22:00:00
    1   2018-12-31 23:00:00
    2   2019-01-01 00:00:00
    3   2019-01-01 01:00:00
    dtype: datetime64[ns]

    >>> to_timestamps = pd.date_range("2019-01-01 00:00", periods=4, freq="h").tolist()
    [Timestamp('2019-01-01 00:00:00'), Timestamp('2019-01-01 01:00:00'), Timestamp('2019-01-01 02:00:00'), Timestamp('2019-01-01 03:00:00')]

    >>> ser2 = wrap_timestamps(ser, to_timestamps)
    0   2019-01-01 02:00:00
    1   2019-01-01 03:00:00
    2   2019-01-01 00:00:00
    3   2019-01-01 01:00:00
    dtype: datetime64[ns]
    """
    arr = np.sort(np.array(to_timestamps))
    freqs = set((np.roll(arr, -1) - arr)[:-1])
    assert len(freqs), f"Timeseries has more than one frequency, {freqs}"
    freq = next(iter(freqs))
    tmin, tmax = arr[0], arr[-1]
    tdelta = tmax - tmin + freq
    ser2 = ser.copy()
    lower_cond = ser < tmin
    if lower_cond.sum() > 0:
        ser2.loc[lower_cond] += tdelta
    upper_cond = ser > tmax
    if upper_cond.sum() > 0:
        ser2.loc[upper_cond] -= tdelta
    return ser2


def roll_time_interval(
    ser: "pd.Series[pd.Timestamp]",
    from_interval_type: TimeIntervalType,
    to_interval_type: TimeIntervalType,
    to_timestamps: list[pd.Timestamp],
) -> "pd.Series[pd.Timestamp]":
    """Roll pandas timeseries by shifting time interval based on interval type and then
    wrapping timestamps

    Example:
    >>> ser = pd.Series(pd.date_range("2018-12-31 22:00", periods=4, freq="h"))
    0   2018-12-31 22:00:00
    1   2018-12-31 23:00:00
    2   2019-01-01 00:00:00
    3   2019-01-01 01:00:00
    dtype: datetime64[ns]

    >>> to_timestamps = pd.date_range("2019-01-01 00:00", periods=4, freq="h").tolist()
    [Timestamp('2019-01-01 00:00:00'), Timestamp('2019-01-01 01:00:00'), Timestamp('2019-01-01 02:00:00'), Timestamp('2019-01-01 03:00:00')]

    >>> ser2 = roll_time_interval(
    ...     ser, TimeIntervalType.PERIOD_BEGINNING, TimeIntervalType.PERIOD_ENDING, to_timestamps
    ... )
    0   2019-01-01 03:00:00
    1   2019-01-01 00:00:00
    2   2019-01-01 01:00:00
    3   2019-01-01 02:00:00
    dtype: datetime64[ns]
    """
    ser = shift_time_interval(ser, from_interval_type, to_interval_type)
    ser = wrap_timestamps(ser, to_timestamps)
    return ser


def get_standard_time_zone(tz: tzinfo | None) -> tzinfo | None:
    ts = datetime(year=2020, month=1, day=1, tzinfo=tz)
    std_tz_name = ts.tzname()
    if not std_tz_name:
        return None
    try:
        return zoneinfo.ZoneInfo(std_tz_name)
    except zoneinfo.ZoneInfoNotFoundError:
        utcoffset = ts.utcoffset()
        if not utcoffset:
            return None
        return timezone(utcoffset)
