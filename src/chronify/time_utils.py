"""Functions related to time"""

import logging
import numpy as np

import pandas as pd

from chronify.time import (
    TimeIntervalType,
)
from chronify.exceptions import InvalidParameter
from chronify.time_configs import DatetimeRange

logger = logging.getLogger(__name__)


def shift_time_interval(
    dfs: "pd.Series[pd.Timestamp]",
    from_interval_type: TimeIntervalType,
    to_interval_type: TimeIntervalType,
) -> "pd.Series[pd.Timestamp]":
    """Shift pandas timeseries by time interval based on interval type.

    Example:
    >>> dfs = pd.Series(pd.date_range("2018-12-31 22:00", periods=4, freq="h"))
    0   2018-12-31 22:00:00
    1   2018-12-31 23:00:00
    2   2019-01-01 00:00:00
    3   2019-01-01 01:00:00
    dtype: datetime64[ns]

    >>> dfs2 = shift_time_interval(
    ...     dfs, TimeIntervalType.PERIOD_BEGINNING, TimeIntervalType.PERIOD_ENDING
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
    arr = np.sort(dfs)
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
    return dfs + freq * mult


def wrap_timestamps(
    dfs: "pd.Series[pd.Timestamp]", to_timestamps: list[pd.Timestamp]
) -> "pd.Series[pd.Timestamp]":
    """Wrap pandas timeseries so it stays within a list of timestamps.

    Example:
    >>> dfs = pd.Series(pd.date_range("2018-12-31 22:00", periods=4, freq="h"))
    0   2018-12-31 22:00:00
    1   2018-12-31 23:00:00
    2   2019-01-01 00:00:00
    3   2019-01-01 01:00:00
    dtype: datetime64[ns]

    >>> to_timestamps = pd.date_range("2019-01-01 00:00", periods=4, freq="h").tolist()
    [Timestamp('2019-01-01 00:00:00'), Timestamp('2019-01-01 01:00:00'), Timestamp('2019-01-01 02:00:00'), Timestamp('2019-01-01 03:00:00')]

    >>> dfs2 = wrap_timestamps(dfs, to_timestamps)
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
    dfs2 = dfs.copy()
    lower_cond = dfs < tmin
    if lower_cond.sum() > 0:
        dfs2.loc[lower_cond] += tdelta
    upper_cond = dfs > tmax
    if upper_cond.sum() > 0:
        dfs2.loc[upper_cond] -= tdelta
    return dfs2


def roll_time_interval(
    dfs: "pd.Series[pd.Timestamp]",
    from_interval_type: TimeIntervalType,
    to_interval_type: TimeIntervalType,
    to_timestamps: list[pd.Timestamp],
    wrap_time_allowed: bool = False,
) -> "pd.Series[pd.Timestamp]":
    """Roll pandas timeseries by shifting time interval based on interval type and
    wrapping timestamps

    Example:
    >>> dfs = pd.Series(pd.date_range("2018-12-31 22:00", periods=4, freq="h"))
    0   2018-12-31 22:00:00
    1   2018-12-31 23:00:00
    2   2019-01-01 00:00:00
    3   2019-01-01 01:00:00
    dtype: datetime64[ns]

    >>> to_timestamps = pd.date_range("2019-01-01 00:00", periods=4, freq="h").tolist()
    [Timestamp('2019-01-01 00:00:00'), Timestamp('2019-01-01 01:00:00'), Timestamp('2019-01-01 02:00:00'), Timestamp('2019-01-01 03:00:00')]

    >>> dfs2 = roll_time_interval(
    ...     dfs, TimeIntervalType.PERIOD_BEGINNING, TimeIntervalType.PERIOD_ENDING, to_timestamps
    ... )
    0   2019-01-01 03:00:00
    1   2019-01-01 00:00:00
    2   2019-01-01 01:00:00
    3   2019-01-01 02:00:00
    dtype: datetime64[ns]
    """
    dfs = shift_time_interval(dfs, from_interval_type, to_interval_type)
    if wrap_time_allowed:
        dfs = wrap_timestamps(dfs, to_timestamps)
    return dfs


def shift_and_wrap_time_intervals(
    to_timestamps: list[pd.Timestamp],
    from_timestamps: "pd.Series[pd.Timestamp]",
    from_time_config: DatetimeRange,
    to_time_config: DatetimeRange,
    wrap_time_allowed: bool = False,
) -> "pd.Series[pd.Timestamp]":
    if from_time_config.interval_type != to_time_config.interval_type:
        # If from_tz or to_tz is naive, use tz_localize
        fm_tz = from_time_config.start.tzinfo
        to_tz = to_time_config.start.tzinfo
        if None in (fm_tz, to_tz) and (fm_tz, to_tz) != (None, None):
            from_timestamps = from_timestamps.dt.tz_localize(to_tz)

        from_timestamps = shift_time_interval(
            from_timestamps,
            from_time_config.interval_type,
            to_time_config.interval_type,
        )

    if wrap_time_allowed:
        from_timestamps = wrap_timestamps(from_timestamps, to_timestamps)

    return from_timestamps
