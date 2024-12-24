"""Functions related to time"""

import logging
import numpy as np

import pandas as pd

from chronify.time import (
    TimeIntervalType,
)
from chronify.exceptions import InvalidParameter

logger = logging.getLogger(__name__)


def shift_time_interval(
    dfs: "pd.Series[pd.Timestamp]",
    from_interval_type: TimeIntervalType,
    to_interval_type: TimeIntervalType,
) -> "pd.Series[pd.Timestamp]":
    """Shift pandas timeseries by time interval based on interval type."""
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
    Example usage:
    dfs = pd.Series([2018-12-31 22:00, 2018-12-31 23:00, ..., 2019-12-31 21:00])
    to_timestamps = [2019-01-01 00:00, ..., 2019-12-31 23:00]
    df2 = wrap_timestamps(dfs, to_timstamps)
    df2 => pd.Series([2019-12-31 22:00, 2019-12-31 23:00, ..., 2019-12-31 21:00])
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
) -> "pd.Series[pd.Timestamp]":
    """Roll pandas timeseries by shifting time interval based on interval type and
    wrapping timestamps
    """
    to_timestamps = dfs.tolist()
    dfs = shift_time_interval(dfs, from_interval_type, to_interval_type)
    dfs = wrap_timestamps(dfs, to_timestamps)
    return dfs
