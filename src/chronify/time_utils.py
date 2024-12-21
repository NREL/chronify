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
    df: pd.Series, from_interval_type: TimeIntervalType, to_interval_type: TimeIntervalType
) -> pd.Series:
    """Shift pandas timeseries by time interval based on interval type."""
    assert (
        from_interval_type != to_interval_type
    ), f"from_ and to_interval_type are the same: {from_interval_type}"
    arr = np.sort(df)
    freqs = set((np.roll(arr, -1) - arr)[:-1])
    assert len(freqs), f"Timeseries has more than one frequency, {freqs}"
    freq = list(freqs)[0]

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
    return df + freq * mult


def roll_time_interval(
    df: pd.Series, from_interval_type: TimeIntervalType, to_interval_type: TimeIntervalType
) -> pd.Series:
    """Roll pandas timeseries by time interval based on interval type with np.roll(),
    which includes time-wrapping.
    """
    assert (
        from_interval_type != to_interval_type
    ), f"from_ and to_interval_type are the same: {from_interval_type}"
    match (from_interval_type, to_interval_type):
        case (TimeIntervalType.PERIOD_BEGINNING, TimeIntervalType.PERIOD_ENDING):
            # shift time forward, 1am pd-beginning >> 2am pd-ending
            shift = -1
        case (TimeIntervalType.PERIOD_ENDING, TimeIntervalType.PERIOD_BEGINNING):
            # shift time backward
            shift = 1
        case _:
            msg = f"Cannot handle from {from_interval_type} to {to_interval_type}"
            raise InvalidParameter(msg)
    return np.roll(df, shift)


def wrap_timestamps(df: pd.Series, to_timestamps: list[pd.Timestamp]) -> pd.Series:
    """Wrap pandas timeseries so it conforms to a list of timestamps."""
    arr = np.sort(to_timestamps)
    freqs = set((np.roll(arr, -1) - arr)[:-1])
    assert len(freqs), f"Timeseries has more than one frequency, {freqs}"
    freq = list(freqs)[0]

    tmin, tmax = arr[0], arr[-1]
    tdelta = tmax - tmin + freq
    df2 = df.copy()
    lower_cond = df < tmin
    if lower_cond.sum() > 0:
        df2.loc[lower_cond] += tdelta
    upper_cond = df > tmax
    if upper_cond.sum() > 0:
        df2.loc[upper_cond] -= tdelta
    return df2
