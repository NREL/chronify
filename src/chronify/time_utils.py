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
