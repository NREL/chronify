import abc
from typing import Any

import pandas as pd


class TimeRangeGeneratorBase(abc.ABC):
    """Base class for classes that generate time ranges."""

    @abc.abstractmethod
    def list_timestamps(self) -> list[Any]:
        """Return a list of timestamps for a time range.
        Type of the timestamps depends on the class.
        Note: For DatetimeRangeGeneratorExternalTimeZone class with more than one time zone,
        this shows distinct timestamps only

        Returns
        -------
        list[Any]
        """

    @abc.abstractmethod
    def list_distinct_timestamps_from_dataframe(self, df: pd.DataFrame) -> list[Any]:
        """Return a list of distinct timestamps present in DataFrame.
        Type of the timestamps depends on the class.

        Returns
        -------
        list[Any]
        """

    @abc.abstractmethod
    def list_time_columns(self) -> list[str]:
        """Return the columns in the table that represent time."""
