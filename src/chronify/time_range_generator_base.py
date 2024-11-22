import abc
from typing import Any, Generator

import pandas as pd


class TimeRangeGeneratorBase(abc.ABC):
    """Base class for classes that generate time ranges."""

    @abc.abstractmethod
    def iter_timestamps(self) -> Generator[Any, None, None]:
        """Return an iterator over all time indexes in the table.
        Type of the time is dependent on the class.
        """

    def list_timestamps(self) -> list[Any]:
        """Return a list of timestamps for a time range.
        Type of the timestamps depends on the class.

        Returns
        -------
        list[Any]
        """
        return list(self.iter_timestamps())

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
