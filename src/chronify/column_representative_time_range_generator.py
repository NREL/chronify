import abc
import pandas as pd
from datetime import datetime
from typing import Generator

from chronify.time_configs import (
    ColumnRepresentativeBase,
    YearMonthDayHourTimeNTZ,
    MonthDayHourTimeNTZ,
    YearMonthDayPeriodTimeNTZ,
)
from chronify import exceptions
from chronify.time_range_generator_base import TimeRangeGeneratorBase


class ColumnRepresentativeTimeGenerator(TimeRangeGeneratorBase):
    """
    Class to generate time for integer based representative time like:
        (year, month, day, hour) or (month, day, hour) as examples.
    """

    def __init__(self, model: ColumnRepresentativeBase):
        self._model = model
        self._time_columns = self._model.list_time_columns()

        if self._model.year is None:
            msg = "Can't generate column representative time without year"
            raise exceptions.InvalidValue(msg)

        self._year: int = self._model.year
        self._handler: ColumnRepresentativeHandlerHourly | ColumnRepresentativeHandlerPeriod
        if isinstance(self._model, (MonthDayHourTimeNTZ, YearMonthDayHourTimeNTZ)):
            self._handler = ColumnRepresentativeHandlerHourly(self._model, self._year)
        elif isinstance(self._model, YearMonthDayPeriodTimeNTZ):
            self._handler = ColumnRepresentativeHandlerPeriod(self._model, self._year)
        else:
            msg = f"No time generator for ColumnRepresentative time with time_config {type(self._model)}"
            raise exceptions.InvalidOperation(msg)

    def iter_timestamps(self) -> Generator[tuple[int, ...], None, None]:
        yield from self._handler._iter_timestamps()

    def list_distinct_timestamps_from_dataframe(self, df: pd.DataFrame) -> list[tuple[int, ...]]:
        return self._handler.list_distinct_timestamps_from_dataframe(df)

    def list_time_columns(self) -> list[str]:
        return self._model.list_time_columns()


class ColumnRepresentativeHandlerBase:
    """
    Base class for handling timestamps in different Column Representative formats.

    Methods
    -------
    list_distinct_timestamps_from_dataframe
        list the unique timestamps as a tuple of integers (for example (2024, 12, 31, 7) for year, month, day, hour)
        given a dataframe of time representing columns
    """

    @abc.abstractmethod
    def _iter_timestamps(self) -> Generator[tuple[int, ...], None, None]:
        """Iterates over a tuples that represent times from column schema."""

    @abc.abstractmethod
    def list_distinct_timestamps_from_dataframe(self, df: pd.DataFrame) -> list[tuple[int, ...]]:
        """Returns all unique tuples of representative column time, sorted by longest to shortest duration columns (year > month > day > hour)."""


class ColumnRepresentativeHandlerHourly(ColumnRepresentativeHandlerBase):
    def __init__(self, model: YearMonthDayHourTimeNTZ | MonthDayHourTimeNTZ, year: int) -> None:
        self._model = model
        self._year = year
        self._time_columns = self._model.list_time_columns()

    def _iter_timestamps(self) -> Generator[tuple[int, ...], None, None]:
        for dt in pd.date_range(
            start=datetime(self._year, 1, 1),
            periods=self._model.unique_timestamps_length,
            freq="1h",
        ):
            if isinstance(self._model, YearMonthDayHourTimeNTZ):
                yield dt.year, dt.month, dt.day, dt.hour + 1
            elif isinstance(self._model, MonthDayHourTimeNTZ):
                yield dt.month, dt.day, dt.hour + 1

    def list_distinct_timestamps_from_dataframe(self, df: pd.DataFrame) -> list[tuple[int, ...]]:
        df = df[self._time_columns].astype(int).drop_duplicates().sort_values(self._time_columns)
        return df.to_records(index=False).tolist()  # type: ignore


class ColumnRepresentativeHandlerPeriod(ColumnRepresentativeHandlerBase):
    def __init__(self, model: YearMonthDayPeriodTimeNTZ, year: int) -> None:
        self._model = model
        self._year = year
        self._time_columns = self._model.list_time_columns()

    def _iter_timestamps(self) -> Generator[tuple[int, ...], None, None]:
        for dt in pd.date_range(
            start=datetime(self._year, 1, 1),
            periods=self._model.unique_timestamps_length,
            freq="1D",
        ):
            yield dt.year, dt.month, dt.day

    def list_distinct_timestamps_from_dataframe(self, df: pd.DataFrame) -> list[tuple[int, ...]]:
        int_columns = [
            self._model.year_column,
            self._model.month_column,
            self._model.day_column,
        ]
        assert all(
            col in df.columns for col in int_columns
        ), f"Not all {int_columns=}, are in the input dataframe {df.columns=}"
        df = df[int_columns].astype(int).drop_duplicates().sort_values(int_columns)
        return df.to_records(index=False).tolist()  # type: ignore
