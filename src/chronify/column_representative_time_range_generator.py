import pandas as pd
from datetime import datetime

from chronify.time_configs import (
    ColumnRepresentativeTimes,
    YearMonthDayHourTimeNTZ,
    MonthDayHourTimeNTZ,
    YearMonthDayPeriodTimeNTZ,
)
from chronify.time_range_generator_base import TimeRangeGeneratorBase


class ColumnRepresentativeTimeGenerator(TimeRangeGeneratorBase):
    def __init__(self, model: ColumnRepresentativeTimes):
        self._model = model
        self._time_columns = self._model.list_time_columns()

        if self._model.is_pivoted:
            # remove hour columns
            self._time_columns = list(set(self._time_columns) - set(self._model.hour_columns))

    def iter_timestamps(self):
        if isinstance(self._model, YearMonthDayPeriodTimeNTZ):
            yield from self._iter_period_timestamps()
        elif self._model.is_pivoted:
            yield from self._iter_pivoted_timestamps()
        else:
            yield from self._iter_unpivoted_timestamps()

    def _iter_pivoted_timestamps(self):
        """Timestamps where columns are 1-24 for each hour."""
        for dt in pd.date_range(
            start=datetime(self._model.year, 1, 1), periods=self._model.length / 24, freq="1D"
        ):
            if isinstance(self._model, YearMonthDayHourTimeNTZ):
                yield dt.year, dt.month, dt.day
            elif isinstance(self._model, MonthDayHourTimeNTZ):
                yield dt.month, dt.day

    def _iter_unpivoted_timestamps(self):
        """Timestamps where columns are (year), month, day, and hour."""
        for dt in pd.date_range(
            start=datetime(self._model.year, 1, 1), periods=self._model.length, freq="1h"
        ):
            if isinstance(self._model, YearMonthDayHourTimeNTZ):
                yield dt.year, dt.month, dt.day, dt.hour + 1
            elif isinstance(self._model, MonthDayHourTimeNTZ):
                yield dt.month, dt.day, dt.hour + 1

    def _iter_period_timestamps(self):
        for dt in pd.date_range(
            start=datetime(self._model.year, 1, 1), periods=self._model.length, freq="1D"
        ):
            yield dt.year, dt.month, dt.day

    def list_distinct_timestamps_from_dataframe(self, df):
        if isinstance(self._model, YearMonthDayPeriodTimeNTZ):
            # drops period column (hours column)
            int_columns = [
                self._model.year_column,
                self._model.month_column,
                self._model.day_column,
            ]
            df = df[int_columns].astype(int).drop_duplicates()
        else:
            df = df[self._time_columns].astype(int)
        return df.to_records(index=False).tolist()

    def list_time_columns(self):
        return self._model.list_time_columns()
