import abc
from collections.abc import Generator
import logging
from typing import Any
from collections import namedtuple

import pandas as pd

from chronify.time import RepresentativePeriodFormat
from chronify.time_configs import RepresentativePeriodTime
from chronify.time_range_generator_base import TimeRangeGeneratorBase

# from chronify.time_utils import (
#    build_time_ranges,
#    filter_to_project_timestamps,
#    shift_time_interval,
#    time_difference,
#    apply_time_wrap,
# )


logger = logging.getLogger(__name__)


class RepresentativePeriodTimeGenerator(TimeRangeGeneratorBase):
    """Implements Behavior in the Representative Period Time."""

    def __init__(self, model: RepresentativePeriodTime):
        super().__init__()
        self._model = model
        match self._model.time_format:
            case RepresentativePeriodFormat.ONE_WEEK_PER_MONTH_BY_HOUR:
                self._handler = OneWeekPerMonthByHourHandler()
            case RepresentativePeriodFormat.ONE_WEEKDAY_DAY_AND_ONE_WEEKEND_DAY_PER_MONTH_BY_HOUR:
                self._handler = OneWeekdayDayAndWeekendDayPerMonthByHourHandler()

    def iter_timestamps(self) -> Generator[tuple, None, None]:
        return self._handler.iter_timestamps()

    def list_distinct_timestamps_from_dataframe(self, df: pd.DataFrame) -> list[Any]:
        return list(
            df[self._model.list_time_columns()]
            .drop_duplicates()
            .itertuples(index=False, name=self._handler.get_time_type())
        )

    def list_time_columns(self) -> list[str]:
        return self._model.list_time_columns()


class RepresentativeTimeFormatHandlerBase(abc.ABC):
    """Provides implementations for different representative time formats."""

    @abc.abstractmethod
    def get_time_type() -> str:
        """Return the time type name representing the data."""

    @abc.abstractmethod
    def iter_timestamps() -> Generator[Any, None, None]:
        """Return an iterator over all time indexes in the table.
        Type of the time is dependent on the class.
        """

    @abc.abstractmethod
    def add_time_attribute_columns(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Extract attributes from timestamp_col as new columns."""

    def create_tz_aware_mapping_dataframe(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        time_zones: list[str],
    ) -> pd.DataFrame:
        """Create time zone-aware time mapping dataframe."""
        dfm = []
        for tz in time_zones:
            dft = df.copy()
            dft["timestamp_tmp"] = dft[timestamp_col].dt.tz_convert(tz)
            dft = self.add_time_attribute_columns(dft, "timestamp_tmp")
            dft["time_zone"] = tz
            dfm.append(dft.drop(columns=["timestamp_tmp"]))
        return pd.concat(dfm, axis=0, ignore_index=True)


class OneWeekPerMonthByHourHandler(RepresentativeTimeFormatHandlerBase):
    """Handler for format with hourly data that includes one week per month."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_type = namedtuple("OneWeekPerMonthByHour", ["month", "day_of_week", "hour"])

    def get_time_type(self) -> str:
        return self.time_type.__name__

    def iter_timestamps(self) -> Generator[tuple[int, int, int], None, None]:
        for month in range(1, 13):
            for dow in range(7):
                for hour in range(24):
                    yield self.time_type(month, dow, hour)

    def add_time_attribute_columns(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        dfm = df.copy()
        dfm["month"] = dfm[timestamp_col].dt.month
        dfm["day_of_week"] = dfm[timestamp_col].dt.day_of_week
        dfm["hour"] = dfm[timestamp_col].dt.hour
        return dfm


class OneWeekdayDayAndWeekendDayPerMonthByHourHandler(RepresentativeTimeFormatHandlerBase):
    """Handler for format with hourly data that includes one weekday day and one weekend day
    per month.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_type = namedtuple(
            "OneWeekdayDayAndWeekendDayPerMonthByHour", ["month", "is_weekday", "hour"]
        )

    def get_time_type(self) -> str:
        return self.time_type.__name__

    def iter_timestamps(self) -> Generator[tuple[int, bool, int], None, None]:
        for month in range(1, 13):
            for is_weekday in [False, True]:
                for hour in range(24):
                    yield self.time_type(month, is_weekday, hour)

    def add_time_attribute_columns(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        dfm = df.copy()
        dfm["month"] = dfm[timestamp_col].dt.month
        dow = dfm[timestamp_col].dt.day_of_week
        dfm["is_weekday"] = False
        dfm.loc[dow < 5, "is_weekday"] = True
        dfm["hour"] = dfm[timestamp_col].dt.hour
        return dfm