from datetime import datetime, timedelta, tzinfo
from typing import Generator, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from chronify.time import (
    LeapDayAdjustmentType,
)
from chronify.time_configs import DatetimeRanges, DatetimeRange, DatetimeRangeWithTZColumn
from chronify.time_utils import adjust_timestamp_by_dst_offset, get_tzname
from chronify.time_range_generator_base import TimeRangeGeneratorBase
from chronify.exceptions import InvalidValue


class DatetimeRangeGeneratorBase(TimeRangeGeneratorBase):
    """Base class that generates datetime ranges based on a DatetimeRange model."""

    def __init__(
        self,
        model: DatetimeRanges,
        leap_day_adjustment: Optional[LeapDayAdjustmentType] = None,
    ) -> None:
        self._model = model
        self._adjustment = leap_day_adjustment or LeapDayAdjustmentType.NONE

    def _iter_timestamps(
        self, start: Optional[datetime] = None
    ) -> Generator[datetime, None, None]:
        """
        if start is supplied, override self._model.start
        """
        if start is None:
            start = self._model.start
        tz = start.tzinfo

        for i in range(self._model.length):
            if not tz:
                cur = adjust_timestamp_by_dst_offset(
                    start + i * self._model.resolution, self._model.resolution
                )
            else:
                # always step in standard time
                cur_utc = start.astimezone(ZoneInfo("UTC")) + i * self._model.resolution
                cur = adjust_timestamp_by_dst_offset(
                    cur_utc.astimezone(tz), self._model.resolution
                )

            is_leap_year = (
                pd.Timestamp(f"{cur.year}-01-01") + timedelta(days=365)
            ).year == cur.year
            if not is_leap_year:
                yield pd.Timestamp(cur)
                continue

            month = cur.month
            day = cur.day
            if not (
                self._adjustment == LeapDayAdjustmentType.DROP_FEB29 and month == 2 and day == 29
            ):
                if not (
                    self._adjustment == LeapDayAdjustmentType.DROP_DEC31
                    and month == 12
                    and day == 31
                ):
                    if not (
                        self._adjustment == LeapDayAdjustmentType.DROP_JAN1
                        and month == 1
                        and day == 1
                    ):
                        yield pd.Timestamp(cur)

    def list_time_columns(self) -> list[str]:
        return self._model.list_time_columns()

    def list_distinct_timestamps_from_dataframe(self, df: pd.DataFrame) -> list[datetime]:  # TODO
        result = sorted(df[self._model.time_column].unique())
        if not isinstance(result[0], datetime):
            result = [pd.Timestamp(x) for x in result]
        return result


class DatetimeRangeGenerator(DatetimeRangeGeneratorBase):
    """Generates datetime ranges based on a DatetimeRange model."""

    def __init__(
        self,
        model: DatetimeRange,
        leap_day_adjustment: Optional[LeapDayAdjustmentType] = None,
    ) -> None:
        super().__init__(model, leap_day_adjustment=leap_day_adjustment)
        assert isinstance(self._model, DatetimeRange)

    def list_timestamps(self) -> list[datetime]:
        return list(self._iter_timestamps())


class DatetimeRangeGeneratorExternalTimeZone(DatetimeRangeGeneratorBase):
    """Generates datetime ranges based on a DatetimeRangeWithTZColumn model.
    datetime ranges will be tz-naive and can be listed by time_zone name using special class func
    These ranges may be localized by the time_zone name.
    # TODO: add offset as a column
    """

    def __init__(
        self,
        model: DatetimeRangeWithTZColumn,
        leap_day_adjustment: Optional[LeapDayAdjustmentType] = None,
    ) -> None:
        super().__init__(model, leap_day_adjustment=leap_day_adjustment)
        assert isinstance(self._model, DatetimeRangeWithTZColumn)
        if self._model.get_time_zones() == []:
            msg = (
                "DatetimeRangeWithTZColumn.time_zones needs to be instantiated for ",
                f"DatetimeRangeGeneratorExternalTimeZone: {self._model}",
            )
            raise InvalidValue(msg)

    def _list_timestamps(self, time_zone: Optional[tzinfo]) -> list[datetime]:
        """always return tz-naive timestamps relative to input time_zone"""
        if self._model.start_time_is_tz_naive():
            if time_zone:
                start = self._model.start.replace(tzinfo=time_zone)
            else:
                start = None
        else:
            if time_zone:
                start = self._model.start.astimezone(time_zone)
            else:
                start = self._model.start.replace(tzinfo=None)
        timestamps = list(self._iter_timestamps(start=start))
        return [x.replace(tzinfo=None) for x in timestamps]

    def list_timestamps(self) -> list[datetime]:
        """return only unique values, this means no duplicates for prevailing time"""
        ts_set = set()
        for tz in self._model.get_time_zones():
            ts_set.update(set(self._list_timestamps(tz)))
        timestamps = sorted(ts_set)
        return timestamps

    def list_timestamps_by_time_zone(self, distinct: bool = False) -> dict[str, list[datetime]]:
        """for each time zone, returns full timestamp iteration with duplicates allowed"""
        dct = {}
        for tz in self._model.get_time_zones():
            timestamps = self._list_timestamps(tz)
            if distinct:
                timestamps = sorted(set(timestamps))
            tz_name = get_tzname(tz)
            dct[tz_name] = timestamps

        return dct

    def list_distinct_timestamps_by_time_zone_from_dataframe(
        self, df: pd.DataFrame
    ) -> dict[str, list[datetime]]:
        tz_col = self._model.get_time_zone_column()
        t_col = self._model.time_column
        df[t_col] = pd.to_datetime(df[t_col])
        df2 = df[[tz_col, t_col]].drop_duplicates()
        dct = {}
        for tz_name in sorted(df2[tz_col].unique()):
            dct[tz_name] = sorted(df2.loc[df2[tz_col] == tz_name, t_col].tolist())
        return dct
