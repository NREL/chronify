from datetime import datetime, tzinfo
from typing import Generator, Optional
from zoneinfo import ZoneInfo
from itertools import chain

import pandas as pd

from chronify.time import (
    LeapDayAdjustmentType,
    TimeDataType,
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
                pd.Timestamp(f"{cur.year}-01-01") + pd.Timedelta(days=365)
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

    def list_distinct_timestamps_from_dataframe(self, df: pd.DataFrame) -> list[datetime]:
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
                f"DatetimeRangeWithTZColumn.time_zones needs to be instantiated for "
                f"DatetimeRangeGeneratorExternalTimeZone: {self._model}"
            )
            raise InvalidValue(msg)

    def _list_timestamps(self, time_zone: Optional[tzinfo]) -> list[datetime]:
        """return timestamps for a given time_zone expected in the dataframe
        returned timestamp dtype matches self._model.dtype, e.g.,
        if time_zone is None, return tz-naive timestamps else return tz-aware timestamps
        """
        match (self._model.start_time_is_tz_naive(), self._model.dtype):
            case (True, TimeDataType.TIMESTAMP_NTZ):
                # aligned_in_local_time of the time zone, all time zones have the same tz-naive timestamps
                start = self._model.start
            case (True, TimeDataType.TIMESTAMP_TZ):
                # aligned_in_local_time of the time zone, all time zones have different tz-aware timestamps that are aligned when adjusted by time zone
                start = self._model.start.replace(tzinfo=time_zone)
            case (False, TimeDataType.TIMESTAMP_NTZ):
                # aligned_in_absolute_time, all time zones have different tz-naive timestamps that are aligned when localized to the time zone
                if time_zone:
                    start = self._model.start.astimezone(time_zone).replace(tzinfo=None)
                else:
                    start = self._model.start.replace(tzinfo=None)
            case (False, TimeDataType.TIMESTAMP_TZ):
                # aligned_in_absolute_time, all time zones have the same tz-aware timestamps
                start = self._model.start
            case _:
                msg = f"Unsupported combination of start_time_is_tz_naive and dtype: {self._model}"
                raise InvalidValue(msg)
        return list(self._iter_timestamps(start=start))

    def list_timestamps(self) -> list[datetime]:
        """return ordered tz-naive timestamps across all time zones in the order of the time zones."""
        dct = self.list_timestamps_by_time_zone()
        return list(chain(*dct.values()))

    def list_timestamps_by_time_zone(self) -> dict[str, list[datetime]]:
        """for each time zone, returns full timestamp iteration
        (duplicates allowed)"""
        dct = {}
        for tz in self._model.get_time_zones():
            tz_name = get_tzname(tz)
            dct[tz_name] = self._list_timestamps(tz)
        return dct

    def list_distinct_timestamps_by_time_zone_from_dataframe(
        self, df: pd.DataFrame
    ) -> dict[str, list[datetime]]:
        """
        from the dataframe, for each time zone, returns distinct timestamps
        """
        tz_col = self._model.get_time_zone_column()
        t_col = self._model.time_column
        df[t_col] = pd.to_datetime(df[t_col])
        df2 = df[[tz_col, t_col]].drop_duplicates()
        dct = {}
        for tz_name in sorted(df2[tz_col].unique()):
            timestamps = sorted(df2.loc[df2[tz_col] == tz_name, t_col].tolist())
            # if timestamps[0].tzinfo:
            #     timestamps = [x.astimezone(tz_name).replace(tzinfo=None) for x in timestamps]
            dct[tz_name] = timestamps
        return dct
