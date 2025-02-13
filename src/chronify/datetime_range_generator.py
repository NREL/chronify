from datetime import datetime, timedelta
from typing import Generator, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from chronify.time import (
    LeapDayAdjustmentType,
)
from chronify.time_configs import (
    DatetimeRange,
)
from chronify.time_utils import adjust_timestamp_by_dst_offset
from chronify.time_range_generator_base import TimeRangeGeneratorBase


class DatetimeRangeGenerator(TimeRangeGeneratorBase):
    """Generates datetime ranges based on a DatetimeRange model."""

    def __init__(
        self,
        model: DatetimeRange,
        leap_day_adjustment: Optional[LeapDayAdjustmentType] = None,
    ) -> None:
        self._model = model
        self._adjustment = leap_day_adjustment or LeapDayAdjustmentType.NONE

    def iter_timestamps(self) -> Generator[datetime, None, None]:
        for i in range(self._model.length):
            if self._model.start_time_is_tz_naive():
                cur = adjust_timestamp_by_dst_offset(
                    self._model.start + i * self._model.resolution, self._model.resolution
                )
            else:
                tz = self._model.start.tzinfo
                # always step in standard time
                cur_utc = (
                    self._model.start.astimezone(ZoneInfo("UTC")) + i * self._model.resolution
                )
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

    def list_distinct_timestamps_from_dataframe(self, df: pd.DataFrame) -> list[datetime]:
        return sorted(df[self._model.time_column].unique())

    def list_time_columns(self) -> list[str]:
        return self._model.list_time_columns()
