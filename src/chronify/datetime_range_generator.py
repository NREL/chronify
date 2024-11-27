from datetime import datetime
from typing import Generator
from zoneinfo import ZoneInfo

import pandas as pd

from chronify.time import (
    LeapDayAdjustmentType,
)
from chronify.time_configs import DatetimeRange, adjust_timestamp_by_dst_offset
from chronify.time_range_generator_base import TimeRangeGeneratorBase


class DatetimeRangeGenerator(TimeRangeGeneratorBase):
    """Generates datetime ranges based on a DatetimeRange model."""

    def __init__(self, model: DatetimeRange) -> None:
        self._model = model

    def iter_timestamps(self) -> Generator[datetime, None, None]:
        for i in range(self._model.length):
            if self._model.is_time_zone_naive():
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
            month = cur.month
            day = cur.day
            if not (
                self._model.time_based_data_adjustment.leap_day_adjustment
                == LeapDayAdjustmentType.DROP_FEB29
                and month == 2
                and day == 29
            ):
                if not (
                    self._model.time_based_data_adjustment.leap_day_adjustment
                    == LeapDayAdjustmentType.DROP_DEC31
                    and month == 12
                    and day == 31
                ):
                    if not (
                        self._model.time_based_data_adjustment.leap_day_adjustment
                        == LeapDayAdjustmentType.DROP_JAN1
                        and month == 1
                        and day == 1
                    ):
                        yield pd.Timestamp(cur)

    def list_distinct_timestamps_from_dataframe(self, df: pd.DataFrame) -> list[datetime]:
        return sorted(df[self._model.time_column].unique())

    def list_time_columns(self) -> list[str]:
        return self._model.list_time_columns()
