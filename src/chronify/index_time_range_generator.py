from typing import Any, Generator

import pandas as pd

from chronify.time_configs import IndexTimeRange
from chronify.time_range_generator_base import TimeRangeGeneratorBase


class IndexTimeRangeGenerator(TimeRangeGeneratorBase):
    """Generates datetime ranges based on a DatetimeRange model."""

    def __init__(self, model: IndexTimeRange) -> None:
        super().__init__()
        self._model = model

    def iter_timestamps(self) -> Generator[int, None, None]:
        # TODO: port from dsgrid
        raise NotImplementedError

    def list_distinct_timestamps_from_dataframe(self, df: pd.DataFrame) -> list[Any]:
        return []

    def list_time_columns(self) -> list[str]:
        return self._model.list_time_columns()
