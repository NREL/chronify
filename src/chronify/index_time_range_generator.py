from typing import Any, Generator

import pandas as pd

from chronify.time_configs import IndexTimeRangeBase
from chronify.time_range_generator_base import TimeRangeGeneratorBase


class IndexTimeRangeGenerator(TimeRangeGeneratorBase):
    """Generates datetime ranges based on an IndexTimeRangeBase model."""

    def __init__(self, model: IndexTimeRangeBase) -> None:
        super().__init__()
        self._model = model

    def iter_timestamps(self) -> Generator[int, None, None]:
        yield from range(self._model.start, self._model.length + self._model.start)

    def list_distinct_timestamps_from_dataframe(self, df: pd.DataFrame) -> list[Any]:
        return sorted(df[self._model.time_column].unique())

    def list_time_columns(self) -> list[str]:
        return self._model.list_time_columns()
