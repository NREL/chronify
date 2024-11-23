from typing import Any, Generator

import pandas as pd

from chronify.time_configs import AnnualTimeRange
from chronify.time_range_generator_base import TimeRangeGeneratorBase


class AnnualTimeRangeGenerator(TimeRangeGeneratorBase):
    def __init__(self, model: AnnualTimeRange) -> None:
        super().__init__()
        self._model = model

    def iter_timestamps(self) -> Generator[int, None, None]:
        for i in range(1, self._model.length + 1):
            yield i

    def list_distinct_timestamps_from_dataframe(self, df: pd.DataFrame) -> list[Any]:
        raise NotImplementedError

    def list_time_columns(self) -> list[str]:
        return self._model.list_time_columns()
