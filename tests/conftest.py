from typing import Any, Generator

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine
from chronify.models import TableSchema
from chronify.time import RepresentativePeriodFormat

from chronify.time_configs import RepresentativePeriodTime


ENGINES: dict[str, dict[str, Any]] = {
    "duckdb": {"url": "duckdb:///:memory:", "connect_args": {}, "kwargs": {}},
    "sqlite": {"url": "sqlite:///:memory:", "connect_args": {}, "kwargs": {}},
}


@pytest.fixture
def create_duckdb_engine() -> Engine:
    """Return a sqlalchemy engine for DuckDB."""
    return create_engine("duckdb:///:memory:")


@pytest.fixture(params=ENGINES.keys())
def iter_engines(request) -> Generator[Engine, None, None]:
    """Return an iterable of sqlalchemy engines to test."""
    engine = ENGINES[request.param]
    yield create_engine(engine["url"], *engine["connect_args"], **engine["kwargs"])


@pytest.fixture
def one_week_per_month_by_hour_table() -> tuple[pd.DataFrame, int, TableSchema]:
    """Return a table suitable for testing one_week_per_month_by_hour data.

    Returns
    -------
    DataFrame, number of time arrays in DataFrame
    """
    months_per_year = 12
    hours_per_week = 24 * 7
    hours_per_year = months_per_year * hours_per_week
    num_time_arrays = 3
    data = {
        "id": np.concat([np.repeat(i, hours_per_year) for i in range(1, 1 + num_time_arrays)]),
        "month": np.repeat(np.repeat(range(1, 13), hours_per_week), num_time_arrays),
        # 0: Monday, 6: Sunday
        "day_of_week": np.repeat(np.tile(np.repeat(range(7), 24), 12), num_time_arrays),
        "hour": np.repeat(np.tile(range(24), 12 * 7), num_time_arrays),
        "value": np.random.random(hours_per_year * num_time_arrays),
    }

    schema = TableSchema(
        name="ev_charging",
        value_column="value",
        time_config=RepresentativePeriodTime(
            time_format=RepresentativePeriodFormat.ONE_WEEK_PER_MONTH_BY_HOUR,
        ),
        time_array_id_columns=["id"],
    )
    return pd.DataFrame(data), num_time_arrays, schema
