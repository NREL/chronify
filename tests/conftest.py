from typing import Any, Generator
from datetime import timedelta, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine
from chronify.models import TableSchema
from chronify.time import RepresentativePeriodFormat

from chronify.time_configs import RepresentativePeriodTime, TimeZone
from chronify.time_range_generator_factory import IndexTimeRange


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
    """Return an iterable of sqlalchemy in-memory engines to test."""
    engine = ENGINES[request.param]
    yield create_engine(engine["url"], *engine["connect_args"], **engine["kwargs"])


@pytest.fixture(params=ENGINES.keys())
def iter_engines_file(request, tmp_path) -> Generator[Engine, None, None]:
    """Return an iterable of sqlalchemy file-based engines to test."""
    engine = ENGINES[request.param]
    file_path = tmp_path / "store.db"
    url = engine["url"].replace(":memory:", str(file_path))
    yield create_engine(url, *engine["connect_args"], **engine["kwargs"])


@pytest.fixture(params=ENGINES.keys())
def iter_engine_names(request) -> Generator[str, None, None]:
    """Return an iterable of engine names."""
    yield request.param


@pytest.fixture
def one_week_per_month_by_hour_table() -> tuple[pd.DataFrame, int, TableSchema]:
    """Return a table suitable for testing one_week_per_month_by_hour data.

    Returns
    -------
    DataFrame, number of time arrays in DataFrame
    """
    hours_per_year = 12 * 7 * 24
    num_time_arrays = 3
    data = {
        "id": np.concat([np.repeat(i, hours_per_year) for i in range(1, 1 + num_time_arrays)]),
        "month": np.tile(np.repeat(range(1, 13), 7 * 24), num_time_arrays),
        # 0: Monday, 6: Sunday
        "day_of_week": np.tile(np.tile(np.repeat(range(7), 24), 12), num_time_arrays),
        "hour": np.tile(np.tile(range(24), 12 * 7), num_time_arrays),
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


@pytest.fixture
def one_weekday_day_and_one_weekend_day_per_month_by_hour_table() -> (
    tuple[pd.DataFrame, int, TableSchema]
):
    """Return a table suitable for testing one_weekday_day_and_one_weekend_day_per_month_by_hour data.

    Returns
    -------
    DataFrame, number of time arrays in DataFrame
    """
    hours_per_year = 12 * 2 * 24
    num_time_arrays = 3
    data = {
        "id": np.concat([np.repeat(i, hours_per_year) for i in range(1, 1 + num_time_arrays)]),
        "month": np.tile(np.repeat(range(1, 13), 2 * 24), num_time_arrays),
        # 0: Monday, 6: Sunday
        "is_weekday": np.tile(np.tile(np.repeat([True, False], 24), 12), num_time_arrays),
        "hour": np.tile(np.tile(range(24), 12 * 2), num_time_arrays),
        "value": np.random.random(hours_per_year * num_time_arrays),
    }
    schema = TableSchema(
        name="ev_charging",
        value_column="value",
        time_config=RepresentativePeriodTime(
            time_format=RepresentativePeriodFormat.ONE_WEEKDAY_DAY_AND_ONE_WEEKEND_DAY_PER_MONTH_BY_HOUR,
        ),
        time_array_id_columns=["id"],
    )
    return pd.DataFrame(data), num_time_arrays, schema


@pytest.fixture
def one_year_index_time_by_hour():
    """
    time_column: str = Field(description="Column in the table that represents time.")
    time_type: Literal[TimeType.INDEX] = TimeType.INDEX
    start: int
    length: int
    resolution: timedelta
    time_zone: TimeZone
    time_based_data_adjustment: TimeBasedDataAdjustment
    """
    hours_per_year = 24 * 365
    num_time_arrays = 3
    # id 1 based index, t_idx 0 based index
    data = {
        "id": np.repeat(np.arange(1, num_time_arrays + 1), hours_per_year),
        "t_idx": np.tile(np.arange(hours_per_year), num_time_arrays),
        "value": np.random.random(hours_per_year * num_time_arrays),
    }
    schema = TableSchema(
        name="ev_charging",
        value_column="value",
        time_config=IndexTimeRange(
            time_column="t_idx",
            start=0,
            start_timestamp=datetime(2019, 1, 1, tzinfo=ZoneInfo("US/Central")),
            length=hours_per_year + 1,
            resolution=timedelta(hours=1),
            time_zone=TimeZone.EST,  # standard time zone
        ),
        time_array_id_columns=["id"],
    )

    return pd.DataFrame(data), num_time_arrays, schema


@pytest.fixture
def one_year_index_time_subhourly():
    hours_per_year = 24 * 365
    index_length = hours_per_year * 2
    num_time_arrays = 3
    # id 1 based index, t_idx 0 based index
    data = {
        "id": np.repeat(np.arange(1, num_time_arrays + 1), index_length),
        "t_idx": np.tile(np.arange(index_length), num_time_arrays),
        "value": np.random.random(index_length * num_time_arrays),
    }
    schema = TableSchema(
        name="ev_charging",
        value_column="value",
        time_config=IndexTimeRange(
            time_column="t_idx",
            start=0,
            start_timestamp=datetime(2019, 1, 1, tzinfo=ZoneInfo("US/Pacific")),
            length=index_length,
            resolution=timedelta(minutes=30),
            time_zone=TimeZone.EPT,  # prevaling time zone
        ),
        time_array_id_columns=["id"],
    )
    return pd.DataFrame(data), num_time_arrays, schema


@pytest.fixture
def one_year_index_time_by_hour_leapyear():
    hours_per_year = (24 * 366) + 1
    num_time_arrays = 3
    # id 1 based index, t_idx 0 based index
    data = {
        "id": np.repeat(np.arange(1, num_time_arrays + 1), hours_per_year),
        "t_idx": np.tile(np.arange(hours_per_year), num_time_arrays),
        "value": np.random.random(hours_per_year * num_time_arrays),
    }
    schema = TableSchema(
        name="ev_charging",
        value_column="value",
        time_config=IndexTimeRange(
            time_column="t_idx",
            start=0,
            start_timestamp=datetime(2020, 1, 1, tzinfo=ZoneInfo("US/Eastern")),
            length=hours_per_year,
            resolution=timedelta(hours=1),
            time_zone=TimeZone.UTC,  # utc time zone
        ),
        time_array_id_columns=["id"],
    )
    return pd.DataFrame(data), num_time_arrays, schema
