import os
from typing import Any, Generator

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine, text
from chronify.models import TableSchema
from chronify.store import Store
from chronify.time import RepresentativePeriodFormat

from chronify.time_configs import RepresentativePeriodTime


ENGINES: dict[str, dict[str, Any]] = {
    "duckdb": {"url": "duckdb:///:memory:", "connect_args": {}, "kwargs": {}},
    # "sqlite": {"url": "sqlite:///:memory:", "connect_args": {}, "kwargs": {}},
}
HIVE_URL = os.getenv("CHRONIFY_HIVE_URL")
if HIVE_URL is not None:
    ENGINES["hive"] = {"url": HIVE_URL, "connect_args": {}, "kwargs": {}}


@pytest.fixture
def create_duckdb_engine() -> Engine:
    """Return a sqlalchemy engine for DuckDB."""
    return create_engine("duckdb:///:memory:")


@pytest.fixture(params=[x for x in ENGINES.keys() if x != "hive"])
def iter_engines(request) -> Generator[Engine, None, None]:
    """Return an iterable of sqlalchemy in-memory engines to test."""
    engine = ENGINES[request.param]
    yield create_engine(engine["url"], *engine["connect_args"], **engine["kwargs"])


@pytest.fixture(params=[x for x in ENGINES.keys() if x != "hive"])
def iter_stores_by_engine(request) -> Generator[Store, None, None]:
    """Return an iterable of stores with different engines to test.
    Will only return engines that support data ingestion.
    """
    engine = ENGINES[request.param]
    engine = create_engine(engine["url"], *engine["connect_args"], **engine["kwargs"])
    store = Store(engine=engine)
    yield store


@pytest.fixture(params=ENGINES.keys())
def iter_stores_by_engine_no_data_ingestion(request) -> Generator[Store, None, None]:
    """Return an iterable of stores with different engines to test."""
    engine = ENGINES[request.param]
    if engine["url"].startswith("hive"):
        store = Store.create_new_hive_store(
            engine["url"], *engine["connect_args"], drop_schema=True, **engine["kwargs"]
        )
        orig_tables_and_views = set()
        with store.engine.begin() as conn:
            for row in conn.execute(text("SHOW TABLES")).all():
                orig_tables_and_views.add(row[1])
    else:
        eng = create_engine(engine["url"], *engine["connect_args"], **engine["kwargs"])
        store = Store(engine=eng)
        orig_tables_and_views = None
    yield store
    if engine["url"].startswith("hive"):
        with store.engine.begin() as conn:
            for row in conn.execute(text("SHOW VIEWS")).all():
                name = row[1]
                if name not in orig_tables_and_views:
                    conn.execute(text(f"DROP VIEW {name}"))
            for row in conn.execute(text("SHOW TABLES")).all():
                name = row[1]
                if name not in orig_tables_and_views:
                    conn.execute(text(f"DROP TABLE {name}"))


@pytest.fixture(params=[x for x in ENGINES.keys() if x != "hive"])
def iter_engines_file(request, tmp_path) -> Generator[Engine, None, None]:
    """Return an iterable of sqlalchemy file-based engines to test."""
    engine = ENGINES[request.param]
    file_path = tmp_path / "store.db"
    url = engine["url"].replace(":memory:", str(file_path))
    yield create_engine(url, *engine["connect_args"], **engine["kwargs"])


@pytest.fixture(params=[x for x in ENGINES.keys() if x != "hive"])
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
