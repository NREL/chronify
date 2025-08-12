import os
from typing import Any, Generator
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd
import pytest

from sqlalchemy import Engine, create_engine, text

from chronify.models import TableSchema
from chronify.store import Store
from chronify.time import RepresentativePeriodFormat
from chronify.time_configs import RepresentativePeriodTimeNTZ, RepresentativePeriodTimeTZ


ENGINES: dict[str, dict[str, Any]] = {
    "duckdb": {"url": "duckdb:///:memory:", "connect_args": {}, "kwargs": {}},
    "sqlite": {"url": "sqlite:///:memory:", "connect_args": {}, "kwargs": {}},
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
    store.dispose()


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


def one_week_per_month_by_hour_data(tz_aware: bool = False) -> tuple[pd.DataFrame, int]:
    hours_per_year = 12 * 7 * 24
    num_time_arrays = 3
    df = pd.DataFrame(
        {
            "id": np.repeat(range(1, 1 + num_time_arrays), hours_per_year),
            "month": np.tile(np.repeat(range(1, 13), 7 * 24), num_time_arrays),
            # 0: Monday, 6: Sunday
            "day_of_week": np.tile(np.tile(np.repeat(range(7), 24), 12), num_time_arrays),
            "hour": np.tile(np.tile(range(24), 12 * 7), num_time_arrays),
            "value": np.random.random(hours_per_year * num_time_arrays),
        }
    )
    if tz_aware:
        df["time_zone"] = df["id"].map(
            dict(zip([1, 2, 3], ["US/Central", "US/Mountain", "US/Pacific"]))
        )
    return df, num_time_arrays


@pytest.fixture
def one_week_per_month_by_hour_table() -> tuple[pd.DataFrame, int, TableSchema]:
    """Return a table suitable for testing one_week_per_month_by_hour tz-naive data.

    Returns
    -------
    DataFrame, number of time arrays in DataFrame
    """
    df, num_time_arrays = one_week_per_month_by_hour_data()

    schema = TableSchema(
        name="ev_charging",
        value_column="value",
        time_config=RepresentativePeriodTimeNTZ(
            time_format=RepresentativePeriodFormat.ONE_WEEK_PER_MONTH_BY_HOUR,
        ),
        time_array_id_columns=["id"],
    )
    return df, num_time_arrays, schema


@pytest.fixture
def one_week_per_month_by_hour_table_tz() -> tuple[pd.DataFrame, int, TableSchema]:
    """Return a table suitable for testing one_week_per_month_by_hour tz-aware data.

    Returns
    -------
    DataFrame, number of time arrays in DataFrame
    """
    df, num_time_arrays = one_week_per_month_by_hour_data(tz_aware=True)

    schema = TableSchema(
        name="ev_charging",
        value_column="value",
        time_config=RepresentativePeriodTimeTZ(
            time_format=RepresentativePeriodFormat.ONE_WEEK_PER_MONTH_BY_HOUR,
            time_zone_column="time_zone",
        ),
        time_array_id_columns=["id"],
    )
    return df, num_time_arrays, schema


def one_weekday_day_and_one_weekend_day_per_month_by_hour_data(
    tz_aware: bool = False,
) -> tuple[pd.DataFrame, int]:
    hours_per_year = 12 * 2 * 24
    num_time_arrays = 3
    df = pd.DataFrame(
        {
            "id": np.repeat(range(1, 1 + num_time_arrays), hours_per_year),
            "month": np.tile(np.repeat(range(1, 13), 2 * 24), num_time_arrays),
            # 0: Monday, 6: Sunday
            "is_weekday": np.tile(np.tile(np.repeat([True, False], 24), 12), num_time_arrays),
            "hour": np.tile(np.tile(range(24), 12 * 2), num_time_arrays),
            "value": np.random.random(hours_per_year * num_time_arrays),
        }
    )
    if tz_aware:
        df["time_zone"] = df["id"].map(
            dict(zip([1, 2, 3], ["US/Central", "US/Mountain", "US/Pacific"]))
        )
    return df, num_time_arrays


@pytest.fixture
def one_weekday_day_and_one_weekend_day_per_month_by_hour_table() -> (
    tuple[pd.DataFrame, int, TableSchema]
):
    """Return a table suitable for testing one_weekday_day_and_one_weekend_day_per_month_by_hour tz-naive data.

    Returns
    -------
    DataFrame, number of time arrays in DataFrame
    """
    df, num_time_arrays = one_weekday_day_and_one_weekend_day_per_month_by_hour_data()
    schema = TableSchema(
        name="ev_charging",
        value_column="value",
        time_config=RepresentativePeriodTimeNTZ(
            time_format=RepresentativePeriodFormat.ONE_WEEKDAY_DAY_AND_ONE_WEEKEND_DAY_PER_MONTH_BY_HOUR,
        ),
        time_array_id_columns=["id"],
    )
    return df, num_time_arrays, schema


@pytest.fixture
def one_weekday_day_and_one_weekend_day_per_month_by_hour_table_tz() -> (
    tuple[pd.DataFrame, int, TableSchema]
):
    """Return a table suitable for testing one_weekday_day_and_one_weekend_day_per_month_by_hour tz-aware data.

    Returns
    -------
    DataFrame, number of time arrays in DataFrame
    """
    df, num_time_arrays = one_weekday_day_and_one_weekend_day_per_month_by_hour_data(tz_aware=True)
    schema = TableSchema(
        name="ev_charging",
        value_column="value",
        time_config=RepresentativePeriodTimeTZ(
            time_format=RepresentativePeriodFormat.ONE_WEEKDAY_DAY_AND_ONE_WEEKEND_DAY_PER_MONTH_BY_HOUR,
            time_zone_column="time_zone",
        ),
        time_array_id_columns=["id"],
    )
    return df, num_time_arrays, schema


def temp_csv_file(data: str):
    with NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(data)
        tmp_file.flush()
        tmp_file = Path(tmp_file.name)
        yield tmp_file

    tmp_file.unlink()


@pytest.fixture
def time_series_NMDH():
    hours = ",".join((str(x) for x in range(1, 25)))
    load1 = ",".join((str(x) for x in range(25, 49)))
    load2 = ",".join((str(x) for x in range(49, 73)))
    yield from temp_csv_file(
        f"name,month,day,{hours}\nGeneration,1,1,{load1}\nGeneration,1,2,{load2}"
    )


@pytest.fixture
def time_series_NYMDH():
    hours = ",".join((str(x) for x in range(1, 25)))
    load1 = ",".join((str(x) for x in range(25, 49)))
    load2 = ",".join((str(x) for x in range(49, 73)))
    yield from temp_csv_file(
        f"name,year,month,day,{hours}\ntest_generator,2023,1,1,{load1}\ntest_generator,2023,1,2,{load2}"
    )


@pytest.fixture
def time_series_NYMDPV():
    header = "name,year,month,day,period,value\n"
    data = "test_generator,2023,1,1,H1-5,100\ntest_generator,2023,1,1,H6-12,200\ntest_generator,2023,1,1,H13-24,300\ntest_generator,2023,1,2,H1-24,400"
    yield from temp_csv_file(header + data)
