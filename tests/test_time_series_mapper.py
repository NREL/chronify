import pandas as pd
import numpy as np
from sqlalchemy import Engine, MetaData
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pytest

from chronify.models import TableSchema
from chronify.time import RepresentativePeriodFormat, TimeIntervalType
from chronify.time_configs import RepresentativePeriodTime, DatetimeRange
from chronify.sqlalchemy.functions import read_database, write_database
from chronify.time_series_mapper import map_time


def generate_data(one_week=False) -> pd.DataFrame:
    def generate__one_week_per_month_by_hour():
        dfd = pd.DataFrame(
            {
                "id": np.repeat(1, 12 * 24 * 7),
                "month": np.repeat(range(1, 13), 24 * 7),
                "day_of_week": np.tile(
                    np.repeat(range(7), 24), 12
                ),  # 0: Monday, 6: Sunday, ~ pyspark.weekday(), duckdb.isodow()-1, pd.day_of_week
                "hour": np.tile(range(24), 12 * 7),
            }
        )
        dfd["value"] = dfd["month"] * 1000 + dfd["day_of_week"] * 100 + dfd["hour"]
        dfd = pd.concat([dfd, dfd.assign(id=2).assign(value=dfd.value * 2)], axis=0)
        return dfd

    def generate__one_weekday_day_and_one_weekend_day_per_month_by_hour():
        dfd = pd.DataFrame(
            {
                "id": np.repeat(1, 12 * 24 * 2),
                "month": np.repeat(range(1, 13), 24 * 2),
                "is_weekday": np.tile(np.repeat([True, False], 24), 12),
                "hour": np.tile(range(24), 12 * 2),
            }
        )
        dfd["value"] = dfd["month"] * 1000 + dfd["is_weekday"] * 100 + dfd["hour"]
        dfd = pd.concat([dfd, dfd.assign(id=2).assign(value=dfd.value * 2)], axis=0)
        return dfd

    if one_week:
        dfd = generate__one_week_per_month_by_hour()
    else:
        dfd = generate__one_weekday_day_and_one_weekend_day_per_month_by_hour()

    # Add more data
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "geography": ["IL", "CO"],
            "time_zone": ["US/Central", "US/Mountain"],
        }
    )
    dfd = dfd.merge(df, on="id", how="left")

    return dfd


def get_destination_schema(tzinfo) -> TableSchema:
    schema = TableSchema(
        name="mapped_data",
        time_config=DatetimeRange(
            start=datetime(year=2018, month=1, day=1, tzinfo=tzinfo),
            resolution=timedelta(hours=1),
            length=8760,
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
            time_column="timestamp",
        ),
        time_array_id_columns=["id", "geography", "time_zone"],
        value_column="value",
    )
    return schema


def get_data_schema(one_week: bool = False) -> TableSchema:
    if one_week:
        time_format = RepresentativePeriodFormat.ONE_WEEK_PER_MONTH_BY_HOUR
    else:
        time_format = (
            RepresentativePeriodFormat.ONE_WEEKDAY_DAY_AND_ONE_WEEKEND_DAY_PER_MONTH_BY_HOUR
        )
    schema = TableSchema(
        name="load_data",
        time_config=RepresentativePeriodTime(
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
            time_format=time_format,
        ),
        time_array_id_columns=["id", "geography", "time_zone"],
        value_column="value",
    )
    return schema


def get_timeseries(time_config: DatetimeRange) -> pd.Series:
    ts = pd.date_range(
        start=time_config.start, freq=time_config.resolution, periods=time_config.length
    ).rename(time_config.time_column)
    return ts


def run_test(engine: Engine, one_week: bool = False, tzinfo: ZoneInfo | None = None) -> None:
    engine.clear_compiled_cache()

    # Generate
    df = generate_data(one_week=one_week)

    # Ingest
    metadata = MetaData()
    schema = get_data_schema(one_week=one_week)
    with engine.connect() as conn:
        write_database(df, conn, schema.name, schema.time_config)
        conn.commit()
    metadata.reflect(engine, views=True)

    # Check time config class functions TODO: move to checker test
    # schema.time_config.iter_timestamps()
    # res = schema.time_config.list_distinct_timestamps_from_dataframe(df)
    # assert isinstance(res[0], tuple), "not tuple"
    # assert len(res[0]) == 3, "not a tuple of three elements"

    ## Map
    dest_schema = get_destination_schema(tzinfo=tzinfo)
    map_time(engine, metadata, schema, dest_schema)  # this creates a table

    # Check mapped table
    with engine.connect() as conn:
        query = f"select * from {dest_schema.name}"
        queried = read_database(query, conn, dest_schema.time_config)
    queried = queried.sort_values(by=["id", "timestamp"]).reset_index(drop=True)

    truth = get_timeseries(dest_schema.time_config)
    check_mapped_table(engine, queried, truth)


def check_mapped_table(engine: Engine, dfs: pd.DataFrame, ts: pd.Series) -> None:
    res = sorted(dfs["timestamp"].drop_duplicates().tolist())
    tru = sorted(ts)
    assert res == tru, "wrong unique timestamps"

    res = dfs.groupby(["geography", "time_zone"])["timestamp"].count().unique().tolist()
    tru = [len(ts)]
    assert res == tru, "wrong number of timestamps"


@pytest.mark.parametrize("tzinfo", [ZoneInfo("US/Eastern"), None])
def test__one_week_per_month_by_hour(iter_engines: Engine, tzinfo):
    run_test(iter_engines, one_week=True, tzinfo=tzinfo)


@pytest.mark.parametrize("tzinfo", [ZoneInfo("US/Eastern"), None])
def test__one_weekday_day_and_one_weekend_day_per_month_by_hour(iter_engines: Engine, tzinfo):
    run_test(iter_engines, one_week=False, tzinfo=tzinfo)
