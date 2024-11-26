import pandas as pd
from sqlalchemy import Engine, MetaData
from zoneinfo import ZoneInfo
import pytest
from datetime import datetime, timedelta

from chronify.sqlalchemy.functions import read_database, write_database
from chronify.time_series_mapper import map_time
from chronify.time_configs import DatetimeRange
from chronify.models import TableSchema
from chronify.time import TimeIntervalType, MeasurementType
from chronify.exceptions import ConflictingInputsError, MissingParameter
from chronify.datetime_range_generator import DatetimeRangeGenerator


def generate_datetime_data(time_config: DatetimeRange) -> pd.Series:
    return pd.to_datetime(list(DatetimeRangeGenerator(time_config).iter_timestamps()))


def get_datetime_schema(year, tzinfo) -> TableSchema:
    start = datetime(year=year, month=1, day=1, tzinfo=tzinfo)
    end = datetime(year=year + 1, month=1, day=1, tzinfo=tzinfo)
    resolution = timedelta(hours=1)
    length = (end - start) / resolution + 1
    schema = TableSchema(
        name="mapped_data",
        time_config=DatetimeRange(
            start=start,
            resolution=resolution,
            length=length,
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
            time_column="timestamp",
        ),
        time_array_id_columns=["id"],
        value_column="value",
    )
    return schema


def run_test(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
    to_schema: TableSchema,
    error: tuple[any, str],
) -> None:
    # Ingest
    metadata = MetaData()
    with engine.connect() as conn:
        write_database(df, conn, from_schema.name, from_schema.time_config)
        conn.commit()
    metadata.reflect(engine, views=True)

    # Map
    if error:
        with pytest.raises(error[0], match=error[1]):
            map_time(engine, metadata, from_schema, to_schema)
    else:
        map_time(engine, metadata, from_schema, to_schema)

        # Check mapped table
        with engine.connect() as conn:
            query = f"select * from {to_schema.name}"
            queried = read_database(query, conn, to_schema.time_config)
        queried = queried.sort_values(by=["id", "timestamp"]).reset_index(drop=True)

        truth = generate_datetime_data(to_schema.time_config)
        check_mapped_table(queried, truth)


def check_mapped_table(df: pd.DataFrame, ts: pd.Series) -> None:
    res = sorted(df["timestamp"].drop_duplicates().tolist())
    tru = sorted(ts)
    assert res == tru, "wrong unique timestamps"

    res = df.groupby(["time_zone"])["timestamp"].count().unique().tolist()
    tru = [len(ts)]
    assert res == tru, "wrong number of timestamps"


@pytest.mark.parametrize("tzinfo", [ZoneInfo("US/Pacific"), None])
def test_one_week_per_month_by_hour(
    iter_engines: Engine, one_week_per_month_by_hour_table, tzinfo
):
    df, _, schema = one_week_per_month_by_hour_table
    df["time_zone"] = df["id"].map(
        dict(zip([1, 2, 3], ["US/Central", "US/Mountain", "US/Pacific"]))
    )
    schema.time_array_id_columns += ["time_zone"]  # allowed in tz-naive

    to_schema = get_datetime_schema(2020, tzinfo)
    to_schema.time_array_id_columns += ["time_zone"]
    error = ()
    run_test(iter_engines, df, schema, to_schema, error)


@pytest.mark.parametrize("tzinfo", [ZoneInfo("US/Eastern"), None])
def test_one_weekday_day_and_one_weekend_day_per_month_by_hour(
    iter_engines: Engine, one_weekday_day_and_one_weekend_day_per_month_by_hour_table, tzinfo
):
    df, _, schema = one_weekday_day_and_one_weekend_day_per_month_by_hour_table
    df["time_zone"] = df["id"].map(
        dict(zip([1, 2, 3], ["US/Central", "US/Mountain", "US/Pacific"]))
    )
    schema.time_array_id_columns += ["time_zone"]  # allowed in tz-naive

    to_schema = get_datetime_schema(2020, tzinfo)
    to_schema.time_array_id_columns += ["time_zone"]
    error = ()
    run_test(iter_engines, df, schema, to_schema, error)


def test_missing_time_zone(iter_engines: Engine, one_week_per_month_by_hour_table):
    df, _, schema = one_week_per_month_by_hour_table

    to_schema = get_datetime_schema(2020, ZoneInfo("US/Mountain"))
    error = (
        MissingParameter,
        "time_zone is required for tz-aware representative time mapping and it is missing from source table",
    )
    run_test(iter_engines, df, schema, to_schema, error)


def test_schema_compatibility(iter_engines: Engine, one_week_per_month_by_hour_table):
    df, _, schema = one_week_per_month_by_hour_table

    to_schema = get_datetime_schema(2020, None)
    to_schema.time_array_id_columns += ["extra_column"]
    error = (ConflictingInputsError, ".* cannot produce the columns")
    run_test(iter_engines, df, schema, to_schema, error)


def test_measurement_type_consistency(iter_engines: Engine, one_week_per_month_by_hour_table):
    df, _, schema = one_week_per_month_by_hour_table

    to_schema = get_datetime_schema(2020, None)
    to_schema.time_config.measurement_type = MeasurementType.MAX
    error = (ConflictingInputsError, "Inconsistent measurement_types")
    run_test(iter_engines, df, schema, to_schema, error)
