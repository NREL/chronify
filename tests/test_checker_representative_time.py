from sqlalchemy import Engine, MetaData, Table
import pytest

import pandas as pd

from chronify.sqlalchemy.functions import write_database
from chronify.models import TableSchema
from chronify.time_series_checker import check_timestamps
from chronify.exceptions import InvalidTable


def ingest_data_and_check(
    engine: Engine, df: pd.DataFrame, schema: TableSchema, error: tuple[any, str]
) -> None:
    metadata = MetaData()
    with engine.connect() as conn:
        write_database(df, conn, schema.name, schema.time_config, if_table_exists="replace")
        conn.commit()
    metadata.reflect(engine, views=True)

    with engine.connect() as conn:
        table = Table(schema.name, metadata)
        if error:
            with pytest.raises(error[0], match=error[1]):
                check_timestamps(conn, table, schema)
        else:
            check_timestamps(conn, table, schema)


def test_one_week_per_month_by_hour(iter_engines: Engine, one_week_per_month_by_hour_table):
    df, _, schema = one_week_per_month_by_hour_table
    error = ()
    ingest_data_and_check(iter_engines, df, schema, error)


def test_one_week_per_month_by_hour_missing_data(
    iter_engines: Engine, one_week_per_month_by_hour_table
):
    df, _, schema = one_week_per_month_by_hour_table
    df2 = df.loc[df["hour"] != 0].copy()
    error = (InvalidTable, "Mismatch number of timestamps")
    ingest_data_and_check(iter_engines, df2, schema, error)


def test_consistent_time_nulls(iter_engines: Engine, one_week_per_month_by_hour_table):
    df, _, schema = one_week_per_month_by_hour_table
    df.loc[len(df)] = [4.0, None, None, None, None]
    error = ()
    ingest_data_and_check(iter_engines, df, schema, error)


def test_inconsistent_time_nulls(iter_engines: Engine, one_week_per_month_by_hour_table):
    df, _, schema = one_week_per_month_by_hour_table
    df.loc[len(df)] = [4.0, None, 1.0, 2.0, 0.345]
    error = (InvalidTable, "If any time columns have a NULL value for a row")
    ingest_data_and_check(iter_engines, df, schema, error)


def test_one_weekday_day_and_one_weekend_day_per_month_by_hour(
    iter_engines: Engine, one_weekday_day_and_one_weekend_day_per_month_by_hour_table
):
    df, _, schema = one_weekday_day_and_one_weekend_day_per_month_by_hour_table
    error = ()
    ingest_data_and_check(iter_engines, df, schema, error)


def test_one_weekday_day_and_one_weekend_day_per_month_by_hour_wrong_data(
    iter_engines: Engine, one_weekday_day_and_one_weekend_day_per_month_by_hour_table
):
    df, _, schema = one_weekday_day_and_one_weekend_day_per_month_by_hour_table
    df3 = df.copy()
    df3.loc[df3["month"] == 12, "month"] = 0
    error = (InvalidTable, "Actual timestamps do not match expected timestamps")
    ingest_data_and_check(iter_engines, df3, schema, error)
