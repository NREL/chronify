from datetime import datetime, timedelta

import duckdb
import pandas as pd
import polars as pl
from IPython import get_ipython

from sqlalchemy import Double, text
from chronify.models import ColumnDType, CsvTableSchema, TableSchema
from chronify.store import Store
from chronify.time import TimeIntervalType, TimeZone
from chronify.time_configs import DatetimeRange


GENERATOR_TIME_SERIES_FILE = "tests/data/gen.csv"


def read_duckdb(conn, name: str):
    return conn.sql(f"SELECT * FROM {name}").df()


def read_pandas(store: Store, name: str):
    with store.engine.begin() as conn:
        query = f"select * from {name}"
        return pd.read_sql(query, conn)


def read_polars(store: Store, name: str):
    with store.engine.begin() as conn:
        query = f"select * from {name}"
        return pl.read_database(query, connection=conn).to_pandas()


def read_sqlalchemy(store: Store, name: str):
    with store.engine.begin() as conn:
        query = f"select * from {name}"
        res = conn.execute(text(query)).fetchall()
        return pd.DataFrame.from_records(res, columns=["timestamp", "generator", "value"])


def setup():
    time_config = DatetimeRange(
        start=datetime(year=2020, month=1, day=1),
        resolution=timedelta(hours=1),
        length=8784,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_columns=["timestamp"],
        time_zone=TimeZone.UTC,
    )

    src_schema = CsvTableSchema(
        time_config=time_config,
        column_dtypes=[
            ColumnDType(name="gen1", dtype=Double),
            ColumnDType(name="gen2", dtype=Double),
            ColumnDType(name="gen3", dtype=Double),
        ],
        value_columns=["gen1", "gen2", "gen3"],
        pivoted_dimension_name="generator",
        time_array_id_columns=[],
    )
    dst_schema = TableSchema(
        name="generators",
        time_config=time_config,
        time_array_id_columns=["generator"],
        value_column="value",
    )
    return src_schema, dst_schema


def run_test(engine_name: str):
    store = Store(engine_name=engine_name)
    src_schema, dst_schema = setup()
    store.ingest_from_csv(GENERATOR_TIME_SERIES_FILE, src_schema, dst_schema)
    ipython = get_ipython()
    df = read_polars(store, dst_schema.name)  # noqa: F841
    conn = duckdb.connect(":memory:")
    conn.sql("CREATE OR REPLACE TABLE perf_test AS SELECT * from df")
    print(f"Run {engine_name} database with read_duckdb.")
    ipython.run_line_magic("timeit", "read_duckdb(conn, 'perf_test')")
    print(f"Run {engine_name} database with read_pandas.")
    ipython.run_line_magic("timeit", "read_pandas(store, dst_schema.name)")
    print(f"Run {engine_name} database with read_polars.")
    ipython.run_line_magic("timeit", "read_polars(store, dst_schema.name)")
    print(f"Run {engine_name} database with read_sqlalchemy.")
    ipython.run_line_magic("timeit", "read_sqlalchemy(store, dst_schema.name)")


run_test("duckdb")
run_test("sqlite")
