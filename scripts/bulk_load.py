"""Runs database write and read tests with many tables.

Findings:
- DuckDB is significantly faster than SQLite for both data ingestion and reads.
- SQLite benefits significantly from indexes.

Usage: python scripts/perf_many_tables.py

Results on 12/27/2024:
Run test with engine_url='duckdb:///:memory:' count=1000 length=8760 create_index=True
duration_write=1.0231969356536865
duration_read=5.646288871765137 avg=0.005646288871765137
Run test with engine_url='sqlite:///:memory:' count=1000 length=8760 create_index=True
duration_write=46.84509587287903
duration_read=14.509371757507324 avg=0.014509371757507324
Run test with engine_url='duckdb:///:memory:' count=1000 length=8760 create_index=False
duration_write=1.1198809146881104
duration_read=0.4036238193511963 avg=0.0004036238193511963
Run test with engine_url='sqlite:///:memory:' count=1000 length=8760 create_index=False
duration_write=46.14994025230408
duration_read=199.04997396469116 avg=0.19904997396469115
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select

from chronify.models import TableSchema
from chronify.store import Store
from chronify.time_configs import DatetimeRange


GENERATOR_TIME_SERIES_FILE = "tests/data/gen.csv"


def make_tables(
    resolution: timedelta, length: int, count: int
) -> Generator[pd.DataFrame, None, None]:
    timestamps = pd.date_range(datetime(2020, 1, 1), periods=length, freq=resolution)

    for i in range(count):
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "value": np.random.random(length),
            }
        )
        df["id"] = i + 1
        yield df


def run_test(engine_url: str, length: int, count: int, create_index: bool):
    time_config = DatetimeRange(
        time_column="timestamp",
        start=datetime(2020, 1, 1, 0),
        length=length,
        resolution=timedelta(hours=1),
    )
    schema = TableSchema(
        name="devices",
        value_column="value",
        time_config=time_config,
        time_array_id_columns=["id"],
    )
    engine = create_engine(engine_url)
    store = Store(engine=engine)
    print(f"Run test with {engine_url=} {count=} {length=} {create_index=}")
    start = time.time()
    store.ingest_tables(make_tables(time_config.resolution, length, count), schema)
    duration_write = time.time() - start
    print(f"{duration_write=}")
    if create_index:
        store.create_index(schema.name)

    table = store.get_table(schema.name)
    start = time.time()
    with engine.connect() as conn:
        for i in range(count):
            ts_id = i + 1
            stmt = select(table).where(table.c.id == ts_id)
            store.read_query(schema.name, stmt, connection=conn)
    duration_read = time.time() - start
    avg = duration_read / count
    print(f"{duration_read=} {avg=}")


duckdb_file = Path("duckdb_many_tables.db")
sqlite_file = Path("sqlite_many_tables.db")
files = (duckdb_file, sqlite_file)
for f in files:
    if f.exists():
        f.unlink()

length = 8760
count = 1000
for create_index in (True, False):
    run_test("duckdb:///:memory:", length, count, create_index)
    # run_test(f"duckdb:///{duckdb_file}", length, count, create_index)
    run_test("sqlite:///:memory:", length, count, create_index)
    # run_test(f"sqlite:///{sqlite_file}", length, count, create_index)

for f in files:
    if f.exists():
        f.unlink()
