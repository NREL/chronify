from datetime import datetime, timedelta

import duckdb
import pytest
from sqlalchemy import Double
from chronify.csv_io import read_csv
from chronify.duckdb.functions import unpivot

from chronify.models import ColumnDType, CsvTableSchema, TableSchema
from chronify.store import Store
from chronify.time import TimeIntervalType, TimeZone
from chronify.time_configs import DatetimeRange


GENERATOR_TIME_SERIES_FILE = "tests/data/gen.csv"


@pytest.fixture
def generators_schema():
    time_config = DatetimeRange(
        start=datetime(year=2020, month=1, day=1),
        resolution=timedelta(hours=1),
        length=8784,
        time_interval_type=TimeIntervalType.PERIOD_BEGINNING,
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
    yield src_schema, dst_schema


def test_ingest_csv(generators_schema):
    src_schema, dst_schema = generators_schema
    store = Store()
    store.ingest_from_csv(GENERATOR_TIME_SERIES_FILE, src_schema, dst_schema)


def test_load_parquet(tmp_path):
    time_config = DatetimeRange(
        start=datetime(year=2020, month=1, day=1),
        resolution=timedelta(hours=1),
        length=8784,
        time_interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_columns=["timestamp"],
        time_zone=TimeZone.EST,
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
    rel = read_csv(GENERATOR_TIME_SERIES_FILE, src_schema)
    rel2 = unpivot(rel, ("gen1", "gen2", "gen3"), "generator", "value")  # noqa: F841
    rel3 = duckdb.sql(
        """
            SELECT CAST(timezone('EST', timestamp) AS TIMESTAMPTZ) AS timestamp
            ,generator
            ,value from rel2
        """
    )
    out_file = tmp_path / "gen.parquet"
    rel3.to_parquet(str(out_file))
    store = Store()
    store.load_table(out_file, dst_schema)
