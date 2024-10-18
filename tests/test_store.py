import fileinput
import shutil

from datetime import datetime, timedelta

import duckdb
import pandas as pd
import pytest
from sqlalchemy import Double, Table, create_engine, select
from chronify.csv_io import read_csv
from chronify.duckdb.functions import unpivot
from chronify.exceptions import ConflictingInputsError, InvalidTable
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
    yield src_schema, dst_schema


def test_ingest_csv(tmp_path, generators_schema):
    src_schema, dst_schema = generators_schema
    store = Store()
    store.ingest_from_csv(GENERATOR_TIME_SERIES_FILE, src_schema, dst_schema)
    df = store.read_table(dst_schema.name)
    assert len(df) == 8784 * 3

    new_file = tmp_path / "gen2.csv"
    shutil.copyfile(GENERATOR_TIME_SERIES_FILE, new_file)
    with fileinput.input([new_file], inplace=True) as f:
        for line in f:
            new_line = line.replace("gen1", "g1b").replace("gen2", "g2b").replace("gen3", "g3b")
            print(new_line, end="")

    # Test addition of new generators to the same table.
    src_schema2 = CsvTableSchema(
        time_config=src_schema.time_config,
        column_dtypes=[
            ColumnDType(name="g1b", dtype=Double),
            ColumnDType(name="g2b", dtype=Double),
            ColumnDType(name="g3b", dtype=Double),
        ],
        value_columns=["g1b", "g2b", "g3b"],
        pivoted_dimension_name="generator",
        time_array_id_columns=[],
    )
    store.ingest_from_csv(new_file, src_schema2, dst_schema)
    df = store.read_table(dst_schema.name)
    assert len(df) == 8784 * 3 * 2


def test_invalid_schema(generators_schema):
    src_schema, dst_schema = generators_schema
    src_schema.value_columns = ["g1", "g2", "g3"]
    store = Store()
    with pytest.raises(InvalidTable):
        store.ingest_from_csv(GENERATOR_TIME_SERIES_FILE, src_schema, dst_schema)


def test_load_parquet(tmp_path):
    time_config = DatetimeRange(
        start=datetime(year=2020, month=1, day=1),
        resolution=timedelta(hours=1),
        length=8784,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
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


def test_to_parquet(tmp_path, generators_schema):
    src_schema, dst_schema = generators_schema
    store = Store()
    store.ingest_from_csv(GENERATOR_TIME_SERIES_FILE, src_schema, dst_schema)
    filename = tmp_path / "data.parquet"
    table = Table(dst_schema.name, store.metadata)
    stmt = select(table).where(table.c.generator == "gen2")
    store.write_query_to_parquet(stmt, filename)
    assert filename.exists()
    df = pd.read_parquet(filename)
    assert len(df) == 8784


def test_create_with_existing_engine():
    engine = create_engine("duckdb:///:memory")
    store = Store(engine=engine)
    assert store.engine is engine


def test_create_with_sqlite():
    Store(engine_name="sqlite")


def test_create_with_conflicting_parameters():
    with pytest.raises(ConflictingInputsError):
        Store(engine=create_engine("duckdb:///:memory"), engine_name="duckdb")
