import fileinput
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import pytest
import sqlalchemy
from sqlalchemy import DateTime, Double, Engine, Table, create_engine, select
from chronify.csv_io import read_csv
from chronify.duckdb.functions import unpivot
from chronify.exceptions import ConflictingInputsError, InvalidTable
from chronify.models import ColumnDType, CsvTableSchema, TableSchema
from chronify.store import Store
from chronify.time import TimeIntervalType
from chronify.time_configs import DatetimeRange
from chronify.time_range_generator_factory import make_time_range_generator


GENERATOR_TIME_SERIES_FILE = "tests/data/gen.csv"


@pytest.fixture
def generators_schema():
    time_config = DatetimeRange(
        start=datetime(year=2020, month=1, day=1, tzinfo=ZoneInfo("EST")),
        resolution=timedelta(hours=1),
        length=8784,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_column="timestamp",
    )

    src_schema = CsvTableSchema(
        time_config=time_config,
        column_dtypes=[
            ColumnDType(name="timestamp", dtype=DateTime(timezone=False)),
            ColumnDType(name="gen1", dtype=Double()),
            ColumnDType(name="gen2", dtype=Double()),
            ColumnDType(name="gen3", dtype=Double()),
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
    yield Path(GENERATOR_TIME_SERIES_FILE), src_schema, dst_schema


@pytest.mark.parametrize("use_time_zone", [True, False])
def test_ingest_csv(iter_engines: Engine, tmp_path, generators_schema, use_time_zone):
    engine = iter_engines
    src_file, src_schema, dst_schema = generators_schema
    src_schema.column_dtypes[0] = ColumnDType(
        name="timestamp", dtype=DateTime(timezone=use_time_zone)
    )
    store = Store(engine=engine)
    if use_time_zone:
        new_src_file = tmp_path / "gen_tz.csv"
        duckdb.sql(
            f"""
            SELECT timezone('EST', timestamp) as timestamp, gen1, gen2, gen3
            FROM read_csv('{src_file}')
        """
        ).to_df().to_csv(new_src_file, index=False)
        src_file = new_src_file
    store.ingest_from_csv(src_file, src_schema, dst_schema)
    df = store.read_table(dst_schema)
    assert len(df) == 8784 * 3

    new_file = tmp_path / "gen2.csv"
    shutil.copyfile(src_file, new_file)
    with fileinput.input([new_file], inplace=True) as f:
        for line in f:
            new_line = line.replace("gen1", "g1b").replace("gen2", "g2b").replace("gen3", "g3b")
            print(new_line, end="")

    timestamp_generator = make_time_range_generator(dst_schema.time_config)
    expected_timestamps = timestamp_generator.list_timestamps()

    # Test addition of new generators to the same table.
    src_schema2 = CsvTableSchema(
        time_config=src_schema.time_config,
        column_dtypes=[
            ColumnDType(name="timestamp", dtype=DateTime(timezone=use_time_zone)),
            ColumnDType(name="g1b", dtype=Double()),
            ColumnDType(name="g2b", dtype=Double()),
            ColumnDType(name="g3b", dtype=Double()),
        ],
        value_columns=["g1b", "g2b", "g3b"],
        pivoted_dimension_name="generator",
        time_array_id_columns=[],
    )
    store.ingest_from_csv(new_file, src_schema2, dst_schema)
    df = store.read_table(dst_schema)
    assert len(df) == 8784 * 3 * 2
    all(df.timestamp.unique() == expected_timestamps)

    # Adding the same rows should fail.
    with pytest.raises(InvalidTable):
        store.ingest_from_csv(new_file, src_schema2, dst_schema)
        df = store.read_table(dst_schema)
        assert len(df) == 8784 * 3 * 2
        all(df.timestamp.unique() == expected_timestamps)


def test_ingest_invalid_csv(iter_engines: Engine, tmp_path, generators_schema):
    engine = iter_engines
    src_file, src_schema, dst_schema = generators_schema
    lines = src_file.read_text().splitlines()[:-10]
    new_file = tmp_path / "data.csv"
    with open(new_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")

    store = Store(engine=engine)
    with pytest.raises(InvalidTable):
        store.ingest_from_csv(new_file, src_schema, dst_schema)
    with pytest.raises((sqlalchemy.exc.OperationalError, sqlalchemy.exc.ProgrammingError)):
        store.read_table(dst_schema)


def test_invalid_schema(iter_engines: Engine, generators_schema):
    engine = iter_engines
    src_file, src_schema, dst_schema = generators_schema
    src_schema.value_columns = ["g1", "g2", "g3"]
    store = Store(engine=engine)
    with pytest.raises(InvalidTable):
        store.ingest_from_csv(src_file, src_schema, dst_schema)


def test_load_parquet(tmp_path):
    time_config = DatetimeRange(
        start=datetime(year=2020, month=1, day=1, tzinfo=ZoneInfo("EST")),
        resolution=timedelta(hours=1),
        length=8784,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_column="timestamp",
    )

    src_schema = CsvTableSchema(
        time_config=time_config,
        column_dtypes=[
            ColumnDType(name="timestamp", dtype=DateTime(timezone=False)),
            ColumnDType(name="gen1", dtype=Double()),
            ColumnDType(name="gen2", dtype=Double()),
            ColumnDType(name="gen3", dtype=Double()),
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
    out_file = tmp_path / "gen2.parquet"
    rel2.to_parquet(str(out_file))
    store = Store()
    store.load_table(out_file, dst_schema)
    df = store.read_table(dst_schema)
    assert len(df) == 8784 * 3
    timestamp_generator = make_time_range_generator(time_config)
    expected_timestamps = timestamp_generator.list_timestamps()
    all(df.timestamp.unique() == expected_timestamps)


def test_to_parquet(tmp_path, generators_schema):
    src_file, src_schema, dst_schema = generators_schema
    store = Store()
    store.ingest_from_csv(src_file, src_schema, dst_schema)
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
