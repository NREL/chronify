import fileinput
import gc
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import DateTime, Double, Engine, Table, create_engine, select

from chronify.csv_io import read_csv
from chronify.duckdb.functions import unpivot
from chronify.exceptions import (
    ConflictingInputsError,
    InvalidOperation,
    InvalidParameter,
    InvalidTable,
    TableNotStored,
)
from chronify.models import ColumnDType, CsvTableSchema, PivotedTableSchema, TableSchema
from chronify.store import Store
from chronify.time import TimeIntervalType
from chronify.time_configs import DatetimeRange
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time_series_checker import check_timestamp_lists
from chronify.utils.sql import make_temp_view_name


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
    df = store.read_table(dst_schema.name)
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
    df = store.read_table(dst_schema.name)
    assert len(df) == 8784 * 3 * 2
    all(df.timestamp.unique() == expected_timestamps)

    # Read a subset of the table.
    df2 = store.read_query(
        dst_schema.name, f"SELECT * FROM {dst_schema.name} WHERE generator = 'gen2'"
    )
    assert len(df2) == 8784
    df_gen2 = df[df["generator"] == "gen2"]
    assert all((df2.values == df_gen2.values)[0])

    # Adding the same rows should fail.
    with pytest.raises(InvalidTable):
        store.ingest_from_csv(new_file, src_schema2, dst_schema)


def test_ingest_multiple_tables(iter_engines: Engine):
    store = Store(engine=iter_engines)
    resolution = timedelta(hours=1)
    df_base = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", "2020-12-31 23:00:00", freq=resolution),
            "value": np.random.random(8784),
        }
    )
    df1 = df_base.copy()
    df2 = df_base.copy()
    df1["id"] = 1
    df2["id"] = 2
    store.ingest_tables(
        [df1, df2],
        TableSchema(
            name="devices",
            value_column="value",
            time_config=DatetimeRange(
                time_column="timestamp",
                start=datetime(2020, 1, 1, 0),
                length=8784,
                resolution=timedelta(hours=1),
            ),
            time_array_id_columns=["id"],
        ),
    )
    query = "SELECT * FROM devices WHERE id = ?"
    params = (2,)
    df = store.read_query("devices", query, params=params)
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
    assert df.equals(df2)


@pytest.mark.parametrize("use_pandas", [False, True])
def test_ingest_pivoted_table(iter_engines: Engine, generators_schema, use_pandas: bool):
    engine = iter_engines
    src_file, src_schema, dst_schema = generators_schema
    pivoted_schema = PivotedTableSchema(**src_schema.model_dump(exclude={"column_dtypes"}))
    rel = read_csv(src_file, src_schema)
    store = Store(engine=engine)
    input_table = rel.to_df() if use_pandas else rel
    store.ingest_pivoted_table(input_table, pivoted_schema, dst_schema)
    table = store.get_table(dst_schema.name)
    stmt = select(table).where(table.c.generator == "gen1")
    df = store.read_query(dst_schema.name, stmt)
    assert len(df) == 8784


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
    with pytest.raises(TableNotStored):
        store.read_table(dst_schema.name)


def test_invalid_schema(iter_engines: Engine, generators_schema):
    engine = iter_engines
    src_file, src_schema, dst_schema = generators_schema
    src_schema.value_columns = ["g1", "g2", "g3"]
    store = Store(engine=engine)
    with pytest.raises(InvalidTable):
        store.ingest_from_csv(src_file, src_schema, dst_schema)


def test_ingest_one_week_per_month_by_hour(iter_engines: Engine, one_week_per_month_by_hour_table):
    engine = iter_engines
    df, num_time_arrays, schema = one_week_per_month_by_hour_table

    store = Store(engine=engine)
    store.ingest_table(df, schema)
    df2 = store.read_table(schema.name)
    assert len(df2["id"].unique()) == num_time_arrays
    assert len(df2) == 24 * 7 * 12 * num_time_arrays
    columns = schema.time_config.list_time_columns()
    columns.insert(0, "value")
    assert all(df.sort_values(columns)["value"] == df2.sort_values(columns)["value"])


def test_ingest_one_week_per_month_by_hour_invalid(
    iter_engines: Engine, one_week_per_month_by_hour_table
):
    engine = iter_engines
    df, _, schema = one_week_per_month_by_hour_table
    df_filtered = df[df["hour"] != 5]
    assert len(df_filtered) < len(df)

    store = Store(engine=engine)
    with pytest.raises(InvalidTable):
        store.ingest_table(df_filtered, schema)


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
    df = store.read_table(dst_schema.name)
    assert len(df) == 8784 * 3
    timestamp_generator = make_time_range_generator(time_config)
    expected_timestamps = timestamp_generator.list_timestamps()
    all(df.timestamp.unique() == expected_timestamps)


@pytest.mark.parametrize(
    "params",
    [
        (True, 2020),
        (True, 2021),
        (False, 2020),
        (False, 2021),
    ],
)
def test_map_datetime_to_one_week_per_month_by_hour(
    iter_engines: Engine, one_week_per_month_by_hour_table, params: tuple[bool, int]
):
    engine = iter_engines
    use_time_zone, year = params
    df, num_time_arrays, src_schema = one_week_per_month_by_hour_table
    if use_time_zone:
        df["time_zone"] = df["id"].map(
            dict(zip([1, 2, 3], ["US/Central", "US/Mountain", "US/Pacific"]))
        )
        src_schema.time_array_id_columns += ["time_zone"]
    tzinfo = ZoneInfo("America/Denver") if use_time_zone else None
    time_array_len = 8784 if year % 4 else 8760
    dst_schema = TableSchema(
        name="ev_charging_datetime",
        value_column="value",
        time_config=DatetimeRange(
            time_column="timestamp",
            start=datetime(year, 1, 1, 0, tzinfo=tzinfo),
            length=time_array_len,
            resolution=timedelta(hours=1),
        ),
        time_array_id_columns=["id"],
    )
    store = Store(engine=engine)
    store.ingest_table(df, src_schema)
    store.map_table_time_config(src_schema.name, dst_schema)
    df2 = store.read_table(dst_schema.name)
    assert len(df2) == time_array_len * num_time_arrays
    actual = sorted(df2["timestamp"].unique())
    expected = make_time_range_generator(dst_schema.time_config).list_timestamps()
    if use_time_zone:
        expected = [pd.Timestamp(x) for x in expected]
    check_timestamp_lists(actual, expected)


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


def test_load_existing_store(iter_engines_file, one_week_per_month_by_hour_table):
    engine = iter_engines_file
    df, _, schema = one_week_per_month_by_hour_table
    store = Store(engine=engine)
    store.ingest_table(df, schema)
    df2 = store.read_table(schema.name)
    assert df2.equals(df)
    file_path = Path(engine.url.database)
    assert file_path.exists()
    store2 = Store.load_from_file(engine_name=engine.name, file_path=file_path)
    df3 = store2.read_table(schema.name)
    assert df3.equals(df2)


def test_create_methods(iter_engine_names, tmp_path):
    path = tmp_path / "data.db"
    assert not path.exists()
    Store.create_file_db(engine_name=iter_engine_names, file_path=path)
    gc.collect()
    assert path.exists()
    with pytest.raises(InvalidOperation):
        Store.create_file_db(engine_name=iter_engine_names, file_path=path)
    Store.create_file_db(engine_name=iter_engine_names, file_path=path, overwrite=True)
    Store.create_in_memory_db(engine_name=iter_engine_names)


def test_create_with_existing_engine():
    engine = create_engine("duckdb:///:memory:")
    store = Store(engine=engine)
    assert store.engine is engine


def test_create_with_sqlite():
    Store(engine_name="sqlite")


def test_create_with_conflicting_parameters():
    with pytest.raises(ConflictingInputsError):
        Store(engine=create_engine("duckdb:///:memory:"), engine_name="duckdb")


def test_backup(iter_engines_file: Engine, one_week_per_month_by_hour_table, tmp_path):
    engine = iter_engines_file
    df, _, schema = one_week_per_month_by_hour_table
    store = Store(engine=engine)
    store.ingest_table(df, schema)
    dst_file = tmp_path / "backup.db"
    assert not dst_file.exists()
    store.backup(dst_file)
    assert dst_file.exists()
    store2 = Store(engine_name=engine.name, file_path=dst_file)
    df2 = store2.read_table(schema.name)
    assert df2.equals(df)

    with pytest.raises(InvalidParameter):
        store.backup(dst_file)
    dst_file2 = tmp_path / "backup2.db"
    dst_file2.touch()
    store.backup(dst_file2, overwrite=True)

    # Make sure the original still works.
    df3 = store.read_table(schema.name)
    assert df3.equals(df)


def test_backup_not_allowed(one_week_per_month_by_hour_table, tmp_path):
    engine = create_engine("duckdb:///:memory:")
    df, _, schema = one_week_per_month_by_hour_table
    store = Store(engine=engine)
    store.ingest_table(df, schema)
    dst_file = tmp_path / "backup.db"
    assert not dst_file.exists()
    with pytest.raises(InvalidOperation):
        store.backup(dst_file)
    assert not dst_file.exists()


def test_delete_rows(iter_engines: Engine, one_week_per_month_by_hour_table):
    engine = iter_engines
    df, _, schema = one_week_per_month_by_hour_table
    store = Store(engine=engine)
    store.ingest_table(df, schema)
    df2 = store.read_table(schema.name)
    assert df2.equals(df)
    assert sorted(df2["id"].unique()) == [1, 2, 3]
    with pytest.raises(InvalidParameter):
        store.delete_rows(schema.name, {})
    store.delete_rows(schema.name, {"id": 2})
    df3 = store.read_table(schema.name)
    assert sorted(df3["id"].unique()) == [1, 3]
    store.delete_rows(schema.name, {"id": 1})
    df4 = store.read_table(schema.name)
    assert sorted(df4["id"].unique()) == [3]
    store.delete_rows(schema.name, {"id": 3})
    with pytest.raises(TableNotStored):
        store.read_table(schema.name)
    with pytest.raises(TableNotStored):
        store.delete_rows(schema.name, {"id": 3})


def test_drop_table(iter_engines: Engine, one_week_per_month_by_hour_table):
    engine = iter_engines
    df, _, schema = one_week_per_month_by_hour_table
    store = Store(engine=engine)
    assert not store.list_tables()
    store.ingest_table(df, schema)
    assert store.read_table(schema.name).equals(df)
    assert store.list_tables() == [schema.name]
    store.drop_table(schema.name)
    with pytest.raises(TableNotStored):
        store.read_table(schema.name)
    assert not store.list_tables()
    with pytest.raises(TableNotStored):
        store.drop_table(schema.name)


def test_drop_view(iter_engines: Engine, one_week_per_month_by_hour_table):
    engine = iter_engines
    df, _, schema = one_week_per_month_by_hour_table
    store = Store(engine=engine)
    store.ingest_table(df, schema)
    table = Table(schema.name, store.metadata)
    stmt = select(table).where(table.c.id == 1)
    inputs = schema.model_dump()
    inputs["name"] = make_temp_view_name()
    schema2 = TableSchema(**inputs)
    store.create_view(schema2, stmt)
    assert schema2.name in store.list_tables()
    store.drop_view(schema2.name)
    assert schema2.name not in store.list_tables()


def test_read_raw_query(iter_engines: Engine, one_week_per_month_by_hour_table):
    engine = iter_engines
    df, _, schema = one_week_per_month_by_hour_table
    store = Store(engine=engine)
    store.ingest_table(df, schema)

    query = f"SELECT * FROM {schema.name}"
    df2 = store.read_raw_query(query)
    assert df2.equals(df)

    query = f"SELECT * FROM {schema.name} where id = ?"
    params = (2,)
    df2 = store.read_raw_query(query, params=params)
    assert df2.equals(df[df["id"] == 2].reset_index(drop=True))
