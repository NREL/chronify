import pandas as pd
from sqlalchemy import Engine, MetaData
import pytest
from datetime import timedelta
from typing import Any

from chronify.sqlalchemy.functions import read_database, write_database
from chronify.time_series_mapper import map_time
from chronify.time_configs import DatetimeRange, IndexTimeRangeNTZ, IndexTimeRangeTZ
from chronify.models import TableSchema
from chronify.time import TimeIntervalType


def output_dst_schema() -> TableSchema:
    return TableSchema(
        name="output_data",
        time_config=DatetimeRange(
            start=pd.Timestamp("2018-01-01 01:00", tz="US/Mountain"),
            resolution=timedelta(hours=1),
            length=8760,
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
            time_column="timestamp",
        ),
        time_array_id_columns=[],
        value_column="value",
    )


def input_for_simple_mapping(tz_naive=False) -> tuple[pd.DataFrame, TableSchema]:
    src_df = pd.DataFrame({"index_time": range(1, 8761), "value": range(1, 8761)})

    if tz_naive:
        time_config = IndexTimeRangeNTZ(
            start=1,
            length=8760,
            start_timestamp=pd.Timestamp("2018-01-01 00:00"),
            resolution=timedelta(hours=1),
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
            time_column="index_time",
        )
    else:
        time_config = IndexTimeRangeTZ(
            start=1,
            length=8760,
            start_timestamp=pd.Timestamp("2018-01-01 00:00", tz="US/Mountain"),
            resolution=timedelta(hours=1),
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
            time_column="index_time",
        )
    src_schema = TableSchema(
        name="input_data",
        time_config=time_config,
        time_array_id_columns=[],
        value_column="value",
    )
    return src_df, src_schema


def run_test(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
    to_schema: TableSchema,
    error: tuple[Any, str],
) -> None:
    # Ingest
    metadata = MetaData()
    with engine.begin() as conn:
        write_database(
            df, conn, from_schema.name, [from_schema.time_config], if_table_exists="replace"
        )
    metadata.reflect(engine, views=True)

    # Map
    if error:
        with pytest.raises(error[0], match=error[1]):
            map_time(engine, metadata, from_schema, to_schema, check_mapped_timestamps=True)
    else:
        map_time(engine, metadata, from_schema, to_schema, check_mapped_timestamps=True)


def get_output_table(engine: Engine, to_schema: TableSchema) -> pd.DataFrame:
    with engine.connect() as conn:
        query = f"select * from {to_schema.name}"
        queried = read_database(query, conn, to_schema.time_config)
    return queried


@pytest.mark.parametrize("tz_naive", [True, False])
def test_simple_mapping(iter_engines: Engine, tz_naive: bool) -> None:
    src_df, src_schema = input_for_simple_mapping(tz_naive=tz_naive)
    dst_schema = output_dst_schema()
    error = ()
    run_test(iter_engines, src_df, src_schema, dst_schema, error)

    dfo = get_output_table(iter_engines, dst_schema)
    assert sorted(dfo["value"]) == sorted(src_df["value"])
