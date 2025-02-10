import pandas as pd
from sqlalchemy import Engine, MetaData
import pytest
from datetime import timedelta
from typing import Any, Optional
import numpy as np

from chronify.sqlalchemy.functions import read_database, write_database
from chronify.time_series_mapper import map_time
from chronify.time_configs import (
    DatetimeRange,
    IndexTimeRangeNTZ,
    IndexTimeRangeTZ,
    IndexTimeRangeLocalTime,
    TimeBasedDataAdjustment,
)
from chronify.models import TableSchema
from chronify.time import TimeIntervalType, DaylightSavingAdjustmentType


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


def data_for_simple_mapping(
    tz_naive: bool = False,
) -> tuple[pd.DataFrame, TableSchema, TableSchema]:
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

    dst_schema = output_dst_schema()
    return src_df, src_schema, dst_schema


def data_for_unaligned_time_mapping() -> tuple[pd.DataFrame, TableSchema, TableSchema]:
    src_df = pd.DataFrame(
        {
            "index_time": np.tile(range(1, 8761), 2),
            "value": np.concatenate([range(1, 8761), range(10, 87610, 10)]),
            "time_zone": np.repeat(["US/Mountain", "US/Central"], 8760),
        }
    )

    time_config = IndexTimeRangeLocalTime(
        start=1,
        length=8760,
        start_timestamp=pd.Timestamp("2018-01-01 00:00"),
        resolution=timedelta(hours=1),
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_column="index_time",
        time_zone_column="time_zone",
    )
    src_schema = TableSchema(
        name="input_data",
        time_config=time_config,
        time_array_id_columns=[],
        value_column="value",
    )
    dst_schema = output_dst_schema()
    dst_schema.time_array_id_columns = ["time_zone"]
    return src_df, src_schema, dst_schema


def run_test(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
    to_schema: TableSchema,
    error: tuple[Any, str],
    data_adjustment: Optional[TimeBasedDataAdjustment] = None,
    wrap_time_allowed: bool = False,
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
            map_time(
                engine,
                metadata,
                from_schema,
                to_schema,
                data_adjustment=data_adjustment,
                wrap_time_allowed=wrap_time_allowed,
                check_mapped_timestamps=True,
            )
    else:
        map_time(
            engine,
            metadata,
            from_schema,
            to_schema,
            data_adjustment=data_adjustment,
            wrap_time_allowed=wrap_time_allowed,
            check_mapped_timestamps=True,
        )


def get_output_table(engine: Engine, to_schema: TableSchema) -> pd.DataFrame:
    with engine.connect() as conn:
        query = f"select * from {to_schema.name}"
        queried = read_database(query, conn, to_schema.time_config)
    return queried


@pytest.mark.parametrize("tz_naive", [True, False])
def test_simple_mapping(iter_engines: Engine, tz_naive: bool) -> None:
    src_df, src_schema, dst_schema = data_for_simple_mapping(tz_naive=tz_naive)
    error = ()
    run_test(iter_engines, src_df, src_schema, dst_schema, error)

    dfo = get_output_table(iter_engines, dst_schema)
    assert sorted(dfo["value"]) == sorted(src_df["value"])


def test_unaligned_time_mapping(iter_engines: Engine) -> None:
    src_df, src_schema, dst_schema = data_for_unaligned_time_mapping()
    error = ()
    data_adjustment = None
    wrap_time_allowed = True
    run_test(
        iter_engines,
        src_df,
        src_schema,
        dst_schema,
        error,
        data_adjustment=data_adjustment,
        wrap_time_allowed=wrap_time_allowed,
    )

    dfo = get_output_table(iter_engines, dst_schema)
    assert sorted(dfo["value"]) == sorted(src_df["value"])


def test_industrial_time_mapping(iter_engines: Engine) -> None:
    """I.e., unaligned time mapping with data_adjustment"""
    src_df, src_schema, dst_schema = data_for_unaligned_time_mapping()
    error = ()
    data_adjustment = TimeBasedDataAdjustment(
        daylight_saving_adjustment=DaylightSavingAdjustmentType.DROP_SPRINGFORWARD_DUPLICATE_FALLBACK
    )
    wrap_time_allowed = True
    run_test(
        iter_engines,
        src_df,
        src_schema,
        dst_schema,
        error,
        data_adjustment=data_adjustment,
        wrap_time_allowed=wrap_time_allowed,
    )

    dfo = get_output_table(iter_engines, dst_schema)
    dfo = dfo.sort_values(by=["time_zone", "value"]).reset_index(drop=True)

    # Check value associated with springforward hour is dropped
    # If the following fail, print to debug: dfo.loc[1656:1659], dfo.loc[1656+8760:1659+8760]
    assert 1659 not in dfo["value"].values
    assert 16590 not in dfo["value"].values

    # Check value associated with fallback hour is duplicated
    breakpoint()
    assert dfo.loc[7367:7370]["value"].value_counts()[73700] == 2
    assert dfo.loc[7367 + 8760 : 7370 + 8760]["value"].value_counts()[7370] == 2
