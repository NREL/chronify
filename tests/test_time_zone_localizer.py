from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, tzinfo
import numpy as np
import pytest
from typing import Any

import pandas as pd
from sqlalchemy import Engine, MetaData

from chronify.sqlalchemy.functions import read_database, write_database
from chronify.time_zone_localizer import (
    TimeZoneLocalizer,
    TimeZoneLocalizerByColumn,
    localize_time_zone,
    localize_time_zone_by_column,
)
from chronify.time_configs import DatetimeRange, DatetimeRangeWithTZColumn
from chronify.models import TableSchema
from chronify.time import TimeDataType, TimeIntervalType
from chronify.datetime_range_generator import (
    DatetimeRangeGenerator,
    DatetimeRangeGeneratorExternalTimeZone,
)
from chronify.exceptions import InvalidParameter


def generate_datetime_dataframe(schema: TableSchema) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            schema.time_config.time_column: pd.to_datetime(
                DatetimeRangeGenerator(schema.time_config).list_timestamps()
            )
        }
    )
    for i, x in enumerate(schema.time_array_id_columns):
        df[x] = i
    df[schema.value_column] = np.random.rand(len(df))
    return df


def generate_dataframe_with_tz_col(schema: TableSchema) -> pd.DataFrame:
    time_col = schema.time_config.time_column
    ts_dct = DatetimeRangeGeneratorExternalTimeZone(
        schema.time_config
    ).list_timestamps_by_time_zone()
    dfo_list = []
    for i, (time_zone, data) in enumerate(ts_dct.items()):
        dfo_list.append(
            pd.DataFrame(
                {x: i for x in schema.time_array_id_columns}
                | {
                    time_col: pd.to_datetime(data).tz_localize(None),
                    "time_zone": time_zone,
                    schema.value_column: np.random.rand(len(data)),
                }
            )
        )
    dfo = pd.concat(dfo_list, ignore_index=True)
    dfo = dfo.reset_index()
    return dfo


def get_datetime_schema(
    year: int,
    tzinfo: tzinfo | None,
    interval_type: TimeIntervalType,
    name: str,
    has_tz_col: bool = False,
) -> TableSchema:
    start = datetime(year=year, month=1, day=1, tzinfo=tzinfo)
    end = datetime(year=year, month=1, day=2, tzinfo=tzinfo)
    resolution = timedelta(hours=1)
    length = (end - start) / resolution + 1
    cols = ["id"]
    # cols += ["time_zone"] if has_tz_col else []
    if has_tz_col:
        time_config = DatetimeRangeWithTZColumn(
            dtype=TimeDataType.TIMESTAMP_NTZ,
            start=start,
            resolution=resolution,
            length=length,
            interval_type=interval_type,
            time_column="timestamp",
            time_zone_column="time_zone",
            time_zones=[
                ZoneInfo("US/Eastern"),
                ZoneInfo("US/Central"),
                ZoneInfo("US/Mountain"),
                # None,
            ],
        )
    else:
        time_config = DatetimeRange(
            dtype=TimeDataType.TIMESTAMP_TZ if tzinfo else TimeDataType.TIMESTAMP_NTZ,
            start=start,
            resolution=resolution,
            length=length,
            interval_type=interval_type,
            time_column="timestamp",
        )
    schema = TableSchema(
        name=name,
        time_config=time_config,
        time_array_id_columns=cols,
        value_column="value",
    )
    return schema


def ingest_data(
    engine: Engine,
    metadata: MetaData,
    df: pd.DataFrame,
    schema: TableSchema,
) -> None:
    with engine.begin() as conn:
        write_database(df, conn, schema.name, [schema.time_config], if_table_exists="replace")
    metadata.reflect(engine, views=True)


def get_mapped_dataframe(
    engine: Engine,
    table_name: str,
    time_config: DatetimeRange,
) -> pd.DataFrame:
    with engine.connect() as conn:
        query = f"select * from {table_name}"
        queried = read_database(query, conn, time_config)
    queried = queried.sort_values(by=["id", "timestamp"]).reset_index(drop=True)
    return queried


def run_localization(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
    to_time_zone: tzinfo | None,
) -> None:
    metadata = MetaData()
    ingest_data(engine, metadata, df, from_schema)
    to_schema = localize_time_zone(
        engine, metadata, from_schema, to_time_zone, check_mapped_timestamps=True
    )
    dfo = get_mapped_dataframe(engine, to_schema.name, to_schema.time_config)
    assert df["value"].equals(dfo["value"])
    if to_time_zone is None:
        expected = df["timestamp"]
    else:
        expected = df["timestamp"].dt.tz_localize(to_time_zone)
    assert (dfo["timestamp"] == expected).prod() == 1


def run_localization_to_column_time_zones(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
) -> None:
    metadata = MetaData()
    ingest_data(engine, metadata, df, from_schema)
    to_schema = localize_time_zone_by_column(
        engine,
        metadata,
        from_schema,
        "time_zone",
        check_mapped_timestamps=True,
    )
    dfo = get_mapped_dataframe(engine, to_schema.name, to_schema.time_config)
    dfo = dfo[df.columns].sort_values(by="index").reset_index(drop=True)
    dfo["timestamp"] = pd.to_datetime(dfo["timestamp"])  # needed for engine 2, not sure why

    assert df["value"].equals(dfo["value"])
    for i in range(len(dfo)):
        tzn = dfo.loc[i, "time_zone"]
        if tzn == "None":
            ts = dfo.loc[i, "timestamp"].replace(tzinfo=None)
        else:
            ts = dfo.loc[i, "timestamp"].tz_convert(ZoneInfo(tzn)).replace(tzinfo=None)
        assert df.loc[i, "timestamp"] == ts


def run_localization_with_error(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
    use_tz_col: bool,
    error: tuple[Any, str],
) -> None:
    metadata = MetaData()
    ingest_data(engine, metadata, df, from_schema)
    with pytest.raises(error[0], match=error[1]):
        if use_tz_col:
            tzl = TimeZoneLocalizerByColumn(
                engine,
                metadata,
                from_schema,
                "time_zone",
            )
            tzl.localize_time_zone(check_mapped_timestamps=True)
        else:
            tzl2 = TimeZoneLocalizer(engine, metadata, from_schema, None)
            tzl2.localize_time_zone(check_mapped_timestamps=True)


def test_src_table_not_tz_naive(iter_engines: Engine) -> None:
    from_schema = get_datetime_schema(
        2018, ZoneInfo("US/Mountain"), TimeIntervalType.PERIOD_BEGINNING, "base_table"
    )
    df = generate_datetime_dataframe(from_schema)
    error = (InvalidParameter, "Source schema time config start time must be tz-naive.")
    run_localization_with_error(
        iter_engines, df, from_schema, False, error
    )  # TODO, support tz-naive to tz-aware conversion


@pytest.mark.parametrize(
    "to_time_zone", [None, ZoneInfo("US/Central"), ZoneInfo("America/Los_Angeles")]
)
def test_time_localization(iter_engines: Engine, to_time_zone: tzinfo | None) -> None:
    from_schema = get_datetime_schema(2018, None, TimeIntervalType.PERIOD_BEGINNING, "base_table")
    df = generate_datetime_dataframe(from_schema)
    run_localization(iter_engines, df, from_schema, to_time_zone)


@pytest.mark.parametrize("from_time_tz", [None, ZoneInfo("US/Mountain")])
def test_time_localization_to_column_time_zones(
    iter_engines: Engine, from_time_tz: tzinfo | None
) -> None:
    from_schema = get_datetime_schema(
        2018,
        from_time_tz,
        TimeIntervalType.PERIOD_BEGINNING,
        "base_table",
        has_tz_col=True,
    )
    df = generate_dataframe_with_tz_col(from_schema)
    run_localization_to_column_time_zones(iter_engines, df, from_schema)
