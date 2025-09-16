from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import numpy as np

import pandas as pd
from sqlalchemy import Engine, MetaData

from chronify.sqlalchemy.functions import read_database, write_database
from chronify.time_zone_converter import TimeZoneConverter, TimeZoneConverterByGeography
from chronify.time_configs import DatetimeRange
from chronify.models import TableSchema
from chronify.time import TimeIntervalType
from chronify.datetime_range_generator import DatetimeRangeGenerator


def generate_datetime_data(time_config: DatetimeRange) -> pd.Series:  # type: ignore
    return pd.to_datetime(list(DatetimeRangeGenerator(time_config).iter_timestamps()))


def generate_datetime_dataframe(schema: TableSchema) -> pd.DataFrame:
    df = pd.DataFrame({schema.time_config.time_column: generate_datetime_data(schema.time_config)})

    for i, x in enumerate(schema.time_array_id_columns):
        df[x] = i
    df[schema.value_column] = np.random.rand(len(df))
    return df


def generate_dataframe_with_tz_col(schema: TableSchema) -> pd.DataFrame:
    df = generate_datetime_dataframe(schema).drop(columns=["id"])
    time_zones = [
        ZoneInfo("US/Eastern"),
        ZoneInfo("US/Central"),
        ZoneInfo("US/Mountain"),
    ]  # , None]
    time_zones = [tz.key if tz is not None else "None" for tz in time_zones]
    dfo = pd.merge(
        df, pd.DataFrame({"id": range(len(time_zones)), "time_zone": time_zones}), how="cross"
    )
    dfo = (
        dfo.drop(columns=["time_zone_x"])
        .rename(columns={"time_zone_y": "time_zone"})
        .reset_index()
    )
    return dfo


def get_datetime_schema(
    year: int,
    tzinfo: ZoneInfo | None,
    interval_type: TimeIntervalType,
    name: str,
    has_tz_col: bool = False,
) -> TableSchema:
    start = datetime(year=year, month=1, day=1, tzinfo=tzinfo)
    end = datetime(year=year, month=1, day=2, tzinfo=tzinfo)
    resolution = timedelta(hours=1)
    length = (end - start) / resolution + 1
    cols = ["id"]
    cols += ["time_zone"] if has_tz_col else []
    schema = TableSchema(
        name=name,
        time_config=DatetimeRange(
            start=start,
            resolution=resolution,
            length=length,
            interval_type=interval_type,
            time_column="timestamp",
        ),
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


def run_conversion(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
    to_time_zone: ZoneInfo | None,
) -> None:
    metadata = MetaData()
    ingest_data(engine, metadata, df, from_schema)
    TZC = TimeZoneConverter(engine, metadata, from_schema, to_time_zone)
    TZC.convert_time_zone(check_mapped_timestamps=True)
    dfo = get_mapped_dataframe(engine, TZC._to_schema.name, TZC._to_schema.time_config)

    assert (df["timestamp"] == dfo["timestamp"]).prod() == 1  # TODO: these will always be equal


def run_conversion_by_geography(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
) -> None:
    metadata = MetaData()
    ingest_data(engine, metadata, df, from_schema)
    TZC = TimeZoneConverterByGeography(engine, metadata, from_schema, "time_zone")
    TZC.convert_time_zone()
    dfo = get_mapped_dataframe(engine, TZC._to_schema.name, TZC._to_schema.time_config)
    dfo = dfo[df.columns].sort_values(by="index").reset_index(drop=True)

    assert df["value"].equals(dfo["value"])
    for i in range(len(df)):
        tz = ZoneInfo(df.loc[i, "time_zone"])
        ts = df.loc[i, "timestamp"].tz_convert(tz).replace(tzinfo=None)
        assert dfo.loc[i, "timestamp"] == ts


def test_time_conversion(iter_engines: Engine) -> None:
    from_schema = get_datetime_schema(
        2018, ZoneInfo("US/Mountain"), TimeIntervalType.PERIOD_BEGINNING, "base_table"
    )
    df = generate_datetime_dataframe(from_schema)
    to_time_zone = ZoneInfo("US/Central")
    run_conversion(iter_engines, df, from_schema, to_time_zone)


def test_time_conversion_by_geography(iter_engines: Engine) -> None:
    from_schema = get_datetime_schema(
        2018,
        ZoneInfo("US/Mountain"),
        TimeIntervalType.PERIOD_BEGINNING,
        "base_table",
        has_tz_col=True,
    )
    df = generate_dataframe_with_tz_col(from_schema)
    run_conversion_by_geography(iter_engines, df, from_schema)
