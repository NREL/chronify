from zoneinfo import ZoneInfo
import pytest
from datetime import datetime, timedelta
from typing import Any
import numpy as np

import pandas as pd
from sqlalchemy import Engine, MetaData

from chronify.sqlalchemy.functions import read_database, write_database
from chronify.time_series_mapper import map_time
from chronify.time_configs import DatetimeRange
from chronify.models import TableSchema
from chronify.time import TimeIntervalType, MeasurementType
from chronify.exceptions import ConflictingInputsError, InvalidParameter
from chronify.datetime_range_generator import DatetimeRangeGenerator
from chronify.time_utils import shift_time_interval, roll_time_interval, wrap_timestamps


def generate_datetime_data(time_config: DatetimeRange) -> pd.Series:
    return pd.to_datetime(list(DatetimeRangeGenerator(time_config).iter_timestamps()))


def generate_datetime_dataframe(schema: TableSchema) -> pd.DataFrame:
    df = pd.DataFrame({schema.time_config.time_column: generate_datetime_data(schema.time_config)})

    for i, x in enumerate(schema.time_array_id_columns):
        df[x] = i
    df[schema.value_column] = np.random.rand(len(df))
    return df


def get_datetime_schema(
    year: int, tzinfo: ZoneInfo | None, interval_type: TimeIntervalType, name: str
) -> TableSchema:
    start = datetime(year=year, month=1, day=1, tzinfo=tzinfo)
    end = datetime(year=year + 1, month=1, day=1, tzinfo=tzinfo)
    resolution = timedelta(hours=1)
    length = (end - start) / resolution + 1
    schema = TableSchema(
        name=name,
        time_config=DatetimeRange(
            start=start,
            resolution=resolution,
            length=length,
            interval_type=interval_type,
            time_column="timestamp",
        ),
        time_array_id_columns=["id"],
        value_column="value",
    )
    return schema


def check_dataframes(
    dfi: pd.DataFrame, dfo: pd.DataFrame, from_schema: TableSchema, to_schema: TableSchema
) -> None:
    assert (
        dfo[to_schema.time_config.time_column] == dfi[from_schema.time_config.time_column]
    ).all()
    match from_schema.time_config.interval_type, to_schema.time_config.interval_type:
        case TimeIntervalType.PERIOD_BEGINNING, TimeIntervalType.PERIOD_ENDING:
            shift = 1
        case TimeIntervalType.PERIOD_ENDING, TimeIntervalType.PERIOD_BEGINNING:
            shift = -1
    assert (np.array(dfo["value"]) == np.roll(dfi["value"], shift)).all()


def run_test(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
    to_schema: TableSchema,
    error: tuple[Any, str],
) -> None:
    # Ingest
    metadata = MetaData()
    with engine.connect() as conn:
        write_database(
            df, conn, from_schema.name, [from_schema.time_config], if_table_exists="replace"
        )
        conn.commit()
    metadata.reflect(engine, views=True)

    # Map
    if error:
        with pytest.raises(error[0], match=error[1]):
            map_time(engine, metadata, from_schema, to_schema)
    else:
        map_time(engine, metadata, from_schema, to_schema)

        # Check mapped table
        with engine.connect() as conn:
            query = f"select * from {to_schema.name}"
            queried = read_database(query, conn, to_schema.time_config)
        queried = queried.sort_values(by=["id", "timestamp"]).reset_index(drop=True)[df.columns]
        assert not queried.equals(df)
        check_dataframes(df, queried, from_schema, to_schema)


def test_roll_time_using_shift_and_wrap(iter_engines: Engine) -> None:
    from_schema = get_datetime_schema(2024, None, TimeIntervalType.PERIOD_ENDING, "from_table")
    data = generate_datetime_data(from_schema.time_config)
    df = generate_datetime_dataframe(from_schema)
    to_schema = get_datetime_schema(2024, None, TimeIntervalType.PERIOD_BEGINNING, "to_table")

    df["rolled"] = roll_time_interval(
        df[from_schema.time_config.time_column],
        from_schema.time_config.interval_type,
        to_schema.time_config.interval_type,
    )
    df["rolled2"] = shift_time_interval(
        df[from_schema.time_config.time_column],
        from_schema.time_config.interval_type,
        to_schema.time_config.interval_type,
    )
    df["rolled2"] = wrap_timestamps(
        df["rolled2"],
        data,
    )
    assert df["rolled"].equals(df["rolled2"])
    assert set(data) == set(df["rolled"].tolist())


@pytest.mark.parametrize("tzinfo", [ZoneInfo("US/Eastern"), None])
def test_time_interval_shift(
    iter_engines: Engine,
    tzinfo: ZoneInfo | None,
) -> None:
    from_schema = get_datetime_schema(
        2020, tzinfo, TimeIntervalType.PERIOD_BEGINNING, "from_table"
    )
    df = generate_datetime_dataframe(from_schema)
    to_schema = get_datetime_schema(2020, tzinfo, TimeIntervalType.PERIOD_ENDING, "to_table")

    error = ()
    run_test(iter_engines, df, from_schema, to_schema, error)


def test_instantaneous_interval_type(
    iter_engines: Engine,
) -> None:
    from_schema = get_datetime_schema(2020, None, TimeIntervalType.INSTANTANEOUS, "from_table")
    df = generate_datetime_dataframe(from_schema)
    to_schema = get_datetime_schema(2020, None, TimeIntervalType.PERIOD_ENDING, "to_table")
    error = (InvalidParameter, "Cannot handle")
    run_test(iter_engines, df, from_schema, to_schema, error)


def test_schema_compatibility(
    iter_engines: Engine,
) -> None:
    from_schema = get_datetime_schema(2020, None, TimeIntervalType.PERIOD_BEGINNING, "from_table")
    df = generate_datetime_dataframe(from_schema)
    to_schema = get_datetime_schema(2020, None, TimeIntervalType.PERIOD_ENDING, "to_table")
    to_schema.time_array_id_columns += ["extra_column"]
    error = (ConflictingInputsError, ".* cannot produce the columns")
    run_test(iter_engines, df, from_schema, to_schema, error)


def test_measurement_type_consistency(
    iter_engines: Engine,
) -> None:
    from_schema = get_datetime_schema(2020, None, TimeIntervalType.PERIOD_BEGINNING, "from_table")
    df = generate_datetime_dataframe(from_schema)
    to_schema = get_datetime_schema(2020, None, TimeIntervalType.PERIOD_ENDING, "to_table")
    to_schema.time_config.measurement_type = MeasurementType.MAX
    error = (ConflictingInputsError, "Inconsistent measurement_types")
    run_test(iter_engines, df, from_schema, to_schema, error)
