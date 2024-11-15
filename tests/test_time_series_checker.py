from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from sqlalchemy import (
    Engine,
    MetaData,
    Table,
)
from chronify.exceptions import InvalidTable
from chronify.models import TableSchema
from chronify.sqlalchemy.functions import write_database
from chronify.time import TimeIntervalType
from chronify.time_configs import DatetimeRange
from chronify.time_series_checker import check_timestamps


def test_valid_datetimes_with_tz(iter_engines: Engine):
    """Valid timestamps with time zones."""
    _run_test(iter_engines, *_get_inputs_for_valid_datetimes_with_tz())


def test_valid_datetimes_without_tz(iter_engines: Engine):
    """Valid timestamps without time zones."""
    _run_test(iter_engines, *_get_inputs_for_valid_datetimes_without_tz())


def test_invalid_datetimes(iter_engines: Engine):
    """Timestamps do not match the schema."""
    _run_test(iter_engines, *_get_inputs_for_incorrect_datetimes())


def test_mismatched_time_array_lengths(iter_engines: Engine):
    """Some time arrays have different lengths."""
    _run_test(iter_engines, *_get_inputs_for_mismatched_time_array_lengths())


def test_incorrect_lengths(iter_engines: Engine):
    """All time arrays are consistent but have the wrong length."""
    _run_test(iter_engines, *_get_inputs_for_incorrect_lengths())


def _run_test(
    engine: Engine,
    df: pd.DataFrame,
    tzinfo: Optional[ZoneInfo],
    length: int,
    message: Optional[str],
) -> None:
    metadata = MetaData()
    schema = TableSchema(
        name="generators",
        time_config=DatetimeRange(
            start=datetime(year=2020, month=1, day=1, tzinfo=tzinfo),
            resolution=timedelta(hours=1),
            length=length,
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
            time_column="timestamp",
        ),
        time_array_id_columns=["generator"],
        value_column="value",
    )
    with engine.connect() as conn:
        write_database(df, conn, schema)
        conn.commit()
    metadata.reflect(engine)

    with engine.connect() as conn:
        table = Table(schema.name, metadata)
        if message is None:
            check_timestamps(conn, table, schema)
        else:
            with pytest.raises(InvalidTable, match=message):
                check_timestamps(conn, table, schema)


def _get_inputs_for_valid_datetimes_with_tz() -> tuple[pd.DataFrame, ZoneInfo, int, None]:
    tzinfo = ZoneInfo("EST")
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2020, 1, 1, 0, tzinfo=tzinfo),
                datetime(2020, 1, 1, 1, tzinfo=tzinfo),
                datetime(2020, 1, 1, 2, tzinfo=tzinfo),
            ],
            "generator": ["gen1", "gen1", "gen1"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    return df, tzinfo, len(df), None


def _get_inputs_for_valid_datetimes_without_tz() -> tuple[pd.DataFrame, None, int, None]:
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2020, 1, 1, 0),
                datetime(2020, 1, 1, 1),
                datetime(2020, 1, 1, 2),
            ],
            "generator": ["gen1", "gen1", "gen1"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    return df, None, len(df), None


def _get_inputs_for_incorrect_datetimes() -> tuple[pd.DataFrame, ZoneInfo, int, str]:
    data_tzinfo = ZoneInfo("America/New_York")
    schema_tzinfo = ZoneInfo("MST")
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2020, 1, 1, 0, tzinfo=data_tzinfo),
                datetime(2020, 1, 1, 1, tzinfo=data_tzinfo),
                datetime(2020, 1, 1, 2, tzinfo=data_tzinfo),
                datetime(2020, 1, 1, 3, tzinfo=data_tzinfo),
                datetime(2020, 1, 1, 4, tzinfo=data_tzinfo),
            ],
            "generator": ["gen1", "gen1", "gen1", "gen1", "gen1"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    return (
        df,
        schema_tzinfo,
        len(df),
        "Actual timestamps do not match expected timestamps",
    )


def _get_inputs_for_mismatched_time_array_lengths() -> tuple[pd.DataFrame, ZoneInfo, int, str]:
    tzinfo = ZoneInfo("EST")
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2020, 1, 1, 0, tzinfo=tzinfo),
                datetime(2020, 1, 1, 1, tzinfo=tzinfo),
                datetime(2020, 1, 1, 0, tzinfo=tzinfo),
                datetime(2020, 1, 1, 1, tzinfo=tzinfo),
                # This one is duplicate.
                datetime(2020, 1, 1, 1, tzinfo=tzinfo),
            ],
            "generator": ["gen1", "gen1", "gen2", "gen2", "gen2"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    return df, tzinfo, 2, "All time arrays must have the same length."


def _get_inputs_for_incorrect_lengths() -> tuple[pd.DataFrame, ZoneInfo, int, str]:
    tzinfo = ZoneInfo("EST")
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2020, 1, 1, 0, tzinfo=tzinfo),
                datetime(2020, 1, 1, 1, tzinfo=tzinfo),
                # This one is duplicate.
                datetime(2020, 1, 1, 1, tzinfo=tzinfo),
                datetime(2020, 1, 1, 0, tzinfo=tzinfo),
                datetime(2020, 1, 1, 1, tzinfo=tzinfo),
                # This one is duplicate.
                datetime(2020, 1, 1, 1, tzinfo=tzinfo),
            ],
            "generator": ["gen1", "gen1", "gen1", "gen2", "gen2", "gen2"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    return df, tzinfo, 2, "Time arrays must have length="
