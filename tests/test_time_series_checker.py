from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from sqlalchemy import MetaData, create_engine
from chronify.exceptions import InvalidTable
from chronify.models import TableSchema
from chronify.time import TimeIntervalType, TimeZone
from chronify.time_configs import DatetimeRange
from chronify.time_series_checker import TimeSeriesChecker


def test_invalid_datetimes():
    """Test the case where there are missing timestamps."""
    _run_test(*_get_inputs_for_invalid_datetimes())


def test_mismatched_time_array_lengths():
    """Test the case where some time arrays have different lengths."""
    _run_test(*_get_inputs_for_mismatched_time_array_lengths())


def test_incorrect_lengths():
    """Test the case where all time arrays are consistent but have the wrong length."""
    _run_test(*_get_inputs_for_incorrect_lengths())


def _run_test(df: pd.DataFrame, length: int, message: str) -> None:
    engine = create_engine("duckdb:///:memory:")
    metadata = MetaData()
    with engine.connect() as conn:
        df.to_sql("generators", conn)
        conn.commit()
    metadata.reflect(engine)

    schema = TableSchema(
        name="generators",
        time_config=DatetimeRange(
            start=datetime(year=2020, month=1, day=1),
            resolution=timedelta(hours=1),
            length=length,
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
            time_columns=["timestamp"],
            time_zone=TimeZone.UTC,
        ),
        time_array_id_columns=["generator"],
        value_column="value",
    )
    checker = TimeSeriesChecker(engine, metadata)
    with pytest.raises(InvalidTable, match=message):
        checker.check_timestamps(schema)


def _get_inputs_for_invalid_datetimes():
    tzinfo = ZoneInfo("UTC")
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2020, 1, 1, 0).replace(tzinfo=tzinfo),
                datetime(2020, 1, 1, 1).replace(tzinfo=tzinfo),
                datetime(2020, 1, 1, 2).replace(tzinfo=tzinfo),
                datetime(2020, 1, 1, 3).replace(tzinfo=tzinfo),
                datetime(2020, 1, 1, 4).replace(tzinfo=tzinfo),
            ],
            "generator": ["gen1", "gen1", "gen1", "gen1", "gen1"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    return df, 10, "Actual timestamps do not match expected timestamps"


def _get_inputs_for_mismatched_time_array_lengths():
    tzinfo = ZoneInfo("UTC")
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2020, 1, 1, 0).replace(tzinfo=tzinfo),
                datetime(2020, 1, 1, 1).replace(tzinfo=tzinfo),
                datetime(2020, 1, 1, 0).replace(tzinfo=tzinfo),
                datetime(2020, 1, 1, 1).replace(tzinfo=tzinfo),
                # This one is duplicate.
                datetime(2020, 1, 1, 1).replace(tzinfo=tzinfo),
            ],
            "generator": ["gen1", "gen1", "gen2", "gen2", "gen2"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    return df, 2, "All time arrays must have the same length."


def _get_inputs_for_incorrect_lengths():
    tzinfo = ZoneInfo("UTC")
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2020, 1, 1, 0).replace(tzinfo=tzinfo),
                datetime(2020, 1, 1, 1).replace(tzinfo=tzinfo),
                # This one is duplicate.
                datetime(2020, 1, 1, 1).replace(tzinfo=tzinfo),
                datetime(2020, 1, 1, 0).replace(tzinfo=tzinfo),
                datetime(2020, 1, 1, 1).replace(tzinfo=tzinfo),
                # This one is duplicate.
                datetime(2020, 1, 1, 1).replace(tzinfo=tzinfo),
            ],
            "generator": ["gen1", "gen1", "gen1", "gen2", "gen2", "gen2"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    return df, 2, "Time arrays must have length="
