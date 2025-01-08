from util.mapper_test_helpers import get_datetime_schema, run_test, add_time_zone_data
from sqlalchemy import Engine
from zoneinfo import ZoneInfo
import pytest


@pytest.mark.parametrize("tzinfo", [ZoneInfo("US/Eastern"), None])
def test_index_time_hourly(iter_engines: Engine, one_year_index_time_by_hour, tzinfo):
    """simple test, values in one_year_index_time should be presereved in datetime mapping."""
    df, _, schema = one_year_index_time_by_hour
    df, schema = add_time_zone_data(df, schema)

    to_schema = get_datetime_schema(2019, tzinfo)
    run_test(iter_engines, df, schema, to_schema)


@pytest.mark.parametrize("tzinfo", [ZoneInfo("US/Pacific"), None])
def test_index_time_subhourly(iter_engines: Engine, one_year_index_time_subhourly, tzinfo):
    """Test aggrigation of index time to hourly datetime."""
    df, _, schema = one_year_index_time_subhourly
    df, schema = add_time_zone_data(df, schema)

    to_schema = get_datetime_schema(2019, tzinfo)
    run_test(iter_engines, df, schema, to_schema)


@pytest.mark.parametrize("tzinfo", [ZoneInfo("US/Central"), None])
def test_index_time_leap_year(iter_engines: Engine, one_year_index_time_by_hour_leapyear, tzinfo):
    """Test mapping in leap year, and then test dropping one day going to non leapyear."""
    df, _, schema = one_year_index_time_by_hour_leapyear
    df, schema = add_time_zone_data(df, schema)

    to_schema = get_datetime_schema(2020, tzinfo)
    run_test(iter_engines, df, schema, to_schema)

    # to_schema = get_datetime_schema(2019, None)
    # run_test(iter_engines, df, schema, to_schema)
