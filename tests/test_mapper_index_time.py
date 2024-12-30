from tests.util.mapper_test_helpers import get_datetime_schema, run_test
from sqlalchemy import Engine

def test_index_time_hourly(iter_engines: Engine, one_year_index_time_by_hour):
    """ simple test, values in one_year_index_time should be presereved in datetime mapping. """
    df, _, schema = one_year_index_time_by_hour
    to_schema = get_datetime_schema(2019, None)
    run_test(iter_engines, df, schema, to_schema)


def test_index_time_subhourly(iter_engines: Engine, one_year_index_time_subhourly):
    """ Test aggrigation of index time to hourly datetime. """
    df, _, schema = one_year_index_time_subhourly
    to_schema = get_datetime_schema(2019, None)
    run_test(iter_engines, df, schema, to_schema)


def test_index_time_leap_year(iter_engines: Engine, one_year_index_time_by_hour_leapyear):
    """ Test mapping in leap year, and then test dropping one day going to non leapyear. """
    df, _, schema = one_year_index_time_by_hour_leapyear
    to_schema = get_datetime_schema(2020, None)
    run_test(iter_engines, df, schema, to_schema)

    # to_schema = get_datetime_schema(2019, None)
    # run_test(iter_engines, df, schema, to_schema)
