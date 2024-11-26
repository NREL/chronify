from sqlalchemy import Connection, Table, select, text
import pandas as pd
from datetime import datetime

from chronify.exceptions import InvalidTable
from chronify.models import TableSchema
from chronify.sqlalchemy.functions import read_database
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time_configs import DatetimeRange
from chronify.utils.sql import make_temp_view_name


def check_timestamps(conn: Connection, table: Table, schema: TableSchema) -> None:
    """Performs checks on time series arrays in a table."""
    TimeSeriesChecker(conn, table, schema).check_timestamps()


class TimeSeriesChecker:
    """Performs checks on time series arrays in a table."""

    def __init__(self, conn: Connection, table: Table, schema: TableSchema) -> None:
        self._conn = conn
        self._schema = schema
        self._table = table
        self._time_generator = make_time_range_generator(schema.time_config)

    def check_timestamps(self) -> None:
        self._check_expected_timestamps_by_time_array()
        self._check_expected_timestamps()

    def _check_expected_timestamps(self) -> None:
        expected = self._time_generator.list_timestamps()
        time_columns = self._time_generator.list_time_columns()
        stmt = select(*(self._table.c[x] for x in time_columns)).distinct()
        for col in time_columns:
            stmt = stmt.where(self._table.c[col].is_not(None))
        df = read_database(stmt, self._conn, self._schema.time_config)
        actual = self._time_generator.list_distinct_timestamps_from_dataframe(df)

        if isinstance(self._schema.time_config, DatetimeRange):
            expected = [pd.Timestamp(x) for x in expected]
        compare_lists(actual, expected)

    def _check_expected_timestamps_by_time_array(self) -> None:
        tmp_name = make_temp_view_name()
        self._run_timestamp_checks_on_tmp_table(tmp_name)
        self._conn.execute(text(f"DROP TABLE IF EXISTS {tmp_name}"))

    def _run_timestamp_checks_on_tmp_table(self, table_name: str) -> None:
        id_cols = ",".join(self._schema.time_array_id_columns)
        filters = [f"{x} IS NOT NULL" for x in self._time_generator.list_time_columns()]
        where_clause = " AND ".join(filters)
        query = f"""
            CREATE TEMP TABLE {table_name} AS
                SELECT
                    {id_cols}
                    ,COUNT(*) AS count_by_ta
                FROM {self._schema.name}
                WHERE {where_clause}
                GROUP BY {id_cols}
        """
        self._conn.execute(text(query))
        query2 = f"SELECT COUNT(DISTINCT count_by_ta) AS counts FROM {table_name}"
        result2 = self._conn.execute(text(query2)).fetchone()
        assert result2 is not None

        if result2[0] != 1:
            msg = f"All time arrays must have the same length. There are {result2[0]} different lengths"
            raise InvalidTable(msg)

        query3 = f"SELECT DISTINCT count_by_ta AS counts FROM {table_name}"
        result3 = self._conn.execute(text(query3)).fetchone()
        assert result3 is not None
        actual_count = result3[0]
        expected_count = len(self._time_generator.list_timestamps())
        if actual_count != expected_count:
            msg = f"Time arrays must have length={expected_count}. Actual = {actual_count}"
            raise InvalidTable(msg)


def compare_lists(actual: list, expected: list) -> None:
    match = actual == expected
    if not match:
        if len(actual) != len(expected):
            msg1 = f"Mismatch number of timestamps: actual: {len(actual)} vs. expected: {len(expected)}"
            raise InvalidTable(msg1)
        missing = [x for x in expected if x not in actual]
        extra = [x for x in actual if x not in expected]
        msg2 = "Actual timestamps do not match expected timestamps. \n"
        msg2 += f"Missing: {missing} \n"
        msg2 += f"Extra: {extra}"
        raise InvalidTable(msg2)


def compare_tz_aware_datetime_lists(
    actual: list[datetime] | list[pd.Timestamp], expected: list[datetime] | list[pd.Timestamp]
) -> None:
    """Convert tz-aware timestamps to posix before comparing."""
    expected_t = [x.timestamp() for x in expected]
    actual_t = [x.timestamp() for x in actual]
    match = actual_t == expected_t
    if not match:
        if len(actual) != len(expected):
            msg1 = f"Mismatch number of timestamps: actual: {len(actual)} vs. expected: {len(expected)}"
            raise InvalidTable(msg1)
        missing = [expected[expected_t.index(x)] for x in expected_t if x not in actual_t]
        extra = [actual[actual_t.index(x)] for x in actual_t if x not in expected_t]
        msg2 = "Actual timestamps do not match expected timestamps. \n"
        msg2 += f"Missing: {missing} \n"
        msg2 += f"Extra: {extra}"
        raise InvalidTable(msg2)
