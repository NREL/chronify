from sqlalchemy import Connection, Table, select, text

import pandas as pd

from chronify.exceptions import InvalidTable
from chronify.models import TableSchema
from chronify.sqlalchemy.functions import read_database
from chronify.time_range_generator_factory import make_time_range_generator


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
        count = self._check_expected_timestamps()
        self._check_null_consistency()
        self._check_expected_timestamps_by_time_array(count)

    def _check_expected_timestamps(self) -> int:
        expected = self._time_generator.list_timestamps()
        time_columns = self._time_generator.list_time_columns()
        stmt = select(*(self._table.c[x] for x in time_columns)).distinct()
        for col in time_columns:
            stmt = stmt.where(self._table.c[col].is_not(None))
        df = read_database(stmt, self._conn, self._schema.time_config)
        actual = self._time_generator.list_distinct_timestamps_from_dataframe(df)
        check_timestamp_lists(actual, expected)
        return len(expected)

    def _check_null_consistency(self) -> None:
        # If any time column has a NULL, all time columns must have a NULL.
        time_columns = self._time_generator.list_time_columns()
        if len(time_columns) == 1:
            return

        all_are_null = " AND ".join((f"{x} IS NULL" for x in time_columns))
        any_are_null = " OR ".join((f"{x} IS NULL" for x in time_columns))
        query_all = f"SELECT COUNT(*) FROM {self._schema.name} WHERE {all_are_null}"
        query_any = f"SELECT COUNT(*) FROM {self._schema.name} WHERE {any_are_null}"
        res_all = self._conn.execute(text(query_all)).fetchone()
        assert res_all is not None
        res_any = self._conn.execute(text(query_any)).fetchone()
        assert res_any is not None
        if res_all[0] != res_any[0]:
            msg = (
                "If any time columns have a NULL value for a row, all time columns in that "
                "row must be NULL. "
                f"Row count where all time values are NULL: {res_all[0]}. "
                f"Row count where any time values are NULL: {res_any[0]}. "
            )
            raise InvalidTable(msg)

    def _check_expected_timestamps_by_time_array(self, count: int) -> None:
        id_cols = ",".join(self._schema.time_array_id_columns)
        time_cols = ",".join(self._schema.time_config.list_time_columns())
        # NULL consistency was checked above.
        where_clause = f"{self._time_generator.list_time_columns()[0]} IS NOT NULL"
        on_expr = " AND ".join([f"t1.{x} = t2.{x}" for x in self._schema.time_array_id_columns])
        t1_id_cols = ",".join((f"t1.{x}" for x in self._schema.time_array_id_columns))

        query = f"""
            WITH distinct_time_values_by_array AS (
                SELECT DISTINCT {time_cols}, {id_cols}
                FROM {self._schema.name}
                WHERE {where_clause}
            ),
            t1 AS (
                SELECT {id_cols}, COUNT(*) AS distinct_count_by_ta
                FROM distinct_time_values_by_array
                GROUP BY {id_cols}
            ),
            t2 AS (
                SELECT {id_cols}, COUNT(*) AS count_by_ta
                FROM {self._schema.name}
                WHERE {where_clause}
                GROUP BY {id_cols}
            )
            SELECT
                t1.distinct_count_by_ta
                ,t2.count_by_ta
                ,{t1_id_cols}
            FROM t1
            JOIN t2
            ON {on_expr}
        """
        for result in self._conn.execute(text(query)).fetchall():
            distinct_count_by_ta = result[0]
            count_by_ta = result[1]
            if not count_by_ta == count == distinct_count_by_ta:
                id_vals = result[2:]
                values = ", ".join(
                    f"{x}={y}" for x, y in zip(self._schema.time_array_id_columns, id_vals)
                )
                msg = (
                    f"The count of time values in each time array must be {count}, and each "
                    "value must be distinct. "
                    f"Time array identifiers: {values}."
                    f"count = {count_by_ta}, distinct count = {distinct_count_by_ta}. "
                )
                raise InvalidTable(msg)


def check_timestamp_lists(actual: list[pd.Timestamp], expected: list[pd.Timestamp]) -> None:
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
