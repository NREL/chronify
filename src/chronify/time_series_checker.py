from sqlalchemy import Connection, Engine, MetaData, Table, select, text

from chronify.exceptions import InvalidTable
from chronify.models import TableSchema
from chronify.sqlalchemy.functions import read_database
from chronify.utils.sql import make_temp_view_name


class TimeSeriesChecker:
    """Performs checks on time series arrays in a table."""

    def __init__(self, engine: Engine, metadata: MetaData) -> None:
        self._engine = engine
        self._metadata = metadata

    def check_timestamps(self, schema: TableSchema) -> None:
        self._check_expected_timestamps(schema)
        self._check_expected_timestamps_by_time_array(schema)

    def _check_expected_timestamps(self, schema: TableSchema) -> None:
        expected = schema.time_config.list_timestamps()
        with self._engine.connect() as conn:
            table = Table(schema.name, self._metadata)
            time_columns = schema.time_config.list_time_columns()
            stmt = select(*(table.c[x] for x in time_columns)).distinct()
            for col in time_columns:
                stmt = stmt.where(table.c[col].is_not(None))
            df = read_database(stmt, conn, schema)
            actual = set(schema.time_config.list_timestamps_from_dataframe(df))
            match = sorted(actual) == expected
            # TODO: This check doesn't work and I'm not sure why.
            # diff = actual.symmetric_difference(expected)
            # if diff:
            #     msg = f"Actual timestamps do not match expected timestamps: {diff}"
            #     # TODO: list diff on each side.
            #     raise InvalidTable(msg)
            if not match:
                msg = "Actual timestamps do not match expected timestamps"
                # TODO: list diff on each side.
                raise InvalidTable(msg)

    def _check_expected_timestamps_by_time_array(self, schema: TableSchema) -> None:
        with self._engine.connect() as conn:
            tmp_name = make_temp_view_name()
            self._run_timestamp_checks_on_tmp_table(schema, conn, tmp_name)
            conn.execute(text(f"DROP TABLE IF EXISTS {tmp_name}"))

    @staticmethod
    def _run_timestamp_checks_on_tmp_table(
        schema: TableSchema, conn: Connection, table_name: str
    ) -> None:
        id_cols = ",".join(schema.time_array_id_columns)
        filters = [f"{x} IS NOT NULL" for x in schema.time_config.list_time_columns()]
        where_clause = "AND ".join(filters)
        query = f"""
            CREATE TEMP TABLE {table_name} AS
                SELECT
                    {id_cols}
                    ,COUNT(*) AS count_by_ta
                FROM {schema.name}
                WHERE {where_clause}
                GROUP BY {id_cols}
        """
        conn.execute(text(query))
        query2 = f"SELECT COUNT(DISTINCT count_by_ta) AS counts FROM {table_name}"
        result2 = conn.execute(text(query2)).fetchone()
        assert result2 is not None

        if result2[0] != 1:
            msg = "All time arrays must have the same length. There are {result2[0]} different lengths"
            raise InvalidTable(msg)

        query3 = f"SELECT DISTINCT count_by_ta AS counts FROM {table_name}"
        result3 = conn.execute(text(query3)).fetchone()
        assert result3 is not None
        actual_count = result3[0]
        if actual_count != schema.time_config.length:
            msg = (
                "Time arrays must have length={schema.time_config.length}. Actual = {actual_count}"
            )
            raise InvalidTable(msg)
