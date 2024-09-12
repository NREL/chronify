from sqlalchemy import Connection, Engine, text

from chronify import TableSchema
from chronify.exceptions import InvalidTable


class TimeSeriesChecker:
    """Performs checks on time series arrays in a table."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def check_timestamps(self, schema: TableSchema) -> None:
        self._check_expected_timestamps(schema)
        self._check_expected_timestamps_by_time_array(schema)

    def _check_expected_timestamps(self, schema: TableSchema) -> None:
        expected = schema.time_config.list_timestamps()
        with self._engine.connect() as conn:
            filters = []
            for col in schema.time_config.time_columns:
                filters.append(f"{col} IS NOT NULL")
            time_cols = ",".join(schema.time_config.time_columns)
            where_clause = "AND ".join(filters)
            query = f"SELECT DISTINCT {time_cols} FROM {schema.name} WHERE {where_clause}"
            actual = set(schema.time_config.convert_database_timestamps(conn.execute(text(query))))
            diff = actual.symmetric_difference(expected)
            if diff:
                msg = f"Actual timestamps do not match expected timestamps: {diff}"
                # TODO: list diff on each side.
                raise InvalidTable(msg)

    def _check_expected_timestamps_by_time_array(self, schema: TableSchema) -> None:
        with self._engine.connect() as conn:
            tmp_name = "tss_tmp_table"
            self._run_timestamp_checks_on_tmp_table(schema, conn, tmp_name)
            conn.execute(text(f"DROP TABLE IF EXISTS {tmp_name}"))

    @staticmethod
    def _run_timestamp_checks_on_tmp_table(schema: TableSchema, conn: Connection, table_name: str):
        id_cols = ",".join(schema.time_array_id_columns)
        filters = [f"{x} IS NOT NULL" for x in schema.time_config.time_columns]
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
        result2 = conn.execute(text(query2)).all()

        if len(result2) != 1:
            msg = "All time arrays must have the same length. There are {len(result)} different lengths"
            raise InvalidTable(msg)

        query3 = f"SELECT DISTINCT count_by_ta AS counts FROM {table_name}"
        result3 = conn.execute(text(query3)).all()
        actual_count = result3[0][0]
        if actual_count != schema.time_config.length:
            msg = "Time arrays must have length={schema.time_config.length}. Actual = {actual_count}"
            raise InvalidTable(msg)
