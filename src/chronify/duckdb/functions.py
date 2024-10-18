from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
from duckdb import DuckDBPyRelation


def add_datetime_column(
    rel: DuckDBPyRelation,
    start: datetime,
    resolution: timedelta,
    length: int,
    time_array_id_columns: Iterable[str],
    time_column: str,
    timestamps: list[datetime],
) -> DuckDBPyRelation:
    """Add a datetime column to the relation."""
    # TODO
    raise NotImplementedError
    # values = []
    # columns = ",".join(rel.columns)
    # return duckdb.sql(
    #    f"""
    #    SELECT
    #        AS {time_column}
    #        ,{columns}
    #    FROM rel
    #    """
    # )


def make_write_parquet_query(table_or_view: str, file_path: Path | str) -> str:
    """Make an SQL string that can be used to write a Parquet file from a table or view."""
    # TODO: Hive partitioning?
    return f"""
        COPY
            (SELECT * FROM {table_or_view})
            TO '{file_path}'
            (FORMAT 'parquet');
        """


def unpivot(
    rel: DuckDBPyRelation,
    pivoted_columns: Iterable[str],
    name_column: str,
    value_column: str,
) -> DuckDBPyRelation:
    pivoted_str = ",".join(pivoted_columns)

    query = f"""
        SELECT * FROM rel
        UNPIVOT INCLUDE NULLS (
            {value_column}
            FOR {name_column} in ({pivoted_str})
        )
        """
    return duckdb.sql(query)
