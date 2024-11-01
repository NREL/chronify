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


def join(
    left_rel: DuckDBPyRelation,
    right_rel: DuckDBPyRelation,
    on: list[str],
    how: str = "inner",
) -> DuckDBPyRelation:
    def get_join_statement(left_df, right_df, keys: list):
        stmts = [f"{left_df.alias}.{key}={right_df.alias}.{key}" for key in keys]
        return " and ".join(stmts)

    def get_select_after_join_statement(left_df, right_df, keys: list):
        left_cols = [f"{left_df.alias}.{x}" for x in left_df.columns]
        right_cols = [x for x in right_df.columns if x not in keys]
        return ", ".join(left_cols + right_cols)

    join_stmt = get_join_statement(left_rel, right_rel, on)
    select_stmt = get_select_after_join_statement(left_rel, right_rel, on)
    query = f"SELECT {select_stmt} from {left_rel.alias} {how.upper()} JOIN {right_rel.alias} ON {join_stmt}"
    breakpoint()
    # return left_rel.join(right_rel, join_stmt).select(select_stmt)
    return duckdb.sql(query)
