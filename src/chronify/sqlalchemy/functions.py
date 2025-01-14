"""This file provides functions to read and write the database as efficiently as possible.
The default behavior of sqlalchemy is to convert data into rows of tuples in Python, which
is very slow. This code attempts to bypass Python as much as possible through Arrow tables
in memory.
"""

from typing import Any, Literal, TypeAlias, Sequence
from collections import Counter

import pandas as pd
import polars as pl
from numpy.dtypes import ObjectDType
from pandas import DatetimeTZDtype
from sqlalchemy import Connection, Selectable

from chronify.exceptions import InvalidOperation, InvalidParameter
from chronify.time_configs import DatetimeRange, TimeBaseModel

# Copied from Polars, which doesn't export the type.
DbWriteMode: TypeAlias = Literal["replace", "append", "fail"]


def read_database(
    query: Selectable | str, conn: Connection, config: TimeBaseModel, params: Any = None
) -> pd.DataFrame:
    """Read a database query into a Pandas DataFrame."""
    if conn.engine.name == "duckdb":
        if isinstance(query, str):
            df = conn._dbapi_connection.driver_connection.sql(query, params=params).to_df()  # type: ignore
        else:
            df = conn.execute(query).cursor.fetch_df()  # type: ignore
    elif conn.engine.name == "sqlite":
        df = pd.read_sql(query, conn, params=params)
        if isinstance(config, DatetimeRange):
            _convert_database_output_for_datetime(df, config)
    else:
        if params is not None:
            msg = "Passing params to a Polars query is not supported yet."
            raise InvalidOperation(msg)
        df = pl.read_database(query, connection=conn).to_pandas()
    return df  # type: ignore


def write_database(
    df: pd.DataFrame,
    conn: Connection,
    table_name: str,
    configs: Sequence[TimeBaseModel],
    if_table_exists: DbWriteMode = "append",
) -> None:
    """Write a Pandas DataFrame to the database.
    configs allows sqlite formatting for more than one datetime columns.
    """
    match conn.engine.name:
        case "duckdb":
            assert conn._dbapi_connection is not None
            assert conn._dbapi_connection.driver_connection is not None
            match if_table_exists:
                case "append":
                    query = f"INSERT INTO {table_name} SELECT * FROM df"
                case "replace":
                    conn._dbapi_connection.driver_connection.sql(
                        f"DROP TABLE IF EXISTS {table_name}"
                    )
                    query = f"CREATE TABLE {table_name} AS SELECT * FROM df"
                case "fail":
                    query = f"CREATE TABLE {table_name} AS SELECT * FROM df"
                case _:
                    msg = f"{if_table_exists=}"
                    raise InvalidOperation(msg)
            conn._dbapi_connection.driver_connection.sql(query)
        case "sqlite":
            _check_one_config_per_datetime_column(configs)
            copied = False
            for config in configs:
                if isinstance(config, DatetimeRange):
                    df, copied = _convert_database_input_for_datetime(df, config, copied)
            pl.DataFrame(df).write_database(
                table_name, connection=conn, if_table_exists=if_table_exists
            )
        case _:
            pl.DataFrame(df).write_database(
                table_name, connection=conn, if_table_exists=if_table_exists
            )


def _check_one_config_per_datetime_column(configs: Sequence[TimeBaseModel]) -> None:
    time_col_count = Counter(
        [config.time_column for config in configs if isinstance(config, DatetimeRange)]
    )
    time_col_dup = {k: v for k, v in time_col_count.items() if v > 1}
    if len(time_col_dup) > 0:
        msg = f"More than one datetime config found for: {time_col_dup}"
        raise InvalidParameter(msg)


def _convert_database_input_for_datetime(
    df: pd.DataFrame, config: DatetimeRange, copied: bool
) -> tuple[pd.DataFrame, bool]:
    if config.is_time_zone_naive():
        return df, copied

    if copied:
        df2 = df
    else:
        df2 = df.copy()
        copied = True

    if isinstance(df2[config.time_column].dtype, DatetimeTZDtype):
        df2[config.time_column] = df2[config.time_column].dt.tz_convert("UTC")
    else:
        df2[config.time_column] = df2[config.time_column].dt.tz_localize("UTC")

    return df2, copied


def _convert_database_output_for_datetime(df: pd.DataFrame, config: DatetimeRange) -> None:
    if config.time_column in df.columns:
        if not config.is_time_zone_naive():
            if isinstance(df[config.time_column].dtype, ObjectDType):
                df[config.time_column] = pd.to_datetime(df[config.time_column], utc=True)
            else:
                df[config.time_column] = df[config.time_column].dt.tz_localize("UTC")
        else:
            if isinstance(df[config.time_column].dtype, ObjectDType):
                df[config.time_column] = pd.to_datetime(df[config.time_column], utc=False)
