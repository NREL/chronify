from typing import Literal, TypeAlias
import pandas as pd
import polars as pl
from numpy.dtypes import ObjectDType
from pandas import DatetimeTZDtype
from sqlalchemy import Connection, Selectable

from chronify.time_configs import DatetimeRange, TimeBaseModel

# Copied from Polars, which doesn't export the type.
DbWriteMode: TypeAlias = Literal["replace", "append", "fail"]


def read_database(
    query: Selectable | str, conn: Connection, config: TimeBaseModel
) -> pd.DataFrame:
    """Read a database query into a Pandas DataFrame."""
    df = pl.read_database(query, connection=conn).to_pandas()
    if isinstance(config, DatetimeRange):
        df = _convert_database_output_for_datetime(df, conn, config)
    return df


def write_database(
    df: pd.DataFrame,
    conn: Connection,
    table_name: str,
    config: TimeBaseModel,
    if_table_exists: DbWriteMode = "append",
) -> None:
    """Write a Pandas DataFrame to the database.
    For efficiency, this may mutate the input dataframe for datetime inputs. The expectation
    is that the dataframe is already a copy of user inputs. This could be changed if needed.
    """
    if isinstance(config, DatetimeRange):
        df = _convert_database_input_for_datetime(df, conn, config)
    pl.DataFrame(df).write_database(table_name, connection=conn, if_table_exists=if_table_exists)


def _convert_database_input_for_datetime(
    df: pd.DataFrame, conn: Connection, config: DatetimeRange
) -> pd.DataFrame:
    if (
        conn.engine.name == "sqlite"
        and isinstance(config, DatetimeRange)
        and not config.is_time_zone_naive()
    ):
        if isinstance(df[config.time_column].dtype, DatetimeTZDtype):
            df[config.time_column] = df[config.time_column].dt.tz_convert("UTC")
        else:
            df[config.time_column] = df[config.time_column].dt.tz_localize("UTC")
    return df


def _convert_database_output_for_datetime(
    df: pd.DataFrame, conn: Connection, config: DatetimeRange
) -> pd.DataFrame:
    if conn.engine.name == "sqlite" and isinstance(config, DatetimeRange):
        if not config.is_time_zone_naive():
            if isinstance(df[config.time_column].dtype, ObjectDType):
                df[config.time_column] = pd.to_datetime(df[config.time_column], utc=True)
            else:
                df[config.time_column] = df[config.time_column].dt.tz_localize("UTC")
        else:
            if isinstance(df[config.time_column].dtype, ObjectDType):
                df[config.time_column] = pd.to_datetime(df[config.time_column], utc=False)
    return df
