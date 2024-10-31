import pandas as pd
import polars as pl
from numpy.dtypes import ObjectDType
from pandas import DatetimeTZDtype
from sqlalchemy import Connection, Selectable

from chronify.models import TableSchema
from chronify.time_configs import DatetimeRange


def read_database(query: Selectable | str, conn: Connection, schema: TableSchema) -> pd.DataFrame:
    """Read a database query into a Pandas DataFrame."""
    df = pl.read_database(query, connection=conn).to_pandas()
    config = schema.time_config
    if config.needs_utc_conversion(conn.engine.name):
        assert isinstance(config, DatetimeRange)
        if isinstance(df[config.time_column].dtype, ObjectDType):
            df[config.time_column] = pd.to_datetime(df[config.time_column], utc=True)
        else:
            df[config.time_column] = df[config.time_column].dt.tz_localize("UTC")
    return df


def write_database(df: pd.DataFrame, conn: Connection, schema: TableSchema) -> None:
    """Write a Pandas DataFrame to a database."""
    config = schema.time_config
    if config.needs_utc_conversion(conn.engine.name):
        assert isinstance(config, DatetimeRange)
        if isinstance(df.timestamp.dtype, DatetimeTZDtype):
            df[config.time_column] = df[config.time_column].dt.tz_convert("UTC")
        else:
            df[config.time_column] = df[config.time_column].dt.tz_localize("UTC")
    pl.DataFrame(df).write_database(schema.name, connection=conn, if_table_exists="append")
