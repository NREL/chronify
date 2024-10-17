import pandas as pd
import polars as pl
from sqlalchemy import Connection, Selectable


def read_database_query(query: Selectable | str, conn: Connection) -> pd.DataFrame:
    """Read a database query into a Pandas DataFrame."""
    return pl.read_database(query, connection=conn).to_pandas()
