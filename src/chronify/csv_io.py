from pathlib import Path
from typing import Any

import duckdb
from duckdb import DuckDBPyRelation

from chronify.models import CsvTableSchema, get_duckdb_type_from_sqlalchemy


def read_csv(path: Path | str, schema: CsvTableSchema, **kwargs: Any) -> DuckDBPyRelation:
    """Read a CSV file into a DuckDB relation."""
    if schema.column_dtypes:
        dtypes = {x.name: get_duckdb_type_from_sqlalchemy(x.dtype) for x in schema.column_dtypes}
        rel = duckdb.read_csv(str(path), dtype=dtypes, **kwargs)
    else:
        rel = duckdb.read_csv(str(path), **kwargs)

    expr = ",".join(rel.columns)
    return duckdb.sql(f"SELECT {expr} FROM rel")
