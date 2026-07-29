"""DuckDB type constants used across the codebase."""

import duckdb
from _duckdb._sqltypes import DuckDBPyType

BIGINT = duckdb.sqltype("BIGINT")
BOOLEAN = duckdb.sqltype("BOOLEAN")
DOUBLE = duckdb.sqltype("DOUBLE")
FLOAT = duckdb.sqltype("FLOAT")
INTEGER = duckdb.sqltype("INTEGER")
TINYINT = duckdb.sqltype("TINYINT")
VARCHAR = duckdb.sqltype("VARCHAR")
TIMESTAMP = duckdb.sqltype("TIMESTAMP")
TIMESTAMP_TZ = duckdb.sqltype("TIMESTAMP WITH TIME ZONE")
TIMESTAMP_MS = duckdb.sqltype("TIMESTAMP_MS")
TIMESTAMP_NS = duckdb.sqltype("TIMESTAMP_NS")
TIMESTAMP_S = duckdb.sqltype("TIMESTAMP_S")

__all__ = [
    "DuckDBPyType",
    "BIGINT",
    "BOOLEAN",
    "DOUBLE",
    "FLOAT",
    "INTEGER",
    "TINYINT",
    "VARCHAR",
    "TIMESTAMP",
    "TIMESTAMP_TZ",
    "TIMESTAMP_MS",
    "TIMESTAMP_NS",
    "TIMESTAMP_S",
]
