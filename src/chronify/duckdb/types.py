"""Compatibility layer for duckdb type constants across versions.

duckdb < 1.2 used duckdb.typing.X constants.
duckdb >= 1.2 removed duckdb.typing; use duckdb.sqltype() instead.
"""

import duckdb

try:
    from duckdb.typing import DuckDBPyType

    BIGINT = duckdb.typing.BIGINT
    BOOLEAN = duckdb.typing.BOOLEAN
    DOUBLE = duckdb.typing.DOUBLE
    FLOAT = duckdb.typing.FLOAT
    INTEGER = duckdb.typing.INTEGER
    TINYINT = duckdb.typing.TINYINT
    VARCHAR = duckdb.typing.VARCHAR
    TIMESTAMP = duckdb.typing.TIMESTAMP
    TIMESTAMP_TZ = duckdb.typing.TIMESTAMP_TZ
    TIMESTAMP_MS = duckdb.typing.TIMESTAMP_MS
    TIMESTAMP_NS = duckdb.typing.TIMESTAMP_NS
    TIMESTAMP_S = duckdb.typing.TIMESTAMP_S
except (ImportError, AttributeError):
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
