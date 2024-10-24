from pathlib import Path
from chronify.time import get_zone_info

import duckdb
from duckdb import DuckDBPyRelation

from chronify.models import CsvTableSchema, get_duckdb_type_from_sqlalchemy
from chronify.time_configs import DatetimeRange


def read_csv(path: Path | str, schema: CsvTableSchema, **kwargs) -> DuckDBPyRelation:
    """Read a CSV file into a DuckDB relation."""
    if schema.column_dtypes:
        dtypes = {x.name: get_duckdb_type_from_sqlalchemy(x.dtype) for x in schema.column_dtypes}
        rel = duckdb.read_csv(str(path), dtype=dtypes, **kwargs)
    else:
        rel = duckdb.read_csv(str(path), **kwargs)

    exprs = []
    for column, dtype in zip(rel.columns, rel.types):
        if dtype is duckdb.typing.TIMESTAMP:
            if isinstance(schema.time_config, DatetimeRange):
                if schema.time_config.time_zone is None:
                    msg = "time_zone cannot be None if the time zone is not part of the timestamp string"
                    raise ValueError(msg)
                zone_info = get_zone_info(schema.time_config.time_zone)
            else:
                msg = f"need to add support for {type(schema.time_config)}"
                raise NotImplementedError(msg)
            expr = f"timezone('UTC', timezone({zone_info.key}, {column})) AS {column}"
        elif dtype is duckdb.typing.TIMESTAMP_TZ:
            msg = "no handling for timestamp with time zone yet"
            raise NotImplementedError(msg)
            # expr = f"timezone('UTC', {column}) AS {column}"
        else:
            expr = column
        exprs.append(expr)
    expr_str = ",".join(exprs)
    return duckdb.sql(f"SELECT {expr_str} FROM rel")
