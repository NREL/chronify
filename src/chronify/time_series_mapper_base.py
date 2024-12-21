import abc
import logging
from functools import reduce
from operator import and_

import pandas as pd
from sqlalchemy import Engine, MetaData, Table, select, text

from chronify.sqlalchemy.functions import write_database
from chronify.models import TableSchema
from chronify.exceptions import TableAlreadyExists
from chronify.utils.sqlalchemy_table import create_table
from chronify.time_series_checker import check_timestamps
from chronify.time_configs import DatetimeRange

logger = logging.getLogger(__name__)


class TimeSeriesMapperBase(abc.ABC):
    """Maps time series data from one configuration to another."""


def apply_mapping(
    df_mapping: pd.DataFrame,
    mapping_table_name: str,
    from_schema: TableSchema,
    to_schema: TableSchema,
    engine: Engine,
    metadata: MetaData,
) -> None:
    """
    Apply mapping to create result table with process to clean up and roll back if checks fail
    """
    if mapping_table_name in metadata.tables:
        msg = (
            f"table {mapping_table_name} already exists, delete it or use a different table name."
        )
        raise TableAlreadyExists(msg)

    time_configs = [to_schema.time_config]
    if isinstance(from_schema.time_config, DatetimeRange):
        from_time_config = from_schema.time_config.model_copy()
        from_time_config.time_column = "from_" + from_time_config.time_column
        time_configs.append(from_time_config)
    try:
        with engine.connect() as conn:
            write_database(
                df_mapping, conn, mapping_table_name, time_configs, if_table_exists="fail"
            )
            metadata.reflect(engine, views=True)
            _apply_mapping(mapping_table_name, from_schema, to_schema, engine, metadata)
            mapped_table = Table(to_schema.name, metadata)
            try:
                check_timestamps(conn, mapped_table, to_schema)
            except Exception:
                logger.exception(
                    "check_timestamps failed on mapped table {}. Drop it", to_schema.name
                )
                conn.rollback()
                raise
            conn.commit()
    finally:
        if mapping_table_name in metadata.tables:
            with engine.connect() as conn:
                conn.execute(text(f"DROP TABLE {mapping_table_name}"))
                conn.commit()
            metadata.remove(Table(mapping_table_name, metadata))


def _apply_mapping(
    mapping_table_name: str,
    from_schema: TableSchema,
    to_schema: TableSchema,
    engine: Engine,
    metadata: MetaData,
) -> None:
    """Apply mapping to create result as a table according to_schema
    - Columns used to join the from_table are prefixed with "from_" in the mapping table
    """
    left_table = Table(from_schema.name, metadata)
    right_table = Table(mapping_table_name, metadata)
    left_table_columns = [x.name for x in left_table.columns]
    right_table_columns = [x.name for x in right_table.columns]

    final_cols = set(to_schema.list_columns())
    right_cols = set(right_table_columns).intersection(final_cols)
    left_cols = final_cols - right_cols

    select_stmt = [left_table.c[x] for x in left_cols]
    select_stmt += [right_table.c[x] for x in right_cols]

    keys = from_schema.time_config.list_time_columns()
    # infer the use of time_zone
    if "from_time_zone" in right_table_columns:
        keys.append("time_zone")
        assert "time_zone" in left_table_columns, f"time_zone not in table={from_schema.name}"

    on_stmt = reduce(and_, (left_table.c[x] == right_table.c["from_" + x] for x in keys))
    query = select(*select_stmt).select_from(left_table).join(right_table, on_stmt)
    create_table(to_schema.name, query, engine, metadata)
