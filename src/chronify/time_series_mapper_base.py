import abc
from functools import reduce
from operator import and_
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from sqlalchemy import Engine, MetaData, Table, select, text
from chronify.hive_functions import create_materialized_view

from chronify.sqlalchemy.functions import write_database
from chronify.models import TableSchema, MappingTableSchema
from chronify.exceptions import TableAlreadyExists, ConflictingInputsError
from chronify.utils.sqlalchemy_table import create_table
from chronify.time_series_checker import check_timestamps
from chronify.time import TimeIntervalType


class TimeSeriesMapperBase(abc.ABC):
    """Maps time series data from one configuration to another."""

    def __init__(
        self, engine: Engine, metadata: MetaData, from_schema: TableSchema, to_schema: TableSchema
    ) -> None:
        self._engine = engine
        self._metadata = metadata
        self._from_schema = from_schema
        self._to_schema = to_schema
        # self._from_time_config = from_schema.time_config
        # self._to_time_config = to_schema.time_config

    @abc.abstractmethod
    def check_schema_consistency(self) -> None:
        """Check that from_schema can produce to_schema."""

    def _check_table_columns_producibility(self) -> None:
        """Check columns in destination table can be produced by source table."""
        available_cols = (
            self._from_schema.list_columns() + self._to_schema.time_config.list_time_columns()
        )
        final_cols = self._to_schema.list_columns()
        if diff := set(final_cols) - set(available_cols):
            msg = f"Source table {self._from_schema.name} cannot produce the columns: {diff}"
            raise ConflictingInputsError(msg)

    def _check_measurement_type_consistency(self) -> None:
        """Check that measurement_type is the same between schema."""
        from_mt = self._from_schema.time_config.measurement_type
        to_mt = self._to_schema.time_config.measurement_type
        if from_mt != to_mt:
            msg = f"Inconsistent measurement_types {from_mt=} vs. {to_mt=}"
            raise ConflictingInputsError(msg)

    def _check_time_interval_type(self) -> None:
        """Check time interval type consistency."""
        from_interval = self._from_schema.time_config.interval_type
        to_interval = self._to_schema.time_config.interval_type
        if TimeIntervalType.INSTANTANEOUS in (from_interval, to_interval) and (
            from_interval != to_interval
        ):
            msg = "If instantaneous time interval is used, it must exist in both from_scheme and to_schema."
            raise ConflictingInputsError(msg)

    @abc.abstractmethod
    def map_time(self) -> None:
        """Convert time columns with from_schema to to_schema configuration."""


def apply_mapping(
    df_mapping: pd.DataFrame,
    mapping_schema: MappingTableSchema,
    from_schema: TableSchema,
    to_schema: TableSchema,
    engine: Engine,
    metadata: MetaData,
    scratch_dir: Optional[Path] = None,
) -> None:
    """
    Apply mapping to create result table with process to clean up and roll back if checks fail
    """
    if mapping_schema.name in metadata.tables:
        msg = (
            f"table {mapping_schema.name} already exists, delete it or use a different table name."
        )
        raise TableAlreadyExists(msg)

    try:
        with engine.begin() as conn:
            write_database(
                df_mapping,
                conn,
                mapping_schema.name,
                mapping_schema.time_configs,
                if_table_exists="fail",
            )

        metadata.reflect(engine, views=True)
        _apply_mapping(mapping_schema.name, from_schema, to_schema, engine, metadata, scratch_dir)
        mapped_table = Table(to_schema.name, metadata)
        try:
            with engine.connect() as conn:
                check_timestamps(conn, mapped_table, to_schema)
        except Exception:
            logger.exception("check_timestamps failed on mapped table {}. Drop it", to_schema.name)
            raise
    finally:
        if mapping_schema.name in metadata.tables:
            with engine.begin() as conn:
                table_type = "view" if engine.name == "hive" else "table"
                conn.execute(text(f"DROP {table_type} {mapping_schema.name}"))
            metadata.remove(Table(mapping_schema.name, metadata))


def _apply_mapping(
    mapping_table_name: str,
    from_schema: TableSchema,
    to_schema: TableSchema,
    engine: Engine,
    metadata: MetaData,
    scratch_dir: Optional[Path] = None,
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
    if engine.name == "hive":
        create_materialized_view(
            str(query), to_schema.name, engine, metadata, scratch_dir=scratch_dir
        )
    else:
        create_table(to_schema.name, query, engine, metadata)
