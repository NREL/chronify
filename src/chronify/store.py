from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import Column, Engine, MetaData, Selectable, Table, create_engine, text

import chronify.duckdb.functions as ddbf
from chronify.exceptions import ConflictingInputsError, InvalidTable
from chronify.csv_io import read_csv
from chronify.models import (
    CsvTableSchema,
    TableSchema,
    TableSchemaBase,
    get_sqlalchemy_type_from_duckdb,
)
from chronify.sqlalchemy.functions import read_database, write_database
from chronify.time_configs import DatetimeRange, IndexTimeRange
from chronify.time_series_checker import check_timestamps
from chronify.utils.sql import make_temp_view_name
from chronify.utils.sqlalchemy_view import create_view


class Store:
    """Data store for time series data"""

    def __init__(
        self,
        engine: Optional[Engine] = None,
        engine_name: Optional[str] = None,
        **connect_kwargs: Any,
    ) -> None:
        """Construct the Store.

        Parameters
        ----------
        engine: sqlalchemy.Engine
            Optional, defaults to a engine connected to an in-memory DuckDB database.

        Examples
        --------
        >>> from sqlalchemy
        >>> store1 = Store()
        >>> store2 = Store(engine=Engine("duckdb:///time_series.db")
        >>> store3 = Store(engine=Engine(engine_name="sqlite")
        >>> store4 = Store(engine=Engine("sqlite:///time_series.db")
        """
        self._metadata = MetaData()
        if engine and engine_name:
            msg = f"{engine=} and {engine_name=} cannot both be set"
            raise ConflictingInputsError(msg)
        if engine is None:
            name = engine_name or "duckdb"
            match name:
                case "duckdb" | "sqlite":
                    engine_path = f"{name}:///:memory:"
                case _:
                    msg = f"{engine_name=}"
                    raise NotImplementedError(msg)
            self._engine = create_engine(engine_path, **connect_kwargs)
        else:
            self._engine = engine

    # TODO
    # @classmethod
    # def load_spark(cls, spark_url: str, **connect_kwargs) -> Store:
    # engine = create_engine(spark_url, **connect_kwargs)
    # return cls(engine)

    # def add_time_series(self, data: np.ndarray) -> None:
    # """Add a time series array to the store."""

    @property
    def engine(self) -> Engine:
        """Return the sqlalchemy engine."""
        return self._engine

    @property
    def metadata(self) -> MetaData:
        """Return the sqlalchemy metadata."""
        return self._metadata

    def create_view_from_parquet(self, name: str, path: Path) -> None:
        """Create a view in the database from a Parquet file."""
        with self._engine.begin() as conn:
            if self._engine.name == "duckdb":
                path_ = f"{path}/**/*.parquet" if path.is_dir() else path
                query = f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path_}')"
            else:
                msg = f"create_view_from_parquet does not support engine={self._engine.name}"
                raise NotImplementedError(msg)
            conn.execute(text(query))
            conn.commit()
        self.update_table_schema()

    def ingest_from_csv(
        self,
        path: Path | str,
        src_schema: CsvTableSchema,
        dst_schema: TableSchema,
    ) -> None:
        """Ingest data into the table specifed by schema. If the table does not exist,
        create it.
        """
        rel = read_csv(path, src_schema)
        check_columns(rel.columns, src_schema.list_columns())
        # TODO: doesn't do anything yet
        check_schema_compatibility(src_schema, dst_schema)

        if src_schema.pivoted_dimension_name is not None:
            rel = ddbf.unpivot(
                rel,
                src_schema.value_columns,
                src_schema.pivoted_dimension_name,
                dst_schema.value_column,
            )

        if isinstance(src_schema.time_config, IndexTimeRange):
            if isinstance(dst_schema.time_config, DatetimeRange):
                rel = ddbf.add_datetime_column(
                    rel=rel,
                    start=dst_schema.time_config.start,
                    resolution=dst_schema.time_config.resolution,
                    length=dst_schema.time_config.length,
                    time_array_id_columns=src_schema.time_array_id_columns,
                    time_column=dst_schema.time_config.time_column,
                    timestamps=src_schema.time_config.list_timestamps(),
                )
            else:
                cls_name = dst_schema.time_config.__class__.__name__
                msg = f"IndexTimeRange cannot be converted to {cls_name}"
                raise NotImplementedError(msg)

        table_exists = self.has_table(dst_schema.name)
        if table_exists:
            table = Table(dst_schema.name, self._metadata)
        else:
            dtypes = [get_sqlalchemy_type_from_duckdb(x) for x in rel.dtypes]
            table = Table(
                dst_schema.name,
                self._metadata,
                *[Column(x, y) for x, y in zip(rel.columns, dtypes)],
            )
            table.create(self._engine)

        with self._engine.begin() as conn:
            write_database(rel.to_df(), conn, dst_schema)
            try:
                check_timestamps(conn, table, dst_schema)
            except Exception:
                conn.rollback()
                if not table_exists:
                    table.drop(self._engine)
                    self.update_table_schema()
                raise
            conn.commit()
        self.update_table_schema()

    def read_query(self, query: Selectable | str, schema: TableSchema) -> pd.DataFrame:
        """Return the query result as a pandas DataFrame."""
        with self._engine.begin() as conn:
            return read_database(query, conn, schema)

    def read_table(self, schema: TableSchema) -> pd.DataFrame:
        """Return the table as a pandas DataFrame."""
        return self.read_query(f"SELECT * FROM {schema.name}", schema)

    def write_query_to_parquet(self, stmt: Selectable, file_path: Path | str) -> None:
        """Write the result of a query to a Parquet file."""
        view_name = make_temp_view_name()
        create_view(view_name, stmt, self._engine, self._metadata)
        try:
            self.write_table_to_parquet(view_name, file_path)
        finally:
            with self._engine.connect() as conn:
                conn.execute(text(f"DROP VIEW {view_name}"))

    def write_table_to_parquet(self, name: str, file_path: Path | str) -> None:
        """Write a table or view to a Parquet file."""
        match self._engine.name:
            case "duckdb":
                cmd = ddbf.make_write_parquet_query(name, file_path)
            # case "spark":
            # pass
            case _:
                msg = f"{self.engine.name=}"
                raise NotImplementedError(msg)

        with self._engine.connect() as conn:
            conn.execute(text(cmd))

        logger.info("Wrote table or view to {}", file_path)

    def has_table(self, name: str) -> bool:
        """Return True if the database has a table with the given name."""
        return name in self._metadata.tables

    def load_table(self, path: Path, schema: TableSchema) -> None:
        """Load a table into the database."""
        # TODO: support unpivoting a pivoted table
        if path.suffix != ".parquet":
            msg = "Only Parquet files are currently supported: {path=}"
            raise NotImplementedError(msg)

        self.create_view_from_parquet(schema.name, path)
        try:
            self._check_table_schema(schema)
            self._check_timestamps(schema)
        except InvalidTable:
            with self._engine.begin() as conn:
                conn.execute(text(f"DROP VIEW {schema.name}"))
            raise

    def update_table_schema(self) -> None:
        """Update the sqlalchemy metadata for table schema. Call this method if you add tables
        in the sqlalchemy engine outside of this class.
        """
        self._metadata.reflect(self._engine, views=True)

    def _check_table_schema(self, schema: TableSchema) -> None:
        table = Table(schema.name, self._metadata)
        columns = {x.name for x in table.columns}
        check_columns(columns, schema.list_columns())

    def _check_timestamps(self, schema: TableSchema) -> None:
        with self._engine.connect() as conn:
            table = Table(schema.name, self._metadata)
            check_timestamps(conn, table, schema)


def check_columns(table_columns: Iterable[str], schema_columns: Iterable[str]) -> None:
    """Check if the columns match the schema.

    Raises
    ------
    InvalidTable
        Raised if the columns don't match the schema.
    """
    expected_columns = set(schema_columns)
    diff = expected_columns.difference(table_columns)
    if diff:
        cols = " ".join(sorted(diff))
        msg = f"These columns are defined in the schema but not present in the table: {cols}"
        raise InvalidTable(msg)


def check_schema_compatibility(src: TableSchemaBase, dst: TableSchemaBase) -> None:
    """Check that a table with src schema can be converted to dst."""
