import json
from collections.abc import Iterable
from pathlib import Path
import shutil
from typing import Any, Optional

import duckdb
import pandas as pd
from duckdb import DuckDBPyRelation
from loguru import logger
from sqlalchemy import (
    Column,
    Connection,
    Engine,
    MetaData,
    Selectable,
    String,
    Table,
    create_engine,
    delete,
    insert,
    select,
    text,
)

import chronify.duckdb.functions as ddbf
from chronify.exceptions import (
    ConflictingInputsError,
    InvalidOperation,
    InvalidParameter,
    InvalidTable,
    TableNotStored,
)
from chronify.csv_io import read_csv
from chronify.models import (
    CsvTableSchema,
    PivotedTableSchema,
    TableSchema,
    TableSchemaBase,
    get_duckdb_types_from_pandas,
    get_sqlalchemy_type_from_duckdb,
)
from chronify.sqlalchemy.functions import read_database, write_database
from chronify.time_configs import DatetimeRange, IndexTimeRange
from chronify.time_series_checker import check_timestamps
from chronify.time_series_mapper import map_time
from chronify.utils.sql import make_temp_view_name
from chronify.utils.sqlalchemy_view import create_view


class Store:
    """Data store for time series data"""

    SCHEMAS_TABLE = "schemas"

    def __init__(
        self,
        engine: Optional[Engine] = None,
        engine_name: Optional[str] = None,
        file_path: Optional[Path | str] = None,
        **connect_kwargs: Any,
    ) -> None:
        """Construct the Store.

        Parameters
        ----------
        engine
            Optional, defaults to a engine connected to an in-memory DuckDB database.
        engine_name
            Optional, name of engine to use ('duckdb', 'sqlite'). Mutually exclusive with engine.
        file_path
            Optional, use this file for the database. If the file does not exist, create a new
            database. If the file exists, load that existing database.
            Defaults to a new in-memory database.

        Examples
        --------
        >>> from sqlalchemy
        >>> store1 = Store()
        >>> store2 = Store(engine=Engine("duckdb:///time_series.db"))
        >>> store3 = Store(engine=Engine(engine_name="sqlite"))
        >>> store4 = Store(engine=Engine("sqlite:///time_series.db"))
        """
        self._metadata = MetaData()
        if engine and engine_name:
            msg = f"{engine=} and {engine_name=} cannot both be set"
            raise ConflictingInputsError(msg)
        filename = ":memory:" if file_path is None else str(file_path)
        if engine is None:
            name = engine_name or "duckdb"
            match name:
                case "duckdb" | "sqlite":
                    engine_path = f"{name}:///{filename}"
                case _:
                    msg = f"{engine_name=}"
                    raise NotImplementedError(msg)
            self._engine = create_engine(engine_path, **connect_kwargs)
        else:
            self._engine = engine

        self.update_table_schema()
        if self.has_table(self.SCHEMAS_TABLE):
            logger.info("Loaded existing database {}", file_path)
        else:
            table = Table(
                self.SCHEMAS_TABLE,
                self._metadata,
                Column("name", String),
                Column("schema", String),  # schema encoded as JSON
            )
            # TODO: will this work for Spark? Needs to be mutable.
            table.create(self._engine)
            self.update_table_schema()
            logger.info("Initialized new database. file = {}", filename)

    def backup(self, dst: Path | str, overwrite: bool = False) -> None:
        """Copy the database to a new location. Not supported for in-memory databases."""
        self._engine.dispose()
        path = Path(dst) if isinstance(dst, str) else dst
        if path.exists():
            if not overwrite:
                msg = f"{path} already exists. Choose a different path or set overwrite=True."
                raise InvalidParameter(msg)
        match self._engine.name:
            case "duckdb" | "sqlite":
                if self._engine.url.database is None or self._engine.url.database == ":memory:":
                    msg = "backup is only supported with a database backed by a file"
                    raise InvalidOperation(msg)
                src_file = Path(self._engine.url.database)
                shutil.copyfile(src_file, path)
                logger.info("Copied database to {}", path)
            case _:
                msg = self._engine.name
                raise NotImplementedError(msg)

    # TODO
    # @classmethod
    # def load_spark(cls, spark_url: str, **connect_kwargs) -> Store:
    # engine = create_engine(spark_url, **connect_kwargs)
    # return cls(engine)

    @property
    def engine(self) -> Engine:
        """Return the sqlalchemy engine."""
        return self._engine

    @property
    def metadata(self) -> MetaData:
        """Return the sqlalchemy metadata."""
        return self._metadata

    def create_view_from_parquet(self, schema: TableSchema, path: Path) -> None:
        """Create a view in the database from a Parquet file."""
        with self._engine.begin() as conn:
            if self._engine.name == "duckdb":
                path_ = f"{path}/**/*.parquet" if path.is_dir() else path
                query = f"CREATE VIEW {schema.name} AS SELECT * FROM read_parquet('{path_}')"
            else:
                msg = f"create_view_from_parquet does not support engine={self._engine.name}"
                raise NotImplementedError(msg)
            conn.execute(text(query))
            self._add_schema(conn, schema)
            conn.commit()
        self.update_table_schema()

    def _get_schema_table(self) -> Table:
        table = self.get_table(self.SCHEMAS_TABLE)
        assert table is not None
        return table

    def get_table(self, name: str) -> Table | None:
        """Return the sqlalchemy Table object or None if it is not stored."""
        if not self.has_table(name):
            return None
        return Table(name, self._metadata)

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

        # TODO
        if isinstance(src_schema.time_config, IndexTimeRange):
            if isinstance(dst_schema.time_config, DatetimeRange):
                raise NotImplementedError
                # timestamps = IndexTimeRangeGenerator(src_schema.time_config).list_timestamps()
                # rel = ddbf.add_datetime_column(
                #    rel=rel,
                #    start=dst_schema.time_config.start,
                #    resolution=dst_schema.time_config.resolution,
                #    length=dst_schema.time_config.length,
                #    time_array_id_columns=src_schema.time_array_id_columns,
                #    time_column=dst_schema.time_config.time_column,
                #    timestamps=timestamps,
                # )
            else:
                cls_name = dst_schema.time_config.__class__.__name__
                msg = f"IndexTimeRange cannot be converted to {cls_name}"
                raise NotImplementedError(msg)

        if src_schema.pivoted_dimension_name is not None:
            return self.ingest_pivoted_table(rel, src_schema, dst_schema)

        self.ingest_table(rel, dst_schema)

    def ingest_pivoted_table(
        self,
        data: pd.DataFrame | DuckDBPyRelation,
        src_schema: PivotedTableSchema | CsvTableSchema,
        dst_schema: TableSchema,
    ) -> None:
        """Ingest pivoted data into the table specifed by schema. If the table does not exist,
        create it.
        """
        if isinstance(data, pd.DataFrame):
            # This is a shortcut for registering a temporary view.
            tmp_df = data  # noqa: F841
            rel = duckdb.sql("SELECT * from tmp_df")
        else:
            rel = data
        assert src_schema.pivoted_dimension_name is not None
        rel2 = ddbf.unpivot(
            rel,
            src_schema.value_columns,
            src_schema.pivoted_dimension_name,
            dst_schema.value_column,
        )
        self.ingest_table(rel2, dst_schema)

    def ingest_table(
        self,
        data: pd.DataFrame | DuckDBPyRelation,
        dst_schema: TableSchema,
    ) -> None:
        """Ingest data into the table specifed by schema. If the table does not exist,
        create it.
        """
        df = data.to_df() if isinstance(data, DuckDBPyRelation) else data
        check_columns(df.columns, dst_schema.list_columns())
        table = self.get_table(dst_schema.name)
        if table is None:
            duckdb_types = (
                data.dtypes
                if isinstance(data, DuckDBPyRelation)
                else get_duckdb_types_from_pandas(data)
            )
            dtypes = [get_sqlalchemy_type_from_duckdb(x) for x in duckdb_types]
            table = Table(
                dst_schema.name,
                self._metadata,
                *[Column(x, y) for x, y in zip(df.columns, dtypes)],
            )
            table.create(self._engine)
            created_table = True
        else:
            created_table = False

        with self._engine.begin() as conn:
            write_database(df, conn, dst_schema.name, dst_schema.time_config)
            try:
                check_timestamps(conn, table, dst_schema)
            except Exception:
                conn.rollback()
                if created_table:
                    table.drop(self._engine)
                    self.update_table_schema()
                raise
            if created_table:
                self._add_schema(conn, dst_schema)
            conn.commit()

        if created_table:
            self.update_table_schema()

    def list_tables(self) -> list[str]:
        """Return a list of user tables in the database."""
        return [x for x in self._metadata.tables if x != self.SCHEMAS_TABLE]

    def map_time(self, src_schema: TableSchema, dst_schema: TableSchema) -> None:
        """Map the existing table represented by src_schema to a new table represented by
        dst_schema.
        """
        map_time(self._engine, self._metadata, src_schema, dst_schema)
        with self._engine.connect() as conn:
            self._add_schema(conn, dst_schema)
            conn.commit()

    def read_query(self, name: str, query: Selectable | str) -> pd.DataFrame:
        """Return the query result as a pandas DataFrame.

        Parameters
        ----------
        name
            Table or view name
        query
            SQL query expressed as a string or salqlchemy Selectable
        """
        schema = self._get_schema(name)
        with self._engine.begin() as conn:
            return read_database(query, conn, schema.time_config)

    def read_table(self, name: str) -> pd.DataFrame:
        """Return the table as a pandas DataFrame."""
        table = self.get_table(name)
        if table is None:
            msg = f"{name=}"
            raise TableNotStored(msg)
        stmt = select(table)
        return self.read_query(name, stmt)

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

        self.create_view_from_parquet(schema, path)
        try:
            self._check_table_schema(schema)
            self._check_timestamps(schema)
        except InvalidTable:
            self.drop_view(schema.name)
            raise

    def delete_rows(self, name: str, time_array_id_columns: dict[str, Any]) -> None:
        """Delete all rows matching the time_array_id_columns."""
        table = self.get_table(name)
        if table is None:
            msg = f"No table with {name=} is stored"
            raise TableNotStored(msg)

        if not time_array_id_columns:
            msg = "time_array_id_columns cannot be empty"
            raise InvalidParameter(msg)

        assert time_array_id_columns
        stmt = delete(table)
        for column, value in time_array_id_columns.items():
            stmt = stmt.where(table.c[column] == value)

        with self._engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()

        logger.info(
            "Delete all rows from table {} with time_array_id_columns {}",
            name,
            time_array_id_columns,
        )

        stmt2 = select(table).limit(1)
        is_empty = False
        with self._engine.connect() as conn:
            res = conn.execute(stmt2).fetchall()
            if not res:
                is_empty = True

        if is_empty:
            logger.info("Delete empty table {}", name)
            self.drop_table(name)

    def drop_table(self, name: str) -> None:
        """Drop a table from the database."""
        self._drop_table_or_view(name, "TABLE")

    def drop_view(self, name: str) -> None:
        """Drop a view from the database."""
        self._drop_table_or_view(name, "VIEW")

    def _drop_table_or_view(self, name: str, tbl_type: str) -> None:
        table = self.get_table(name)
        if table is None:
            msg = f"{name=}"
            raise TableNotStored(msg)

        with self._engine.connect() as conn:
            conn.execute(text(f"DROP {tbl_type} {name}"))
            self._remove_schema(conn, name)
            conn.commit()

        self._metadata.remove(table)
        logger.info("Dropped {} {}", tbl_type.lower(), name)

    def update_table_schema(self) -> None:
        """Update the sqlalchemy metadata for table schema. Call this method if you add tables
        in the sqlalchemy engine outside of this class.
        """
        self._metadata.reflect(self._engine, views=True)

    def _add_schema(self, conn: Connection, schema: TableSchema) -> None:
        table = self._get_schema_table()
        stmt = insert(table).values(name=schema.name, schema=schema.model_dump_json())
        conn.execute(stmt)
        logger.debug("Added schema for table {}", schema.name)

    def _get_schema(self, name: str) -> TableSchema:
        table = self._get_schema_table()
        with self._engine.connect() as conn:
            stmt = select(table.c.schema).where(table.c["name"] == name)
            res = conn.execute(stmt).fetchall()
            length = len(res)
            if length == 0:
                msg = f"table={name} is not stored in the schemas"
                raise TableNotStored(msg)
            elif length > 1:
                msg = f"Bug: found more than one table={name} in the schemas."
                raise Exception(msg)
            text = res[0][0]
            return TableSchema(**json.loads(text))

    def _remove_schema(self, conn: Connection, name: str) -> None:
        table = self._get_schema_table()
        stmt = delete(table).where(table.c["name"] == name)
        conn.execute(stmt)

    def _check_table_schema(self, schema: TableSchema) -> None:
        table = self.get_table(schema.name)
        assert table is not None
        columns = {x.name for x in table.columns}
        check_columns(columns, schema.list_columns())

    def _check_timestamps(self, schema: TableSchema) -> None:
        with self._engine.connect() as conn:
            table = self.get_table(schema.name)
            assert table is not None
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
