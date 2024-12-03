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
    Table,
    create_engine,
    delete,
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
from chronify.schema_manager import SchemaManager
from chronify.time_configs import DatetimeRange, IndexTimeRange
from chronify.time_series_checker import check_timestamps
from chronify.time_series_mapper import map_time
from chronify.utils.sql import make_temp_view_name
from chronify.utils.sqlalchemy_view import create_view


class Store:
    """Data store for time series data"""

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
        >>> store3 = Store(engine=Engine("sqlite:///time_series.db"))
        >>> store4 = Store(engine_name="sqlite")
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

        self._schema_mgr = SchemaManager(self._engine, self._metadata)
        if self._engine.url.database != ":memory:":
            self.update_sqlalchemy_metadata()

    @classmethod
    def create_in_memory_db(
        cls,
        engine_name: str = "duckdb",
        **connect_kwargs: Any,
    ) -> "Store":
        """Create a Store with an in-memory database."""
        return Store(engine=create_engine(f"{engine_name}:///:memory:", **connect_kwargs))

    @classmethod
    def create_file_db(
        cls,
        file_path: Path | str = "time_series.db",
        engine_name: str = "duckdb",
        overwrite: bool = False,
        **connect_kwargs: Any,
    ) -> "Store":
        """Create a Store with a file-based database."""
        path = file_path if isinstance(file_path, Path) else Path(file_path)
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                msg = f"{path} already exists. Choose a different path or set overwrite=True."
                raise InvalidOperation(msg)
        return Store(engine=create_engine(f"{engine_name}:///{path}", **connect_kwargs))

    @classmethod
    def load_from_file(
        cls,
        file_path: Path | str,
        engine_name: str = "duckdb",
        **connect_kwargs: Any,
    ) -> "Store":
        """Load an existing store from a database."""
        path = file_path if isinstance(file_path, Path) else Path(file_path)
        if not path.exists():
            msg = str(path)
            raise FileNotFoundError(msg)
        return Store(engine=create_engine(f"{engine_name}:///{path}", **connect_kwargs))

    def get_table(self, name: str) -> Table:
        """Return the sqlalchemy Table object."""
        if not self.has_table(name):
            msg = f"{name=}"
            raise TableNotStored(msg)
        return Table(name, self._metadata)

    def has_table(self, name: str) -> bool:
        """Return True if the database has a table with the given name."""
        return name in self._metadata.tables

    def list_tables(self) -> list[str]:
        """Return a list of user tables in the database."""
        return [x for x in self._metadata.tables if x != SchemaManager.SCHEMAS_TABLE]

    def try_get_table(self, name: str) -> Table | None:
        """Return the sqlalchemy Table object or None if it is not stored."""
        if not self.has_table(name):
            return None
        return Table(name, self._metadata)

    def update_sqlalchemy_metadata(self) -> None:
        """Update the sqlalchemy metadata for table schema. Call this method if you add tables
        in the sqlalchemy engine outside of this class.
        """
        self._metadata.reflect(self._engine, views=True)

    def backup(self, dst: Path | str, overwrite: bool = False) -> None:
        """Copy the database to a new location. Not yet supported for in-memory databases."""
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

    def create_view_from_parquet(self, schema: TableSchema, path: Path | str) -> None:
        """Create a view in the database from a Parquet file.

        Parameters
        ----------
        schema
            Defines the schema of the view to create in the database. Must match the input data.
        path
            Path to Parquet file.

        Raises
        ------
        InvalidTable
            Raised if the schema does not match the input data.

        Examples
        --------
        >>> store = Store()
        >>> store.create_view_from_parquet(
        ...     TableSchema(
        ...         name="devices",
        ...         value_column="value",
        ...         time_config=DatetimeRange(
        ...             time_column="timestamp",
        ...             start=datetime(2020, 1, 1, 0),
        ...             length=8784,
        ...             resolution=timedelta(hours=1),
        ...         ),
        ...         time_array_id_columns=["id"],
        ...     ),
        ...     "table.parquet",
        ... )
        """
        path_ = path if isinstance(path, Path) else Path(path)
        with self._engine.begin() as conn:
            if self._engine.name == "duckdb":
                str_path = f"{path}/**/*.parquet" if path_.is_dir() else str(path_)
                query = f"CREATE VIEW {schema.name} AS SELECT * FROM read_parquet('{str_path}')"
            else:
                msg = f"create_view_from_parquet does not support engine={self._engine.name}"
                raise NotImplementedError(msg)
            conn.execute(text(query))
            self._schema_mgr.add_schema(conn, schema)
            conn.commit()

        self.update_sqlalchemy_metadata()

    def ingest_from_csv(
        self,
        path: Path | str,
        src_schema: CsvTableSchema,
        dst_schema: TableSchema,
        connection: Optional[Connection] = None,
    ) -> None:
        """Ingest data from a CSV file.

        Parameters
        ----------
        path
            Source data file
        src_schema
            Defines the schema of the source file.
        dst_schema
            Defines the destination table in the database.
        connection
            Optional connection to reuse. Refer to :meth:`ingest_table` for notes.

        Raises
        ------
        InvalidTable
            Raised if the data does not match the schema.

        Examples
        --------
        >>> resolution = timedelta(hours=1)
        >>> time_config = DatetimeRange(
        ...     time_column="timestamp",
        ...     start=datetime(2020, 1, 1, 0),
        ...     length=8784,
        ...     resolution=timedelta(hours=1),
        ... )
        >>> store = Store()
        >>> store.ingest_from_csv(
        ...     "data.csv",
        ...     CsvTableSchema(
        ...         time_config=time_config,
        ...         pivoted_dimension_name="device",
        ...         value_columns=["device1", "device2", "device3"],
        ...     ),
        ...     TableSchema(
        ...         name="devices",
        ...         value_column="value",
        ...         time_config=time_config,
        ...         time_array_id_columns=["device"],
        ...     ),
        ... )

        See Also
        --------
        ingest_from_csvs
        """
        return self.ingest_from_csvs((path,), src_schema, dst_schema, connection=connection)

    def ingest_from_csvs(
        self,
        paths: Iterable[Path | str],
        src_schema: CsvTableSchema,
        dst_schema: TableSchema,
        connection: Optional[Connection] = None,
    ) -> None:
        """Ingest data into the table specifed by schema. If the table does not exist,
        create it. This is faster than calling :meth:`ingest_from_csv` many times.
        Each file is loaded into memory one at a time.
        If any error occurs, all added data will be removed and the state of the database will
        be the same as the original state.

        Parameters
        ----------
        path
            Source data files
        src_schema
            Defines the schema of the source files.
        dst_schema
            Defines the destination table in the database.
        connection
            Optional connection to reuse. Refer to :meth:`ingest_table` for notes.

        Raises
        ------
        InvalidTable
            Raised if the data does not match the schema.

        See Also
        --------
        ingest_from_csv
        """
        if connection is None:
            with self._engine.connect() as conn:
                try:
                    self._ingest_from_csvs(conn, paths, src_schema, dst_schema)
                    conn.commit()
                except Exception:
                    logger.exception("Failed to ingest_from_csvs")
                    conn.rollback()
                    raise
            self._schema_mgr.update_cache(dst_schema)
        else:
            # Let the caller commit or rollback when ready.
            self._ingest_from_csvs(connection, paths, src_schema, dst_schema)

    def _ingest_from_csvs(
        self,
        connection: Connection,
        paths: Iterable[Path | str],
        src_schema: CsvTableSchema,
        dst_schema: TableSchema,
    ) -> None:
        if not paths:
            return

        for path in paths:
            self._ingest_from_csv(connection, path, src_schema, dst_schema)
        table = Table(dst_schema.name, self._metadata)
        check_timestamps(connection, table, dst_schema)

    def _ingest_from_csv(
        self,
        conn: Connection,
        path: Path | str,
        src_schema: CsvTableSchema,
        dst_schema: TableSchema,
    ) -> None:
        rel = read_csv(path, src_schema)
        columns = set(src_schema.list_columns())
        check_columns(rel.columns, columns)
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
            return self._ingest_pivoted_table(conn, rel, src_schema, dst_schema)

        return self._ingest_table(conn, rel, dst_schema)

    def ingest_pivoted_table(
        self,
        data: pd.DataFrame | DuckDBPyRelation,
        src_schema: PivotedTableSchema | CsvTableSchema,
        dst_schema: TableSchema,
        connection: Optional[Connection] = None,
    ) -> None:
        """Ingest pivoted data into the table specifed by schema. If the table does not exist,
        create it. Chronify will unpivot the data before ingesting it.

        Parameters
        ----------
        data
            Input data to ingest into the database.
        src_schema
            Defines the schema of the input data.
        dst_schema
            Defines the destination table in the database.
        connection
            Optional connection to reuse. Refer to :meth:`ingest_table` for notes.

        Raises
        ------
        InvalidTable
            Raised if the data does not match the schema.

        Examples
        --------
        >>> resolution = timedelta(hours=1)
        >>> df = pd.DataFrame(
        ...     {
        ...         "timestamp": pd.date_range(
        ...             "2020-01-01", "2020-12-31 23:00:00", freq=resolution
        ...         ),
        ...         "device1": np.random.random(8784),
        ...         "device2": np.random.random(8784),
        ...         "device3": np.random.random(8784),
        ...     }
        ... )
        >>> time_config = DatetimeRange(
        ...     time_column="timestamp",
        ...     start=datetime(2020, 1, 1, 0),
        ...     length=8784,
        ...     resolution=timedelta(hours=1),
        ... )
        >>> store = Store()
        >>> store.ingest_pivoted_table(
        ...     df,
        ...     PivotedTableSchema(
        ...         time_config=time_config,
        ...         pivoted_dimension_name="device",
        ...         value_columns=["device1", "device2", "device3"],
        ...     ),
        ...     TableSchema(
        ...         name="devices",
        ...         value_column="value",
        ...         time_config=time_config,
        ...         time_array_id_columns=["device"],
        ...     ),
        ... )

        See Also
        --------
        ingest_pivoted_tables
        """
        return self.ingest_pivoted_tables((data,), src_schema, dst_schema, connection=connection)

    def ingest_pivoted_tables(
        self,
        data: Iterable[pd.DataFrame | DuckDBPyRelation],
        src_schema: PivotedTableSchema | CsvTableSchema,
        dst_schema: TableSchema,
        connection: Optional[Connection] = None,
    ) -> None:
        """Ingest pivoted data into the table specifed by schema.

        If the table does not exist, create it. Unpivot the data before ingesting it.
        This is faster than calling :meth:`ingest_pivoted_table` many times.
        If any error occurs, all added data will be removed and the state of the database will
        be the same as the original state.

        Parameters
        ----------
        data
            Data to ingest into the database.
        src_schema
            Defines the schema of all input tables.
        dst_schema
            Defines the destination table in the database.
        connection
            Optional connection to reuse. Refer to :meth:`ingest_table` for notes.

        See Also
        --------
        ingest_pivoted_table
        """
        if connection is None:
            with self._engine.connect() as conn:
                try:
                    self._ingest_pivoted_tables(conn, data, src_schema, dst_schema)
                    conn.commit()
                except Exception:
                    logger.exception("Failed to ingest_pivoted_tables")
                    conn.rollback()
                    raise
            self._schema_mgr.update_cache(dst_schema)
        else:
            # Let the caller commit or rollback when ready.
            self._ingest_pivoted_tables(connection, data, src_schema, dst_schema)

    def _ingest_pivoted_tables(
        self,
        connection: Connection,
        data: Iterable[pd.DataFrame | DuckDBPyRelation],
        src_schema: PivotedTableSchema | CsvTableSchema,
        dst_schema: TableSchema,
    ) -> None:
        for table in data:
            self._ingest_pivoted_table(connection, table, src_schema, dst_schema)
        check_timestamps(connection, Table(dst_schema.name, self._metadata), dst_schema)

    def _ingest_pivoted_table(
        self,
        conn: Connection,
        data: pd.DataFrame | DuckDBPyRelation,
        src_schema: PivotedTableSchema | CsvTableSchema,
        dst_schema: TableSchema,
    ) -> None:
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
        return self._ingest_table(conn, rel2, dst_schema)

    def ingest_table(
        self,
        data: pd.DataFrame | DuckDBPyRelation,
        schema: TableSchema,
        connection: Optional[Connection] = None,
    ) -> None:
        """Ingest data into the table specifed by schema. If the table does not exist,
        create it.

        Parameters
        ----------
        data
            Input data to ingest into the database.
        schema
            Defines the destination table in the database.
        connection
            Optional connection to reuse. If adding many tables at once, it is significantly
            faster to use one connection. Refer to :meth:`ingest_tables` for built-in support.
            If connection is not set, chronify will commit the database changes
            or perform a rollback on error. If it is set, the caller must perform those actions.

        Raises
        ------
        InvalidTable
            Raised if the data does not match the schema.

        Examples
        --------
        >>> store = Store()
        >>> resolution = timedelta(hours=1)
        >>> df = pd.DataFrame(
        ...     {
        ...         "timestamp": pd.date_range(
        ...             "2020-01-01", "2020-12-31 23:00:00", freq=resolution
        ...         ),
        ...         "value": np.random.random(8784),
        ...     }
        ... )
        >>> df["id"] = 1
        >>> store.ingest_table(
        ...     df,
        ...     TableSchema(
        ...         name="devices",
        ...         value_column="value",
        ...         time_config=DatetimeRange(
        ...             time_column="timestamp",
        ...             start=datetime(2020, 1, 1, 0),
        ...             length=8784,
        ...             resolution=timedelta(hours=1),
        ...         ),
        ...         time_array_id_columns=["id"],
        ...     ),
        ... )

        See Also
        --------
        ingest_tables
        """
        return self.ingest_tables((data,), schema, connection=connection)

    def ingest_tables(
        self,
        data: Iterable[pd.DataFrame | DuckDBPyRelation],
        schema: TableSchema,
        connection: Optional[Connection] = None,
    ) -> None:
        """Ingest multiple input tables to the same database table.
        All tables must have the same schema.
        This offers significant performance advantages over calling :meth:`ingest_table` many
        times.

        Parameters
        ----------
        data
            Input tables to ingest into one database table.
        schema
            Defines the destination table.
        connection
            Optional connection to reuse. Refer to :meth:`ingest_table` for notes.

        Raises
        ------
        InvalidTable
            Raised if the data does not match the schema.

        See Also
        --------
        ingest_table
        """
        if not data:
            return

        if connection is None:
            with self._engine.connect() as conn:
                try:
                    self._ingest_tables(conn, data, schema)
                    conn.commit()
                except Exception:
                    logger.exception("Failed to ingest_tables")
                    conn.rollback()
                    raise
            self._schema_mgr.update_cache(schema)
        else:
            # Let the caller commit or rollback when ready.
            self._ingest_tables(connection, data, schema)

    def _ingest_tables(
        self,
        conn: Connection,
        data: Iterable[pd.DataFrame | DuckDBPyRelation],
        schema: TableSchema,
    ) -> None:
        for table in data:
            self._ingest_table(conn, table, schema)
        check_timestamps(conn, Table(schema.name, self._metadata), schema)

    def _ingest_table(
        self,
        conn: Connection,
        data: pd.DataFrame | DuckDBPyRelation,
        schema: TableSchema,
    ) -> None:
        df = data.to_df() if isinstance(data, DuckDBPyRelation) else data
        check_columns(df.columns, schema.list_columns())

        table = self.try_get_table(schema.name)
        if table is None:
            duckdb_types = get_duckdb_types_from_pandas(df)
            dtypes = [get_sqlalchemy_type_from_duckdb(x) for x in duckdb_types]
            table = Table(
                schema.name,
                self._metadata,
                *[Column(x, y) for x, y in zip(df.columns, dtypes)],
            )
            self._metadata.create_all(self._engine, tables=[table])
            created_table = True
        else:
            created_table = False

        try:
            write_database(df, conn, schema.name, schema.time_config)
        except Exception:
            if created_table:
                table.drop(self._engine)
            raise

        if created_table:
            self._schema_mgr.add_schema(conn, schema)
            if self._engine.name == "sqlite":
                # It's possible that this should be a kwarg or config option.
                id_cols = ",".join(schema.time_array_id_columns)
                query = f"CREATE INDEX {schema.name}_index ON {schema.name}({id_cols})"
                conn.execute(text(query))
            # Indexes don't seem to matter for duckdb.

    def map_table_time_config(self, src_name: str, dst_schema: TableSchema) -> None:
        """Map the existing table represented by src_name to a new table represented by
        dst_schema with a different time configuration.

        Parameters
        ----------
        src_name
            Refers to the table name of the source data.
        dst_schema
            Defines the table to create in the database. Must not already exist.

        Raises
        ------
        InvalidTable
            Raised if the schemas are incompatible.

        Examples
        --------
        >>> store = Store()
        >>> hours_per_year = 12 * 7 * 24
        >>> num_time_arrays = 3
        >>> df = pd.DataFrame(
        ...     {
        ...         "id": np.concat(
        ...             [np.repeat(i, hours_per_year) for i in range(1, 1 + num_time_arrays)]
        ...         ),
        ...         "month": np.tile(np.repeat(range(1, 13), 7 * 24), num_time_arrays),
        ...         "day_of_week": np.tile(np.tile(np.repeat(range(7), 24), 12), num_time_arrays),
        ...         "hour": np.tile(np.tile(range(24), 12 * 7), num_time_arrays),
        ...         "value": np.random.random(hours_per_year * num_time_arrays),
        ...     }
        ... )
        >>> schema = TableSchema(
        ...     name="devices_by_representative_time",
        ...     value_column="value",
        ...     time_config=RepresentativePeriodTime(
        ...         time_format=RepresentativePeriodFormat.ONE_WEEK_PER_MONTH_BY_HOUR,
        ...     ),
        ...     time_array_id_columns=["id"],
        ... )
        >>> store.ingest_table(df, schema)
        >>> store.map_table_time_config(
        ...     "devices_by_representative_time",
        ...     TableSchema(
        ...         name="devices_by_datetime",
        ...         value_column="value",
        ...         time_config=DatetimeRange(
        ...             time_column="timestamp",
        ...             start=datetime(2020, 1, 1, 0),
        ...             length=8784,
        ...             resolution=timedelta(hours=1),
        ...         ),
        ...         time_array_id_columns=["id"],
        ...     ),
        ... )
        """
        src_schema = self._schema_mgr.get_schema(src_name)
        map_time(self._engine, self._metadata, src_schema, dst_schema)
        with self._engine.connect() as conn:
            self._schema_mgr.add_schema(conn, dst_schema)
            conn.commit()
            self._schema_mgr.update_cache(dst_schema)

    def read_query(
        self,
        name: str,
        query: Selectable | str,
        params: Any = None,
        connection: Optional[Connection] = None,
    ) -> pd.DataFrame:
        """Return the query result as a pandas DataFrame.

        Parameters
        ----------
        name
            Table or view name
        query
            SQL query expressed as a string or salqlchemy Selectable
        params
            Parameters for SQL query if expressed as a string

        Examples
        --------
        >>> df = store.read_query("SELECT * FROM devices")
        >>> df = store.read_query("SELECT * FROM devices WHERE id = ?", params=(3,))

        >>> from sqlalchemy import select
        >>> table = store.schemas.get_table("devices")
        >>> df = store.read_query(select(table).where(table.c.id == 3)
        """
        schema = self._schema_mgr.get_schema(name)
        if connection is None:
            with self._engine.begin() as conn:
                return read_database(query, conn, schema.time_config, params=params)
        else:
            return read_database(query, connection, schema.time_config, params=params)

    def read_table(self, name: str, connection: Optional[Connection] = None) -> pd.DataFrame:
        """Return the table as a pandas DataFrame."""
        table = self.get_table(name)
        stmt = select(table)
        return self.read_query(name, stmt, connection=connection)

    def read_raw_query(
        self, query: str, params: Any = None, connection: Optional[Connection] = None
    ) -> pd.DataFrame:
        """Execute a query directly on the backend database connection, bypassing sqlalchemy, and
        return the results as a DataFrame.

        Note: Unlike :meth:`read_query`, no conversion of timestamps is performed.
        Timestamps will be in the format of the underlying database. SQLite backends will return
        strings instead of datetime.

        Parameters
        ----------
        query
            SQL query to execute
        params
            Optional parameters for SQL query
        connection
            Optional sqlalchemy connection returned by `Store.engine.connect()`. This can
            improve performance when performing many reads. If used for database modifications,
            it is the caller's responsibility to perform a commit and ensure that the connection
            is closed correctly. Use of sqlalchemy's context manager is recommended.

        Examples
        --------
        >>> store = Store()
        >>> query1 = "SELECT * from my_table WHERE column = ?"
        >>> params1 = ("value1",)
        >>> query2 = "SELECT * from my_table WHERE column = ?'"
        >>> params2 = ("value2",)

        >>> df = store.read_raw_query(query1, params=params1)

        >>> with store.engine.connect() as conn:
        ...     df1 = store.read_raw_query(query1, params=params1, connection=conn)
        ...     df2 = store.read_raw_query(query2, params=params2, connection=conn)
        """
        if connection is None:
            with self._engine.connect() as conn:
                return self._read_raw_query(query, params, conn)
        else:
            return self._read_raw_query(query, params, connection)

    def _read_raw_query(self, query: str, params: Any, conn: Connection) -> pd.DataFrame:
        assert conn._dbapi_connection is not None
        assert conn._dbapi_connection.driver_connection is not None
        match self._engine.name:
            case "duckdb":
                df = conn._dbapi_connection.driver_connection.sql(query, params=params).to_df()
                assert isinstance(df, pd.DataFrame)
                return df
            case "sqlite":
                return pd.read_sql(query, conn._dbapi_connection.driver_connection, params=params)
            case _:
                msg = self._engine.name
                raise NotImplementedError(msg)

    def write_query_to_parquet(self, stmt: Selectable, file_path: Path | str) -> None:
        """Write the result of a query to a Parquet file."""
        view_name = make_temp_view_name()
        create_view(view_name, stmt, self._engine, self._metadata)
        try:
            self.write_table_to_parquet(view_name, file_path)
        finally:
            with self._engine.connect() as conn:
                conn.execute(text(f"DROP VIEW {view_name}"))
            self._metadata.remove(Table(view_name, self._metadata))

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

    def load_table(self, path: Path, schema: TableSchema) -> None:
        """Load a table into the database."""
        # TODO: support unpivoting a pivoted table
        if path.suffix != ".parquet":
            msg = "Only Parquet files are currently supported: {path=}"
            raise NotImplementedError(msg)

        self.create_view_from_parquet(schema, path)
        try:
            with self._engine.connect() as conn:
                table = self.get_table(schema.name)
                check_timestamps(conn, table, schema)
        except InvalidTable:
            # This doesn't use conn.rollback because we can't update the sqlalchemy metadata
            # for this view inside the connection.
            self.drop_view(schema.name)
            raise

    def delete_rows(
        self,
        name: str,
        time_array_id_values: dict[str, Any],
        connection: Optional[Connection] = None,
    ) -> None:
        """Delete all rows matching the time_array_id_values.

        Parameters
        ----------
        name
            Name of table
        time_array_id_values
            Values for the time_array_id_values. Keys must match the columns in the schema.
        connnection
            Optional connection to the database. Refer :meth:`ingest_table` for notes.

        Examples
        --------
        >>> store.delete_rows("devices", {"id": 47})
        """
        # TODO: consider supporting a user-defined query. Would need to check consistency
        # afterwards.
        # The current approach doesn't need to check because only one single complete time
        # array can be deleted on each call.
        table = self.get_table(name)
        if not time_array_id_values:
            msg = "time_array_id_values cannot be empty"
            raise InvalidParameter(msg)

        schema = self._schema_mgr.get_schema(name)
        if sorted(time_array_id_values.keys()) != sorted(schema.time_array_id_columns):
            msg = (
                "The keys of time_array_id_values must match the schema columns. "
                f"Passed = {sorted(time_array_id_values.keys())} "
                f"Schema = {sorted(schema.time_array_id_columns)}"
            )
            raise InvalidParameter(msg)

        assert time_array_id_values
        stmt = delete(table)
        for column, value in time_array_id_values.items():
            stmt = stmt.where(table.c[column] == value)

        if connection is None:
            with self._engine.connect() as conn:
                conn.execute(stmt)
                conn.commit()
        else:
            connection.execute(stmt)
            # Let the caller commit or rollback when ready.

        logger.info(
            "Delete all rows from table {} with time_array_id_values {}",
            name,
            time_array_id_values,
        )

        stmt2 = select(table).limit(1)
        is_empty = False
        if connection is None:
            with self._engine.connect() as conn:
                res = conn.execute(stmt2).fetchall()
        else:
            res = connection.execute(stmt2).fetchall()

        if not res:
            is_empty = True

        if is_empty:
            logger.info("Delete empty table {}", name)
            self.drop_table(name)

    def drop_table(self, name: str) -> None:
        """Drop a table from the database."""
        self._drop_table_or_view(name, "TABLE")

    def create_view(self, schema: TableSchema, stmt: Selectable) -> None:
        """Create a view in the database."""
        create_view(schema.name, stmt, self._engine, self._metadata)
        with self._engine.connect() as conn:
            self._schema_mgr.add_schema(conn, schema)
            conn.commit()
            self._schema_mgr.update_cache(schema)

    def drop_view(self, name: str) -> None:
        """Drop a view from the database."""
        self._drop_table_or_view(name, "VIEW")

    def _drop_table_or_view(self, name: str, tbl_type: str) -> None:
        table = self.get_table(name)
        with self._engine.connect() as conn:
            conn.execute(text(f"DROP {tbl_type} {name}"))
            self._schema_mgr.remove_schema(conn, name)
            conn.commit()

        self._metadata.remove(table)
        logger.info("Dropped {} {}", tbl_type.lower(), name)


def check_columns(table_columns: Iterable[str], schema_columns: Iterable[str]) -> None:
    """Check if the columns match the schema.

    Raises
    ------
    InvalidTable
        Raised if the columns don't match the schema.
    """
    expected_columns = schema_columns if isinstance(schema_columns, set) else set(schema_columns)
    diff = expected_columns.difference(table_columns)
    if diff:
        cols = " ".join(sorted(diff))
        msg = f"These columns are defined in the schema but not present in the table: {cols}"
        raise InvalidTable(msg)


def check_schema_compatibility(src: TableSchemaBase, dst: TableSchemaBase) -> None:
    """Check that a table with src schema can be converted to dst."""
