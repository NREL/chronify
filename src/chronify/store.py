import itertools
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from sqlalchemy import Column, Engine, MetaData, Table, create_engine, text

from chronify.exceptions import InvalidTable
from chronify.csv_io import read_csv
from chronify.duckdb.functions import unpivot as duckdb_unpivot
from chronify.duckdb.functions import add_datetime_column
from chronify.models import (
    CsvTableSchema,
    TableSchema,
    TableSchemaBase,
    get_sqlalchemy_type_from_duckdb,
)
from chronify.time_configs import DatetimeRange, IndexTimeRange
from chronify.time_series_checker import TimeSeriesChecker


class Store:
    """Data store for time series data"""

    def __init__(self, engine: Optional[Engine] = None, **connect_kwargs) -> None:
        """Construct the Store.

        Parameters
        ----------
        engine: sqlalchemy.Engine
            Optional, defaults to a DuckDB engine.
        """
        self._metadata = MetaData()
        if engine is None:
            self._engine = create_engine("duckdb:///:memory:", **connect_kwargs)
        else:
            self._engine = engine

    # TODO
    # @classmethod
    # def load_spark(cls, spark_url: str, **connect_kwargs) -> Store:
    # engine = create_engine(spark_url, **connect_kwargs)
    # return cls(engine)

    # def add_time_series(self, data: np.ndarray) -> None:
    # """Add a time series array to the store."""

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

    # def export_csv(self, table: str, path: Path) -> None:
    #    """Export a table or view to a CSV file."""

    # def export_parquet(self, table: str, path: Path) -> None:
    #    """Export a table or view to a Parquet file."""

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
            rel = duckdb_unpivot(
                rel,
                src_schema.value_columns,
                src_schema.pivoted_dimension_name,
                dst_schema.value_column,
            )

        if isinstance(src_schema.time_config, IndexTimeRange):
            if isinstance(dst_schema.time_config, DatetimeRange):
                rel = add_datetime_column(
                    rel=rel,
                    start=dst_schema.time_config.start,
                    resolution=dst_schema.time_config.resolution,
                    length=dst_schema.time_config.length,
                    time_array_id_columns=src_schema.time_array_id_columns,
                    time_column=dst_schema.time_config.time_columns[0],
                    timestamps=list(src_schema.time_config.iter_timestamps()),
                )
            else:
                cls_name = dst_schema.time_config.__class__.__name__
                msg = f"IndexTimeRange cannot be converted to {cls_name}"
                raise NotImplementedError(msg)

        if self.has_table(dst_schema.name):
            table = Table(dst_schema.name, self._metadata)
        else:
            dtypes = [get_sqlalchemy_type_from_duckdb(x) for x in rel.dtypes]
            columns = [Column(x, y) for x, y in zip(rel.columns, dtypes)]
            table = Table(dst_schema.name, self._metadata, *columns)
            table.create(self._engine)

        values = rel.fetchall()
        columns = table.columns.keys()
        placeholder = ",".join(itertools.repeat("?", len(columns)))
        cols = ",".join(columns)
        with self._engine.begin() as conn:
            query = f"INSERT INTO {dst_schema.name} ({cols}) VALUES ({placeholder})"
            conn.exec_driver_sql(query, values)
            query = f"select * from {dst_schema.name}"
            conn.commit()

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
        checker = TimeSeriesChecker(self._engine)
        checker.check_timestamps(schema)


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
