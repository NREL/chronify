from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text, Engine, MetaData, Table

from chronify.exceptions import InvalidTable
from chronify.models import TableSchema
from chronify.time_series_checker import TimeSeriesChecker

g_metadata = MetaData()


class Store:
    """Data store for time series data"""

    def __init__(self, engine: Optional[Engine] = None, **connect_args) -> None:
        """Construct the Store.

        Parameters
        ----------
        engine: sqlalchemy.Engine
            Optional, defaults to a DuckDB engine.
        """
        if engine is None:
            self._engine = create_engine("duckdb:///:memory:", **connect_args)
        else:
            self._engine = engine

    # def add_time_series(self, data: np.ndarray) -> None:
    # """Add a time series array to the store."""

    def create_view_from_parquet(self, name: str, path: Path) -> None:
        """Create a view in the database from a Parquet file."""
        with self._engine.begin() as conn:
            query = f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}/**/*.parquet')"
            conn.execute(text(query))

    def export_csv(self, table: str, path: Path) -> None:
        """Export a table or view to a CSV file."""

    def export_parquet(self, table: str, path: Path) -> None:
        """Export a table or view to a Parquet file."""

    def load_table(self, path: Path, schema: TableSchema) -> None:
        """Load a table into the database."""
        # TODO: support unpivoting a pivoted table
        if path.suffix != ".parquet":
            msg = "Only Parquet files are currently supported: {path=}"
            raise NotImplementedError(msg)

        self.create_view_from_parquet(schema.name, path)
        self.update_table_schema()
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
        g_metadata.reflect(self._engine, views=True, autoload_with=self._engine)

    def _check_table_schema(self, schema: TableSchema) -> None:
        table = Table(schema.name, g_metadata)
        expected_columns = set(
            schema.time_array_id_columns + schema.time_config.time_columns + schema.value_columns
        )
        existing_columns = {x.name for x in table.columns}
        diff = expected_columns - existing_columns
        if diff:
            cols = " ".join(sorted(diff))
            msg = f"These columns are defined in the schema but not present in the table: {cols}"
            raise InvalidTable(msg)

    def _check_timestamps(self, schema: TableSchema) -> None:
        checker = TimeSeriesChecker(self._engine)
        checker.check_timestamps(schema)
