import json
from typing import Optional

from loguru import logger
from sqlalchemy import (
    Column,
    Connection,
    Engine,
    MetaData,
    String,
    Table,
    delete,
    insert,
    select,
    text,
)

from chronify.exceptions import (
    TableNotStored,
)
from chronify.models import (
    TableSchema,
)


class SchemaManager:
    """Manages schemas for the Store. Provides a cache to avoid repeated database reads."""

    SCHEMAS_TABLE = "schemas"

    def __init__(self, engine: Engine, metadata: MetaData):
        self._engine = engine
        self._metadata = metadata
        # Caching is not necessary if using SQLite, which provides very fast performance (~1 us)
        # for checking schemas in the **tiny** schemas table.
        # The same lookups in DuckDB are taking over 100 us.
        self._cache: dict[str, TableSchema] = {}

        if self.SCHEMAS_TABLE in self._metadata.tables:
            logger.info("Loaded existing database {}", self._engine.url.database)
            self.rebuild_cache()
        else:
            if self._engine.name == "hive":
                # metadata.create_all doesn't work here.
                with self._engine.begin() as conn:
                    conn.execute(text(f"DROP TABLE IF EXISTS {self.SCHEMAS_TABLE}"))
                    conn.execute(
                        text(f"CREATE TABLE {self.SCHEMAS_TABLE}(name STRING, schema STRING)")
                    )
                self._metadata.reflect(self._engine)
            else:
                table = Table(
                    self.SCHEMAS_TABLE,
                    self._metadata,
                    Column("name", String, nullable=False, unique=True),
                    Column("schema", String),  # schema encoded as JSON
                )
                self._metadata.create_all(self._engine, tables=[table])
            logger.info("Initialized new database: {}", self._engine.url.database)

    def _get_schema_table(self) -> Table:
        return (
            self._metadata.tables[self.SCHEMAS_TABLE]
            if self._engine.name == "hive"
            else Table(self.SCHEMAS_TABLE, self._metadata)
        )

    def add_schema(self, conn: Connection, schema: TableSchema) -> Table:
        """Add the schema to the store."""
        table = self._get_schema_table()
        stmt = insert(table).values(name=schema.name, schema=schema.model_dump_json())
        conn.execute(stmt)
        # If there is a rollback after this addition to cached, things _should_ still be OK.
        # The table will be deleted and any attempted reads will fail with an error.
        # There will be a stale entry in cache, but it will be overwritten if the user ever
        # adds a new table with the same name.
        self._cache[schema.name] = schema
        logger.trace("Added schema for table {}", schema.name)
        return table

    def get_schema(self, name: str, conn: Optional[Connection] = None) -> TableSchema:
        """Retrieve the schema for the table with name."""
        schema = self._cache.get(name)
        if schema is None:
            self.rebuild_cache(conn=conn)

        schema = self._cache.get(name)
        if schema is None:
            msg = f"{name=}"
            raise TableNotStored(msg)

        return self._cache[name]

    def remove_schema(self, conn: Connection, name: str) -> None:
        """Remove the schema from the store."""
        table = self._get_schema_table()
        if self._engine.name == "hive":
            # Hive/Spark doesn't support delete, so we have to re-create the table without
            # this one entry
            stmt = select(table).where(table.c.name != name)
            rows = conn.execute(stmt).fetchall()
            conn.execute(text(f"DROP TABLE {self.SCHEMAS_TABLE}"))
            conn.execute(text(f"CREATE TABLE {self.SCHEMAS_TABLE}(name STRING, schema STRING)"))
            for row in rows:
                params = {"name": row[0], "schema": row[1]}
                conn.execute(
                    text(f"INSERT INTO {self.SCHEMAS_TABLE} VALUES(:name, :schema)"),
                    params,
                )
        else:
            stmt2 = delete(table).where(table.c["name"] == name)
            conn.execute(stmt2)

        self._cache.pop(name)

    def rebuild_cache(self, conn: Optional[Connection] = None) -> None:
        """Rebuild the cache of schemas."""
        self._cache.clear()
        if conn is None:
            with self._engine.connect() as conn:
                self._rebuild_cache(conn)
        else:
            self._rebuild_cache(conn)

    def _rebuild_cache(self, conn: Connection) -> None:
        table = self._get_schema_table()
        stmt = select(table)
        res = conn.execute(stmt).fetchall()
        for name, json_text in res:
            schema = TableSchema(**json.loads(json_text))
            assert name == schema.name
            assert name not in self._cache
            self._cache[name] = schema
