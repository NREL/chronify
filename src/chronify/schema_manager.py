import json

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
        self._cache: dict[str, TableSchema] = {}

        if self.SCHEMAS_TABLE in self._metadata.tables:
            logger.info("Loaded existing database {}", self._engine.url.database)
            self.rebuild_cache()
        else:
            table = Table(
                self.SCHEMAS_TABLE,
                self._metadata,
                Column("name", String),
                Column("schema", String),  # schema encoded as JSON
            )
            # TODO: will this work for Spark? Needs to be mutable.
            self._metadata.create_all(self._engine, tables=[table])
            logger.info("Initialized new database: {}", self._engine.url.database)

    def _get_schema_table(self) -> Table:
        return Table(self.SCHEMAS_TABLE, self._metadata)

    def add_schema(self, conn: Connection, schema: TableSchema) -> Table:
        """Add the schema to the store."""
        table = self._get_schema_table()
        stmt = insert(table).values(name=schema.name, schema=schema.model_dump_json())
        conn.execute(stmt)
        logger.debug("Added schema for table {}", schema.name)
        # Note: Don't add to the schema cache here. That should only happen after a
        # database commit.
        return table

    def get_schema(self, name: str) -> TableSchema:
        """Retrieve the schema for the table with name."""
        schema = self._cache.get(name)
        if schema is None:
            self.rebuild_cache()

        schema = self._cache.get(name)
        if schema is None:
            msg = f"{name=}"
            raise TableNotStored(msg)

        return self._cache[name]

    def remove_schema(self, conn: Connection, name: str) -> None:
        table = self._get_schema_table()
        stmt = delete(table).where(table.c["name"] == name)
        conn.execute(stmt)

    def rebuild_cache(self) -> None:
        """Rebuild the cache of schemas."""
        self._cache.clear()
        table = self._get_schema_table()
        with self._engine.connect() as conn:
            stmt = select(table)
            res = conn.execute(stmt).fetchall()
            for name, text in res:
                schema = TableSchema(**json.loads(text))
                assert name == schema.name
                assert name not in self._cache
                self._cache[name] = schema

    def update_cache(self, schema: TableSchema) -> None:
        # Allow overwrites.
        self._cache[schema.name] = schema
