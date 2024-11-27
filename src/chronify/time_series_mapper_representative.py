from sqlalchemy import Engine, MetaData, Table, select, text

import pandas as pd

from chronify.sqlalchemy.functions import read_database, write_database
from chronify.models import TableSchema
from chronify.exceptions import MissingParameter, ConflictingInputsError, TableAlreadyExists
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time_series_mapper_base import TimeSeriesMapperBase
from chronify.utils.sqlalchemy_table import create_table
from chronify.representative_time_range_generator import RepresentativePeriodTimeGenerator
from chronify.time_series_checker import check_timestamps


class MapperRepresentativeTimeToDatetime(TimeSeriesMapperBase):
    def __init__(
        self, engine: Engine, metadata: MetaData, from_schema: TableSchema, to_schema: TableSchema
    ) -> None:
        self._engine = engine
        self._metadata = metadata
        self._from_schema = from_schema
        self._to_schema = to_schema
        self._generator = RepresentativePeriodTimeGenerator(from_schema.time_config)

    def check_schema_consistency(self) -> None:
        """Check that from_schema can produce to_schema."""
        self._check_table_column_producibility()
        self._check_schema_measurement_type_consistency()

    def _check_table_column_producibility(self) -> None:
        """Check columns in destination table can be produced by source table."""
        available_cols = self._from_schema.list_columns() + [
            self._to_schema.time_config.time_column
        ]
        final_cols = self._to_schema.list_columns()
        if diff := set(final_cols) - set(available_cols):
            msg = f"Source table {self._from_schema.name} cannot produce the columns: {diff}"
            raise ConflictingInputsError(msg)

    def _check_schema_measurement_type_consistency(self) -> None:
        """Check that measurement_type is the same between schema."""
        from_mt = self._from_schema.time_config.measurement_type
        to_mt = self._to_schema.time_config.measurement_type
        if from_mt != to_mt:
            msg = f"Inconsistent measurement_types {from_mt=} vs. {to_mt=}"
            raise ConflictingInputsError(msg)

    def _check_source_table_has_time_zone(self) -> None:
        """Check source table has time_zone column."""
        if "time_zone" not in self._from_schema.list_columns():
            msg = f"time_zone is required for tz-aware representative time mapping and it is missing from source table: {self._from_schema.name}"
            raise MissingParameter(msg)

    def map_time(self) -> None:
        """Convert time columns with from_schema to to_schema configuration."""

        is_tz_naive = self._to_schema.time_config.is_time_zone_naive()
        self.check_schema_consistency()
        if not is_tz_naive:
            self._check_source_table_has_time_zone()
        # TODO: add interval type handling (note annual has no interval type)

        map_table_name = "map_table"
        dfm = self._create_mapping(is_tz_naive)
        if map_table_name in self._metadata.tables:
            msg = (
                f"table {map_table_name} already exists, delete it or use a different table name."
            )
            raise TableAlreadyExists(msg)
        try:
            with self._engine.connect() as conn:
                write_database(dfm, conn, map_table_name, self._to_schema.time_config)
                conn.commit()
            self._metadata.reflect(self._engine, views=True)
            self._apply_mapping(map_table_name)

            mapped_table = Table(self._to_schema.name, self._metadata)
            with self._engine.connect() as conn:
                try:
                    check_timestamps(conn, mapped_table, self._to_schema)
                except Exception as e:
                    print(e)
                    print(f"check_timestamps failed, dropping mapped table {self._to_schema.name}")
                    conn.execute(text(f"DROP TABLE {self._to_schema.name}"))
        finally:
            with self._engine.connect() as conn:
                conn.execute(text(f"DROP TABLE {map_table_name}"))

    def _create_mapping(self, is_tz_naive) -> pd.DataFrame:
        """Create mapping dataframe"""
        timestamp_generator = make_time_range_generator(self._to_schema.time_config)

        to_time_col = self._to_schema.time_config.time_column
        dft = pd.Series(timestamp_generator.list_timestamps()).rename(to_time_col).to_frame()
        if is_tz_naive:
            dfm = self._generator.create_tz_naive_mapping_dataframe(dft, to_time_col)
        else:
            with self._engine.connect() as conn:
                table = Table(self._from_schema.name, self._metadata)
                stmt = (
                    select(table.c["time_zone"])
                    .distinct()
                    .where(table.c["time_zone"].is_not(None))
                )
                time_zones = read_database(stmt, conn, self._from_schema.time_config)[
                    "time_zone"
                ].to_list()
            dfm = self._generator.create_tz_aware_mapping_dataframe(dft, to_time_col, time_zones)
        return dfm

    def _apply_mapping(self, map_table_name: str):
        """Apply mapping to create result as a table according to_schema"""
        left_table = Table(self._from_schema.name, self._metadata)
        right_table = Table(map_table_name, self._metadata)
        left_table_columns = [x.name for x in left_table.columns]
        right_table_columns = [x.name for x in right_table.columns]

        final_cols = set(self._to_schema.list_columns())
        left_cols = set(left_table_columns).intersection(final_cols)
        right_cols = final_cols - left_cols

        select_stmt = [left_table.c[x] for x in left_cols]
        select_stmt += [right_table.c[x] for x in right_cols]

        keys = self._from_schema.time_config.list_time_columns()
        if not self._to_schema.time_config.is_time_zone_naive():
            keys.append("time_zone")
            assert (
                "time_zone" in left_table_columns
            ), f"time_zone not in table={self._from_schema.name}"
            assert "time_zone" in right_table_columns, f"time_zone not in table={map_table_name}"
        on_stmt = ()
        for i, x in enumerate(keys):
            if i == 0:
                on_stmt = left_table.c[x] == right_table.c[x]
            else:
                on_stmt &= left_table.c[x] == right_table.c[x]
        query = select(*select_stmt).select_from(left_table).join(right_table, on_stmt)
        create_table(self._to_schema.name, query, self._engine, self._metadata)
        self._metadata.reflect(self._engine, views=True)
