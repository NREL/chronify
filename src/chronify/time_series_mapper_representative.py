from sqlalchemy import Engine, MetaData, Table, select

import pandas as pd

from chronify.sqlalchemy.functions import read_database, write_database
from chronify.models import TableSchema
from chronify.exceptions import MissingParameter, ConflictingInputsError
from chronify.time_series_mapper_base import TimeSeriesMapperBase
from chronify.utils.sqlalchemy_view import create_view
from chronify.time_models import RepresentativePeriodTimeGenerator


class MapperRepresentativeTimeToDatetime(TimeSeriesMapperBase):
    def __init__(
        self, engine: Engine, metadata: MetaData, from_schema: TableSchema, to_schema: TableSchema
    ) -> None:
        self._engine = engine
        self._metadata = metadata
        self.from_schema = from_schema
        self.to_schema = to_schema
        self._generator = RepresentativePeriodTimeGenerator(from_schema.time_config)

    def check_schema_consistency(self) -> None:
        """Check that from_schema can produce to_schema."""
        self._check_table_column_producibility()
        self._check_schema_measurement_type_consistency()

    def _check_table_column_producibility(self) -> None:
        """Check columns in destination table can be produced by source table."""
        available_cols = self.from_schema.list_columns() + [self.to_schema.time_config.time_column]
        final_cols = self.to_schema.list_columns()
        if diff := set(final_cols) - set(available_cols):
            msg = f"Source table {self.from_schema.time_config.name} cannot produce the columns: {diff}"
            raise ConflictingInputsError(msg)

    def _check_schema_measurement_type_consistency(self) -> None:
        """Check that measurement_type is the same between schema."""
        from_mt = self.from_schema.time_config.measurement_type
        to_mt = self.from_schema.time_config.measurement_type
        if from_mt != to_mt:
            msg = f"Inconsistent measurement_types {from_mt=} vs. {to_mt=}"
            raise ConflictingInputsError(msg)

    def _check_source_table_has_time_zone(self) -> None:
        """Check source table has time_zone column."""
        if "time_zone" not in self.from_schema.time_array_id_columns:
            msg = f"time_zone is required for tz-aware representative time mapping and it is missing from source table: {self.from_schema.name}"
            raise MissingParameter(msg)

    def map_time(self) -> None:
        """Convert time columns with from_schema to to_schema configuration."""

        is_tz_naive = self.to_schema.time_config.is_time_zone_naive()
        self.check_schema_consistency()
        if not is_tz_naive:
            self._check_source_table_has_time_zone()
        # TODO: add interval type handling (note annual has no interval type)

        to_time_col = self.to_schema.time_config.time_column
        dft = (
            pd.Series(self.to_schema.time_config.list_timestamps()).rename(to_time_col).to_frame()
        )
        if is_tz_naive:
            dfm = self._generator._handler.add_time_attribute_columns(dft, to_time_col)
        else:
            with self._engine.connect() as conn:
                table = Table(self.from_schema.name, self._metadata)
                stmt = (
                    select(table.c["time_zone"])
                    .distinct()
                    .where(table.c["time_zone"].is_not(None))
                )
                time_zones = read_database(stmt, conn, self.from_schema.time_config)[
                    "time_zone"
                ].to_list()
            dfm = self._generator._handler.create_tz_aware_mapping_dataframe(
                dft, to_time_col, time_zones
            )

        # Ingest mapping into db
        time_array_id_columns = [
            x for x in self.from_schema.time_config.list_time_columns() if x != "hour"
        ]
        if not is_tz_naive:
            time_array_id_columns += ["time_zone"]

        map_table_name = "map_table"
        with self._engine.connect() as conn:
            write_database(
                dfm, conn, map_table_name, self.to_schema.time_config, if_table_exists="replace"
            )
            conn.commit()
        self._metadata.reflect(self._engine, views=True)

        self._apply_mapping(map_table_name)
        # TODO write output to parquet?

    def _apply_mapping(self, map_table_name: str):
        """Apply mapping to create result as a view according to_schema"""
        left_table = Table(self.from_schema.name, self._metadata)
        right_table = Table(map_table_name, self._metadata)
        left_table_columns = [x.name for x in left_table.columns]
        right_table_columns = [x.name for x in right_table.columns]

        final_cols = self.to_schema.list_columns()
        left_cols = [x for x in left_table_columns if x in final_cols]
        right_cols = [x for x in right_table_columns if x in final_cols and x not in left_cols]
        assert set(left_cols + right_cols) == set(
            final_cols
        ), f"table join does not produce the {final_cols=}"

        select_stmt = [left_table.c[x] for x in left_cols]
        select_stmt += [right_table.c[x] for x in right_cols]

        keys = (
            self.from_schema.time_config.list_time_columns().copy()
        )  # TODO copy is required here not sure why
        if not self.to_schema.time_config.is_time_zone_naive():
            keys += ["time_zone"]
            assert (
                "time_zone" in left_table_columns
            ), f"time_zone not in table={self.from_schema.name}"
            assert "time_zone" in right_table_columns, f"time_zone not in table={map_table_name}"
        on_stmt = ()
        for i, x in enumerate(keys):
            if i == 0:
                on_stmt = left_table.c[x] == right_table.c[x]
            else:
                on_stmt &= left_table.c[x] == right_table.c[x]
        query = select(*select_stmt).select_from(left_table).join(right_table, on_stmt)
        create_view(self.to_schema.name, query, self._engine, self._metadata)
