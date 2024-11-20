from sqlalchemy import Engine, MetaData, Table, select
import time

import pandas as pd

from chronify.sqlalchemy.functions import read_database, write_database
from chronify.models import TableSchema
from chronify.exceptions import MissingParameter, ConflictingInputsError
from chronify.time_series_mapper_base import TimeSeriesMapperBase
from chronify.time import (
    representative_weekday_column,
)
from chronify.utils.sqlalchemy_view import create_view


class MapperRepresentativeTimeToDatetime(TimeSeriesMapperBase):
    def __init__(
        self, engine: Engine, metadata: MetaData, from_schema: TableSchema, to_schema: TableSchema
    ) -> None:
        self._engine = engine
        self._metadata = metadata
        self.from_schema = from_schema
        self.to_schema = to_schema
        self.weekday_column = representative_weekday_column[from_schema.time_config.time_format]

    def check_schema_consistency(self) -> None:
        self._check_table_column_producibility()
        self._check_schema_measurement_type_consistency()

    def _check_table_column_producibility(self) -> None:
        available_cols = self.from_schema.list_columns() + [self.to_schema.time_config.time_column]
        final_cols = self.to_schema.list_columns()
        if diff := set(final_cols) - set(available_cols):
            msg = f"Source table {self.from_schema.time_config.name} cannot produce the columns: {diff}"
            raise ConflictingInputsError(msg)

    def _check_schema_measurement_type_consistency(self) -> None:
        from_mt = self.from_schema.time_config.measurement_type
        to_mt = self.from_schema.time_config.measurement_type
        if from_mt != to_mt:
            msg = f"Inconsistent measurement_types {from_mt=} vs. {to_mt=}"
            raise ConflictingInputsError(msg)

    def _check_source_table_has_time_zone(self) -> None:
        if "time_zone" not in self.from_schema.time_array_id_columns:
            msg = f"time_zone is required for representative time mapping and it is missing from source table: {self.from_schema.time_config.name}"
            raise MissingParameter(msg)

    def map_time(self) -> None:
        # Destination time
        to_time_col = self.to_schema.time_config.time_column
        dft = (
            pd.Series(self.to_schema.time_config.list_timestamps()).rename(to_time_col).to_frame()
        )

        # Apply checks
        self.check_schema_consistency()
        if not self.to_schema.time_config.is_time_zone_naive():
            self._check_source_table_has_time_zone()
        # TODO: add interval type handling (note annual has no interval type)

        # Create mapping
        if self.to_schema.time_config.is_time_zone_naive():
            dfm = self._create_mapping_dataframe_tz_naive(dft, to_time_col)
        else:
            dfm = self._create_mapping_dataframe_tz_aware(dft, to_time_col)

        # Ingest mapping into db
        time_array_id_columns = [
            x for x in self.from_schema.time_config.list_time_columns() if x != "hour"
        ]
        if not self.to_schema.time_config.is_time_zone_naive():
            time_array_id_columns += ["time_zone"]

        map_table_schema = TableSchema(
            name="map_table" + str(int(time.time())),
            time_config=self.to_schema.time_config,
            time_array_id_columns=time_array_id_columns,
            value_column="hour",  # this is a workaround
        )
        with self._engine.connect() as conn:
            write_database(dfm, conn, map_table_schema, if_table_exists="replace")
            conn.commit()
        self._metadata.reflect(self._engine, views=True)

        # Apply mapping and downselect to final cols
        self._apply_mapping(map_table_schema)

        # TODO write output to parquet?

    def _create_mapping_dataframe_tz_naive(
        self, dft: pd.DataFrame, to_time_col: str
    ) -> pd.DataFrame:
        """Create tz-naive time mapping dataframe"""
        dfm = dft.copy()
        dfm["month"] = dfm[to_time_col].dt.month
        dow = dfm[to_time_col].dt.day_of_week
        if self.weekday_column == "day_of_week":
            dfm[self.weekday_column] = dow
        elif self.weekday_column == "is_weekday":
            dfm[self.weekday_column] = False
            dfm.loc[dow < 5, self.weekday_column] = True
        else:
            msg = f"No representative time mapping support for time columns: {self.from_schema.time_config.list_time_columns()}"
            raise NotImplementedError(msg)

        dfm["hour"] = dfm[to_time_col].dt.hour
        return dfm

    def _create_mapping_dataframe_tz_aware(
        self, dft: pd.DataFrame, to_time_col: str
    ) -> pd.DataFrame:
        """Create tz-aware time mapping dataframe according to to_schema.time_config"""
        with self._engine.connect() as conn:
            table = Table(self.from_schema.name, self._metadata)
            stmt = select(table.c["time_zone"]).distinct().where(table.c["time_zone"].is_not(None))
            df_tz = read_database(stmt, conn, self.from_schema)

        dfm = []
        for idx, row in df_tz.iterrows():
            dfgt = dft.copy()
            dfgt["timestamp_tmp"] = dfgt[to_time_col].dt.tz_convert(row["time_zone"])
            dfgt["month"] = dfgt["timestamp_tmp"].dt.month

            dow = dfgt["timestamp_tmp"].dt.day_of_week
            if self.weekday_column == "day_of_week":
                dfgt[self.weekday_column] = dow
            elif self.weekday_column == "is_weekday":
                dfgt[self.weekday_column] = False
                dfgt.loc[dow < 5, self.weekday_column] = True
            else:
                msg = f"No representative time mapping support for time columns: {self.from_schema.time_config.list_time_columns()}"
                raise NotImplementedError(msg)

            dfgt["hour"] = dfgt["timestamp_tmp"].dt.hour
            dfgt["time_zone"] = row["time_zone"]
            dfm.append(dfgt.drop(columns=["timestamp_tmp"]))
        dfm = pd.concat(dfm, axis=0, ignore_index=True)
        return dfm

    def _apply_mapping(self, map_table_schema: TableSchema):
        """Apply mapping to create result as a view according to_schema"""
        left_table = Table(self.from_schema.name, self._metadata)
        right_table = Table(map_table_schema.name, self._metadata)
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
            assert (
                "time_zone" in right_table_columns
            ), f"time_zone not in table={map_table_schema.name}"
        on_stmt = ()
        for i, x in enumerate(keys):
            if i == 0:
                on_stmt = left_table.c[x] == right_table.c[x]
            else:
                on_stmt &= left_table.c[x] == right_table.c[x]
        query = select(*select_stmt).select_from(left_table).join(right_table, on_stmt)
        create_view(self.to_schema.name, query, self._engine, self._metadata)
