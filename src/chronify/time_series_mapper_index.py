
import logging
from functools import reduce
from operator import and_

import pandas as pd
from sqlalchemy import Engine, MetaData, Table, select, text

from chronify.sqlalchemy.functions import read_database, write_database
from chronify.models import TableSchema
from chronify.exceptions import (
    TableAlreadyExists,
    InvalidParameter,
)
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time_series_mapper_base import TimeSeriesMapperBase, CheckSchemaMixins
from chronify.utils.sqlalchemy_table import create_table
from chronify.index_time_range_generator import IndexTimeRangeGenerator
from chronify.time_series_checker import check_timestamps
from chronify.time_configs import DatetimeRange, IndexTimeRange


logger = logging.getLogger(__name__)


class MapperIndexTimeToDatetime(TimeSeriesMapperBase, CheckSchemaMixins):
    def __init__(
        self, engine: Engine, metadata: MetaData, from_schema: TableSchema, to_schema: TableSchema
    ) -> None:
        if not isinstance(from_schema.time_config, IndexTimeRange):
            msg = "source schema does not have IndexTimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        if not isinstance(to_schema.time_config, DatetimeRange):
            msg = "destination schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        self._from_time_config: IndexTimeRange = from_schema.time_config
        self._to_time_config: DatetimeRange = to_schema.time_config
        self._from_schema = from_schema
        self._to_schema = to_schema
        self._engine = engine
        self._metadata = metadata
        self._generator = IndexTimeRangeGenerator(self._from_time_config)

    def map_time(self) -> None:
        """Convert time columns with from_schema to to_schema configuration."""
        is_tz_naive = self._to_time_config.is_time_zone_naive()
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
                write_database(
                    dfm, conn, map_table_name, self._to_schema.time_config, if_table_exists="fail"
                )
                self._metadata.reflect(self._engine, views=True)
                self._apply_mapping(map_table_name)
                mapped_table = Table(self._to_schema.name, self._metadata)
                try:
                    check_timestamps(conn, mapped_table, self._to_schema)
                except Exception:
                    msg = f"check_timestamps failed on mapped table {self._to_schema.name}. Drop it"
                    logger.exception(msg)
                    conn.rollback()
                    raise
                conn.commit()
        finally:
            if map_table_name in self._metadata.tables:
                with self._engine.connect() as conn:
                    conn.execute(text(f"DROP TABLE {map_table_name}"))
                    conn.commit()
                self._metadata.remove(Table(map_table_name, self._metadata))

    def create_tz_naive_mapping_dataframe(self, dft, to_time_col):
        # dft needs to be sorted and unique? does resolution matter here? 
        # using the from schema, the resolution can be used 
        # only mapping to datetime
        dft[self._from_time_config.time_column] = round((dft[to_time_col] - self._from_time_config.start_timestamp) / self._from_time_config.resolution).astype(int)
        return dft

    def create_tz_aware_mapping_dataframe(self, dft, to_time_col, time_zones: list[str]):
        dfm = []
        for tz in time_zones:
            dfc = dft.copy()
            dft["timestamp_tmp"] = dfc[to_time_col].dt.tz_convert(tz)
            dfc = self.create_tz_naive_mapping_dataframe(dft, "timestamp_tmp")
            dft["time_zone"] = tz
            dfm.append(dft.drop(columns=["timestamp_tmp"]))
        return pd.concat(dfm, ignore_index=True)

    def _create_mapping(self, is_tz_naive: bool) -> pd.DataFrame:
        """Create mapping dataframe"""
        timestamp_generator = make_time_range_generator(self._to_schema.time_config)

        to_time_col = self._to_time_config.time_column
        dft = pd.Series(timestamp_generator.list_timestamps()).rename(to_time_col).to_frame()
        if is_tz_naive:
            dfm = self.create_tz_naive_mapping_dataframe(dft, to_time_col)
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
            dfm = self.create_tz_aware_mapping_dataframe(dft, to_time_col, time_zones)
        return dfm

    def _apply_mapping(self, map_table_name: str) -> None:
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
        if not self._to_time_config.is_time_zone_naive():
            keys.append("time_zone")
            assert (
                "time_zone" in left_table_columns
            ), f"time_zone not in table={self._from_schema.name}"
            assert "time_zone" in right_table_columns, f"time_zone not in table={map_table_name}"

        on_stmt = reduce(and_, (left_table.c[x] == right_table.c[x] for x in keys))
        query = select(*select_stmt).select_from(left_table).join(right_table, on_stmt)
        create_table(self._to_schema.name, query, self._engine, self._metadata)
