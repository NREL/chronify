from sqlalchemy import Engine, MetaData, Table, select

import pandas as pd
import polars as pl

from chronify.sqlalchemy.functions import read_database
from chronify.models import TableSchema
from chronify.exceptions import (
    InvalidParameter,
    MissingParameter,
    InvalidTable,
)
from chronify.time_configs import DatetimeRange, RepresentativePeriodTimeRange


class TimeSeriesMapper:
    """Maps time series data from one configuration to another."""

    def __init__(self, engine: Engine, metadata: MetaData) -> None:
        self._engine = engine
        self._metadata = metadata

    def map_time_series(
        self,
        from_schema: TableSchema,
        to_schema: TableSchema,
    ) -> None:
        pass

    def map_representative_time(
        self,
        from_schema: TableSchema,
        to_schema: TableSchema,
    ):
        if isinstance(from_schema.time_config) != RepresentativePeriodTimeRange:
            msg = f"{from_schema=} needs to be RepresentativePeriodTimeRange"
            raise InvalidParameter(msg)

        if isinstance(to_schema.time_config) != DatetimeRange:
            msg = f"{to_schema=} needs to be DatetimeRange"
            raise InvalidParameter(msg)

        # Destination time
        to_time_col = to_schema.time_config.time_column
        dft = pd.Series(to_schema.time_config.list_timestamps()).rename(to_time_col).to_frame()

        # Check source table has the right data
        # [1] from_schema can produce to_schema
        available_cols = from_schema.list_columns + [to_time_col]
        final_cols = to_schema.list_columns
        if diff := final_cols - available_cols:
            msg = f"source table {from_schema.time_config.name} cannot produce the destination table columns {diff}"
            raise InvalidTable(msg)

        # [2] src_table has time_zone
        if "time_zone" not in from_schema.time_config.time_array_id_columns:
            msg = f"time_zone is required for representative time mapping and it is missing from source table: {from_schema.time_config.name}"
            raise MissingParameter(msg)  # TODO does this belong in checker?

        # [3] src_table weekday column has the expected records
        from_time_cols = from_schema.time_config.time_columns
        week_col = [x for x in from_time_cols if x not in ["month", "hour"]]
        assert len(week_col) == 1, f"Unexpected {week_col=}"
        week_col = week_col[0]

        with self._engine.connect() as conn:
            table = Table(from_schema.name, self._metadata)
            stmt = select(table.c["time_zone"]).distinct().where(table.c["time_zone"].is_not(None))
            df_tz = read_database(stmt, conn)

            stmt2 = select(table.c[week_col]).distinct().where(table.c[week_col].is_not(None))
            week_records = read_database(stmt2, conn).to_list()

        self.check_week_col_data(week_col, week_records, from_schema.time_config.name)

        # Create mapping and ingest into db
        dfm = []
        for idx, row in df_tz.iterrows():
            dfgt = dft.copy()
            dfgt["timestamp_tmp"] = dfgt[to_time_col].dt.tz_convert(row["timezone"])
            dfgt["month"] = dfgt["timestamp_tmp"].dt.month

            dow = dfgt["timestamp_tmp"].dt.day_of_week
            if week_col == "day_of_week":
                dfgt["day_of_week"] = dow
            elif week_col == "is_weekday":
                dfgt["is_weekday"] = False
                dfgt.loc[dow < 5, "is_weekday"] = True  # TODO do these need to be in str format?
            else:
                msg = f"No representative time mapping support for time columns: {from_time_cols}"
                raise NotImplementedError(msg)

            dfgt["hour"] = dfgt["timestamp_tmp"].dt.hour
            dfgt["timezone"] = row["timezone"]
            dfm.append(dfgt.drop(columns=["timestamp_tmp"]))
        dfm = pd.concat(dfm, axis=0, ignore_index=True)

        with self._engine.connect() as conn:
            pl.DataFrame(dfm).write_database(
                "map_table", connection=conn, if_table_exists="append"
            )
            conn.commit()
        self.update_table_schema()

        # Apply mapping and downselect to final cols
        keys = from_time_cols + ["timezone"]
        with self._engine.connect() as conn:
            left_table = Table(from_schema.name, self._metadata)
            right_table = Table("map_table", self._metadata)
            left_cols = [x for x in left_table.columns if x in final_cols]
            right_cols = [x for x in right_table.columns if x in final_cols and x not in left_cols]
            assert set(left_cols + right_cols) == set(
                final_cols
            ), f"table join does not produce the {final_cols=}"
            select_stmt = [left_table.c[x] for x in left_cols]
            select_stmt += [right_table.c[x] for x in right_cols]
            on_stmt = [left_table.c[x] == right_table.c[x] for x in keys]
            stmt = select(*select_stmt).where(*on_stmt)
            df = read_database(stmt, conn, to_schema)  # TODO create as new db table?

        return df

    @staticmethod
    def check_week_col_data(week_col, week_records, table_name):
        msg = f"Unexpected values in column: {week_col} of source table: {table_name}"
        if week_col == "day_of_week":
            if set(week_records) != set(range(7)):
                msg2 = msg + f"\n{week_records}"
                raise InvalidTable(msg2)
        elif week_col == "is_weekday":
            if set(week_records) != {True, False}:
                msg2 = msg + f"\n{week_records}"
                raise InvalidTable(msg2)  # TODO does this belong in checker?
        else:
            pass
