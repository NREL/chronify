import abc
from zoneinfo import ZoneInfo
from sqlalchemy import Engine, MetaData, Table, select
from typing import Optional
from pathlib import Path
import pandas as pd

from chronify.models import TableSchema, MappingTableSchema
from chronify.time_configs import (
    DatetimeRangeBase,
    DatetimeRange,
    DatetimeRangeWithTZColumn,
    TimeBasedDataAdjustment,
)
from chronify.exceptions import InvalidParameter, MissingValue
from chronify.time_series_mapper_base import apply_mapping
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.sqlalchemy.functions import read_database
from chronify.time import TimeType
from chronify.time_utils import wrap_timestamps


def convert_time_zone(
    engine: Engine,
    metadata: MetaData,
    src_schema: TableSchema,
    to_time_zone: ZoneInfo | None,
    scratch_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    check_mapped_timestamps: bool = False,
) -> TableSchema:
    """Convert time zone of a table to a specified time zone.
    Output timestamp is tz-naive with a new time_zone column added.
    Parameters
        ----------
        engine
            sqlalchemy engine
        metadata
            sqlalchemy metadata
        src_schema
            Defines the source table in the database.
        to_time_zone
            time zone to convert to. If None, convert to tz-naive.
        scratch_dir
            Directory to use for temporary writes. Default to the system's tmp filesystem.
        output_file
            If set, write the mapped table to this Parquet file.
        check_mapped_timestamps
            Perform time checks on the result of the mapping operation. This can be slow and
            is not required.

        Returns
        -------
        dst_schema
            schema of output table with converted timestamps
    """
    TZC = TimeZoneConverter(engine, metadata, src_schema, to_time_zone)
    TZC.convert_time_zone(
        scratch_dir=scratch_dir,
        output_file=output_file,
        check_mapped_timestamps=check_mapped_timestamps,
    )

    return TZC._to_schema


def convert_time_zone_by_column(
    engine: Engine,
    metadata: MetaData,
    src_schema: TableSchema,
    time_zone_column: str,
    wrap_time_allowed: Optional[bool] = False,
    scratch_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    check_mapped_timestamps: bool = False,
) -> TableSchema:
    """Convert time zone of a table to multiple time zones specified by a column.
    Output timestamp is tz-naive, reflecting the local time relative to the time_zone_column.
    Parameters
        ----------
        engine
            sqlalchemy engine
        metadata
            sqlalchemy metadata
        srd_schema
            Defines the source table in the database.
        time_zone_column
            Column name in the source table that contains the time zone information.
        wrap_time_allowed
            If False, the converted timestamps will aligned with the original timestamps in real time scale
                E.g. 2018-01-01 00:00 ~ 2018-12-31 23:00 in US/Eastern becomes
                    2017-12-31 23:00 ~ 2018-12-31 22:00 in US/Central
            If True, the converted timestamps will fit into the time range of the src_schema in tz-naive clock time
                E.g. 2018-01-01 00:00 ~ 2018-12-31 23:00 in US/Eastern becomes
                    2017-12-31 23:00 ~ 2018-12-31 22:00 in US/Central, which is then wrapped such that
                    no clock time timestamps are in 2017. The final timestamps are:
                    2018-12-31 23:00, 2018-01-01 00:00 ~ 2018-12-31 22:00 in US/Central
        scratch_dir
            Directory to use for temporary writes. Default to the system's tmp filesystem.
        output_file
            If set, write the mapped table to this Parquet file.
        check_mapped_timestamps
            Perform time checks on the result of the mapping operation. This can be slow and
            is not required.

        Returns
        -------
        dst_schema
            schema of output table with converted timestamps
    """
    TZC = TimeZoneConverterByColumn(
        engine, metadata, src_schema, time_zone_column, wrap_time_allowed
    )
    TZC.convert_time_zone(
        scratch_dir=scratch_dir,
        output_file=output_file,
        check_mapped_timestamps=check_mapped_timestamps,
    )
    return TZC._to_schema


class TimeZoneConverterBase(abc.ABC):
    """Base class for time zone conversion of time series data."""

    def __init__(
        self,
        engine: Engine,
        metadata: MetaData,
        from_schema: TableSchema,
    ):
        self._engine = engine
        self._metadata = metadata
        self._from_schema = from_schema

    def check_from_schema(self) -> None:
        msg = ""
        if not isinstance(self._from_schema.time_config, DatetimeRange):
            msg += "Source schema does not have DatetimeRange time config. "
        if self._from_schema.time_config.start_time_is_tz_naive():
            msg += "Source schema start_time must be timezone-aware. "
            msg += "To convert from timezone-naive to timezone-aware, use the TimeSeriesMapperDatetime.map_time() method instead. "
        if msg != "":
            raise InvalidParameter(msg)

    @abc.abstractmethod
    def generate_to_schema(self) -> TableSchema:
        """Generate to_schema based on from_schema"""

    @abc.abstractmethod
    def convert_time_zone(
        self,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        """Convert time zone of the from_schema"""


class TimeZoneConverter(TimeZoneConverterBase):
    """Class for time zone conversion of time series data to a specified time zone."""

    def __init__(
        self,
        engine: Engine,
        metadata: MetaData,
        from_schema: TableSchema,
        to_time_zone: ZoneInfo | None,
    ):
        super().__init__(engine, metadata, from_schema)
        self._to_time_zone = to_time_zone
        self._to_schema = self.generate_to_schema()

    def generate_to_time_config(self) -> DatetimeRangeBase:
        if self._to_time_zone:
            to_time_config = self._from_schema.time_config.convert_time_zone(
                self._to_time_zone
            ).replace_time_zone(None)
        else:
            to_time_config = self._from_schema.time_config.replace_time_zone(None)
        time_kwargs = to_time_config.model_dump()
        time_kwargs = dict(
            filter(
                lambda k_v: k_v[0] in DatetimeRangeWithTZColumn.model_fields,
                time_kwargs.items(),
            )
        )
        time_kwargs["time_type"] = TimeType.DATETIME_TZ_COL
        time_kwargs["time_zone_column"] = "time_zone"
        time_kwargs["time_zones"] = [self._to_time_zone]
        return DatetimeRangeWithTZColumn(**time_kwargs)

    def generate_to_schema(self) -> TableSchema:
        id_cols = self._from_schema.time_array_id_columns
        if "time_zone" not in id_cols:
            id_cols.append("time_zone")
        to_schema: TableSchema = self._from_schema.model_copy(
            update={
                "name": f"{self._from_schema.name}_tz_converted",
                "time_config": self.generate_to_time_config(),
                "time_array_id_columns": id_cols,
            }
        )
        return to_schema

    def convert_time_zone(
        self,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        self.check_from_schema()
        df, mapping_schema = self._create_mapping()

        apply_mapping(
            df,
            mapping_schema,
            self._from_schema,
            self._to_schema,
            self._engine,
            self._metadata,
            TimeBasedDataAdjustment(),
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

    def _create_mapping(self) -> tuple[pd.DataFrame, MappingTableSchema]:
        """Create mapping dataframe for converting datetime to geography-based time zone"""
        time_col = self._from_schema.time_config.time_column
        from_time_col = "from_" + time_col
        from_time_data = make_time_range_generator(self._from_schema.time_config).list_timestamps()
        to_time_data_dct = make_time_range_generator(
            self._to_schema.time_config
        ).list_timestamps_by_time_zone()

        from_time_config = self._from_schema.time_config.model_copy(
            update={"time_column": from_time_col}
        )
        to_time_config = self._to_schema.time_config
        tz_col = to_time_config.time_zone_column
        tz_name = self._to_time_zone.key if self._to_time_zone else "None"
        to_time_data = to_time_data_dct[tz_name]
        df = pd.DataFrame(
            {
                from_time_col: from_time_data,
                tz_col: tz_name,
                time_col: to_time_data,
            }
        )

        mapping_schema = MappingTableSchema(
            name="mapping_table_gtz_conversion",
            time_configs=[from_time_config, to_time_config],
        )
        return df, mapping_schema


class TimeZoneConverterByColumn(TimeZoneConverterBase):
    """Class for time zone conversion of time series data based on a time zone column."""

    def __init__(
        self,
        engine: Engine,
        metadata: MetaData,
        from_schema: TableSchema,
        time_zone_column: str,
        wrap_time_allowed: Optional[bool] = False,
    ):
        if time_zone_column not in from_schema.time_array_id_columns:
            msg = f"{time_zone_column=} is missing from {from_schema.time_array_id_columns=}"
            raise MissingValue(msg)
        super().__init__(engine, metadata, from_schema)
        self.time_zone_column = time_zone_column
        self._wrap_time_allowed = wrap_time_allowed
        self._to_schema = self.generate_to_schema()

    def generate_to_time_config(self) -> DatetimeRangeBase:
        if self._wrap_time_allowed:
            to_time_config = self._from_schema.time_config.replace_time_zone(None)
        else:
            to_time_config = self._from_schema.time_config
        time_kwargs = to_time_config.model_dump()
        time_kwargs = dict(
            filter(
                lambda k_v: k_v[0] in DatetimeRangeWithTZColumn.model_fields,
                time_kwargs.items(),
            )
        )
        time_kwargs["time_type"] = TimeType.DATETIME_TZ_COL
        time_kwargs["time_zone_column"] = self.time_zone_column
        time_kwargs["time_zones"] = self._get_time_zones()
        return DatetimeRangeWithTZColumn(**time_kwargs)

    def generate_to_schema(self) -> TableSchema:
        id_cols = self._from_schema.time_array_id_columns
        if "time_zone" not in id_cols:
            id_cols.append("time_zone")
        to_schema: TableSchema = self._from_schema.model_copy(
            update={
                "name": f"{self._from_schema.name}_tz_converted",
                "time_config": self.generate_to_time_config(),
                "time_array_id_columns": id_cols,
            }
        )
        return to_schema

    def convert_time_zone(
        self,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        self.check_from_schema()
        df, mapping_schema = self._create_mapping()

        apply_mapping(
            df,
            mapping_schema,
            self._from_schema,
            self._to_schema,
            self._engine,
            self._metadata,
            TimeBasedDataAdjustment(),
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

    def _get_time_zones(self) -> list[ZoneInfo | None]:
        with self._engine.connect() as conn:
            table = Table(self._from_schema.name, self._metadata)
            stmt = (
                select(table.c[self.time_zone_column])
                .distinct()
                .where(table.c[self.time_zone_column].is_not(None))
            )
            time_zones = read_database(stmt, conn, self._from_schema.time_config)[
                self.time_zone_column
            ].to_list()

        time_zones = [None if tz == "None" else ZoneInfo(tz) for tz in time_zones]
        return time_zones

    def _create_mapping(self) -> tuple[pd.DataFrame, MappingTableSchema]:
        """Create mapping dataframe for converting datetime to column time zones"""
        time_col = self._from_schema.time_config.time_column
        from_time_col = "from_" + time_col
        from_time_data = make_time_range_generator(self._from_schema.time_config).list_timestamps()
        to_time_data_dct = make_time_range_generator(
            self._to_schema.time_config
        ).list_timestamps_by_time_zone()

        from_tz_col = "from_" + self.time_zone_column
        from_time_config = self._from_schema.time_config.model_copy(
            update={"time_column": from_time_col}
        )
        to_time_config = self._to_schema.time_config

        df_tz = []
        for tz_name, time_data in to_time_data_dct.items():
            if self._wrap_time_allowed:
                # assume it is being wrapped based on the tz-naive version of the original time data
                final_time_data = [x.replace(tzinfo=None) for x in from_time_data]
                to_time_data = wrap_timestamps(pd.Series(time_data), pd.Series(final_time_data))
            else:
                to_time_data = time_data
            df_tz.append(
                pd.DataFrame(
                    {
                        from_time_col: from_time_data,
                        from_tz_col: tz_name,
                        time_col: to_time_data,
                    }
                )
            )
        df = pd.concat(df_tz, ignore_index=True)

        mapping_schema = MappingTableSchema(
            name="mapping_table_gtz_conversion",
            time_configs=[from_time_config, to_time_config],
        )
        return df, mapping_schema
