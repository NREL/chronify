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
from chronify.exceptions import InvalidParameter
from chronify.time_series_mapper_base import apply_mapping
from chronify.time_series_mapper_datetime import MapperDatetimeToDatetime
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.sqlalchemy.functions import read_database
from chronify.time import TimeType
from chronify.time_utils import wrap_timestamps


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

    def generate_to_schema(self) -> TableSchema:
        to_schema: TableSchema = self._from_schema.model_copy(
            update={
                "name": f"{self._from_schema.name}_tz_converted",
                "time_config": self._from_schema.time_config.convert_time_zone(self._to_time_zone),
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
        MapperDatetimeToDatetime(
            self._engine,
            self._metadata,
            self._from_schema,
            self._to_schema,
        ).map_time(
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )


class TimeZoneConverterByGeography(TimeZoneConverterBase):
    """Class for time zone conversion of time series data based on a geography-based time zone column."""

    def __init__(
        self,
        engine: Engine,
        metadata: MetaData,
        from_schema: TableSchema,
        time_zone_column: str,
        wrap_time_allowed: Optional[bool] = False,
    ):
        super().__init__(engine, metadata, from_schema)
        self.time_zone_column = time_zone_column
        self._wrap_time_allowed = wrap_time_allowed
        self._to_schema = self.generate_to_schema()

    def generate_to_time_config(self) -> DatetimeRangeBase:
        if self._wrap_time_allowed:
            time_kwargs = self._from_schema.time_config.model_dump()
            time_kwargs = dict(
                filter(
                    lambda k_v: k_v[0] in DatetimeRangeWithTZColumn.model_fields,
                    time_kwargs.items(),
                )
            )
            time_kwargs["time_type"] = TimeType.DATETIME_TZ_COL
            time_kwargs["start"] = self._from_schema.time_config.start.replace(tzinfo=None)
            time_kwargs["time_zone_column"] = self.time_zone_column
            return DatetimeRangeWithTZColumn(**time_kwargs)

        return self._from_schema.time_config.replace_time_zone(None)

    def generate_to_schema(self) -> TableSchema:
        to_schema: TableSchema = self._from_schema.model_copy(
            update={
                "name": f"{self._from_schema.name}_tz_converted",
                "time_config": self.generate_to_time_config(),
            }
        )
        return to_schema

    def convert_time_zone(
        self,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,  # will not be used
    ) -> None:
        self.check_from_schema()
        df, mapping_schema = self._create_interm_map_with_time_zone()

        # Do not check mapped timestamps when not wrap_time_allowed
        # because they cannot be fully described by the to_schema time_config
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
            check_mapped_timestamps=check_mapped_timestamps if self._wrap_time_allowed else False,
        )

    def _create_interm_map_with_time_zone(self) -> tuple[pd.DataFrame, MappingTableSchema]:
        """Create mapping dataframe for converting datetime to geography-based time zone"""
        mapped_time_col = self._from_schema.time_config.time_column
        from_time_col = "from_" + mapped_time_col
        from_time_data = make_time_range_generator(self._from_schema.time_config).list_timestamps()

        if self._wrap_time_allowed:
            to_time_data = make_time_range_generator(self._to_schema.time_config).list_timestamps()

        from_tz_col = "from_" + self.time_zone_column

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

        from_time_config = self._from_schema.time_config.model_copy(
            update={"time_column": from_time_col}
        )
        to_time_config = self.generate_to_time_config()

        df_tz = []
        for time_zone in time_zones:
            tz = ZoneInfo(time_zone) if time_zone not in [None, "None"] else None
            converted_time_data = [x.tz_convert(tz).tz_localize(None) for x in from_time_data]
            if self._wrap_time_allowed:
                final_time_data = wrap_timestamps(
                    pd.Series(converted_time_data), pd.Series(to_time_data)
                )
            else:
                final_time_data = converted_time_data
            df_tz.append(
                pd.DataFrame(
                    {
                        from_time_col: from_time_data,
                        from_tz_col: time_zone,
                        mapped_time_col: final_time_data,
                    }
                )
            )
        df = pd.concat(df_tz, ignore_index=True)

        mapping_schema = MappingTableSchema(
            name="mapping_table_gtz_conversion",
            time_configs=[from_time_config, to_time_config],
        )
        return df, mapping_schema
