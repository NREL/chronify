import abc
from zoneinfo import ZoneInfo
from sqlalchemy import Engine, MetaData, Table, select
from typing import Optional
from pathlib import Path
import pandas as pd

from chronify.models import TableSchema, MappingTableSchema
from chronify.time_configs import DatetimeRange, TimeBasedDataAdjustment
from chronify.exceptions import InvalidParameter
from chronify.time_series_mapper_base import apply_mapping
from chronify.time_series_mapper_datetime import MapperDatetimeToDatetime
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.sqlalchemy.functions import read_database


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
        if not isinstance(self._from_schema.time_config, DatetimeRange):
            msg = "Source schema does not have DatetimeRange time config."
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
        self, engine: Engine, metadata: MetaData, from_schema: TableSchema, time_zone_column: str
    ):
        super().__init__(engine, metadata, from_schema)
        self.time_zone_column = time_zone_column
        self._to_schema = self.generate_to_schema()

    def generate_to_schema(self) -> TableSchema:
        to_schema: TableSchema = self._from_schema.model_copy(
            update={
                "name": f"{self._from_schema.name}_tz_converted",
                "time_config": self._from_schema.time_config.replace_time_zone(None),
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

        # Do not check mapped timestamps because they cannot be described by the mapped_schema time_config
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
        )

    def _create_interm_map_with_time_zone(self) -> tuple[pd.DataFrame, MappingTableSchema]:
        """Create mapping dataframe for converting datetime to geography-based time zone"""
        mapped_time_col = self._from_schema.time_config.time_column
        from_time_col = "from_" + mapped_time_col
        from_time_data = make_time_range_generator(self._from_schema.time_config).list_timestamps()

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
        to_time_config = self._from_schema.time_config.replace_time_zone(None)

        df_tz = []
        for time_zone in time_zones:
            tz = ZoneInfo(time_zone) if time_zone not in [None, "None"] else None
            mapped_time_data = [x.tz_convert(tz).tz_localize(None) for x in from_time_data]
            df_tz.append(
                pd.DataFrame(
                    {
                        from_time_col: from_time_data,
                        from_tz_col: time_zone,
                        mapped_time_col: mapped_time_data,
                    }
                )
            )
        df = pd.concat(df_tz, ignore_index=True)

        mapping_schema = MappingTableSchema(
            name="mapping_table_gtz_conversion",
            time_configs=[from_time_config, to_time_config],
        )
        return df, mapping_schema
