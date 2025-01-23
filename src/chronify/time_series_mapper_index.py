import logging

import pandas as pd
from sqlalchemy import Engine, MetaData


from chronify.models import TableSchema, MappingTableSchema
from chronify.exceptions import (
    InvalidParameter,
)
from chronify.time_series_mapper_base import TimeSeriesMapperBase, apply_mapping
from chronify.index_time_range_generator import IndexTimeRangeGenerator
from chronify.time_configs import DatetimeRange, IndexTimeRange


logger = logging.getLogger(__name__)


class MapperIndexTimeToDatetime(TimeSeriesMapperBase):
    def __init__(
        self,
        engine: Engine,
        metadata: MetaData,
        from_schema: TableSchema,
        to_schema: TableSchema,
    ) -> None:
        """
        Parameters
        ----------

        engine : Engine
            sqlalchemy engine to connect to database

        metadata : MetaData
            metadata for the database

        from_schema : TableSchema
            schema mapping from the input dataset

        to_schema : TableSchema
            schema mapping to the project

        """
        if not isinstance(from_schema.time_config, IndexTimeRange):
            msg = "source schema does not have IndexTimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        if not isinstance(to_schema.time_config, DatetimeRange):
            msg = "destination schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)

        super().__init__(engine, metadata, from_schema, to_schema)

        self._from_time_config: IndexTimeRange = from_schema.time_config
        self._to_time_config: DatetimeRange = to_schema.time_config
        self._from_schema = from_schema
        self._to_schema = to_schema
        self._engine = engine
        self._metadata = metadata
        self._generator = IndexTimeRangeGenerator(self._from_time_config)

        self._tz_naive: tuple[bool, bool] = (
            self._from_time_config.is_time_zone_naive(),
            self._to_time_config.is_time_zone_naive(),
        )

        self._check_tz_compatability()
        self._not_supported_mappings()

    def _not_supported_mappings(self) -> None:
        """Check for mappings that aren't currently supported"""

        # can't map from one tz to another
        if (
            self._tz_naive == (False, False)
            and self._to_time_config.start.tzinfo != self._from_time_config.start_timestamp.tzinfo
        ):
            msg = f"{self.__class__} can't map across timezones"
            raise NotImplementedError(msg)

        matching_fields = ["resolution", "length"]
        for field in matching_fields:
            if getattr(self._to_time_config, field) != getattr(self._from_time_config, field):
                msg = f"{self.__class__} can't map across different values for {field}"
                raise NotImplementedError(msg)

    def map_time(self) -> None:
        """Convert time columns with from_schema to to_schema configuration."""
        is_tz_naive = self._to_time_config.is_time_zone_naive()
        # self.check_schema_consistency()

        map_table_name = "map_table"
        dfm = self._create_mapping(is_tz_naive)
        mapping_schema = self._create_mapping_schema(map_table_name)
        apply_mapping(
            dfm, mapping_schema, self._from_schema, self._to_schema, self._engine, self._metadata
        )

    def create_single_tz_mapping_dataframe(
        self, to_time_col: str, from_time_col: str
    ) -> pd.DataFrame:
        """
        Creates a dataframe mapping by subtracting timestamps from the start_timestamp of
        the index_time schema.

        timestamps and start_timestamp need to either both be timezone aware or both be timezone naive
        """

        if self._tz_naive[0]:
            # create index to dt_naive table

            dfm_tz_naive = pd.DataFrame(
                {
                    from_time_col: self._generator.list_timestamps(),
                    to_time_col: pd.date_range(
                        start=self._from_time_config.start_timestamp,
                        periods=self._from_time_config.length,
                        freq=self._from_time_config.resolution,
                    ),
                }
            )

            if not self._tz_naive[1]:
                # mapping to tz specific, create datetime to timezone mapping
                dfm_tz_naive.rename(columns={to_time_col: "clock_time"}, inplace=True)
                dfm_tz_naive["clock_time"] = dfm_tz_naive["clock_time"].astype(str)

                dfm_dt_tz = pd.DataFrame(
                    {
                        to_time_col: pd.date_range(
                            start=self._to_time_config.start,
                            periods=self._to_time_config.length,
                            freq=self._to_time_config.resolution,
                        )
                    }
                )
                dfm_dt_tz["clock_time"] = dfm_dt_tz[to_time_col].dt.strftime("%Y-%m-%d %H:%M:%S")
                dfm = dfm_dt_tz.merge(dfm_tz_naive, on="clock_time")
                dfm["time_zone"] = str(self._to_time_config.start.tzinfo)

            else:
                dfm = dfm_tz_naive

        else:
            # from schema has tz, mapping is simple
            dfm = pd.DataFrame(
                {
                    from_time_col: self._generator.list_timestamps(),
                    to_time_col: pd.date_range(
                        start=self._to_time_config.start,
                        periods=self._to_time_config.length,
                        freq=self._to_time_config.resolution,
                        tz=str(self._to_time_config.start.tzinfo),
                    ),
                    "time_zone": str(self._to_time_config.start.tzinfo),
                }
            )

        return dfm

    def create_multi_tz_mapping_dataframe(
        self, dft: pd.DataFrame, to_time_col: str, time_zones: list[str], from_time_col: str
    ) -> pd.DataFrame:
        """Would index time ever have multiple timezones?"""
        # dfm = []
        # for tz in time_zones:
        #     dfc = dft.copy()
        #     dft["timestamp_tmp"] = dfc[to_time_col].dt.tz_convert(tz)
        #     dfc = self.create_single_tz_mapping_dataframe(dft, "timestamp_tmp", from_time_col)
        #     dft["from_time_zone"] = tz
        #     dfm.append(dft.drop(columns=["timestamp_tmp"]))
        # return pd.concat(dfm, ignore_index=True)
        return dft

    def _create_mapping(self) -> pd.DataFrame:
        """Create mapping dataframe"""

        to_time_col = self._to_time_config.time_column
        from_time_col = "from_" + self._from_time_config.time_column
        dfm = self.create_single_tz_mapping_dataframe(to_time_col, from_time_col)

        return dfm

    def _create_mapping_schema(self, table_name: str):
        return MappingTableSchema(name=table_name, time_configs=[self._to_time_config])
