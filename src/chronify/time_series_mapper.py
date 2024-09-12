from chronify.time_configs import TimeConfig


class TimeSeriesMapper:
    """Maps time series data from one configuration to another."""

    def map_time_series(
        self,
        from_table: str,
        from_config: TimeConfig,
        to_config: TimeConfig,
        to_table: str,
    ) -> None:
        pass
