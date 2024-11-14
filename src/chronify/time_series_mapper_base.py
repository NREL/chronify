import abc


class TimeSeriesMapperBase(abc.ABC):
    """Maps time series data from one configuration to another."""

    @abc.abstractmethod
    def check_schema_consistency(self) -> None:
        """Check that from_schema can produce to_schema."""

    @abc.abstractmethod
    def map_time(self) -> None:
        """Convert time columns with from_schema to to_schema configuration."""

    @abc.abstractmethod
    def _check_schema_measurement_type_consistency(self) -> None:
        """Check that measurement_type is the same between schema."""
