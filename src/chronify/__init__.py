import importlib.metadata as metadata

from chronify.exceptions import (
    ChronifyExceptionBase,
    ConflictingInputsError,
    InvalidTable,
    InvalidOperation,
    InvalidParameter,
    MissingParameter,
    TableAlreadyExists,
    TableNotStored,
)
from chronify.models import (
    ColumnDType,
    CsvTableSchema,
    PivotedTableSchema,
    TableSchema,
)
from chronify.store import Store
from chronify.time import RepresentativePeriodFormat
from chronify.time_configs import (
    AnnualTimeRange,
    DatetimeRange,
    IndexTimeRangeBase,
    RepresentativePeriodTime,
    TimeBaseModel,
    TimeBasedDataAdjustment,
)

__all__ = (
    "AnnualTimeRange",
    "ChronifyExceptionBase",
    "ColumnDType",
    "ConflictingInputsError",
    "CsvTableSchema",
    "DatetimeRange",
    "IndexTimeRangeBase",
    "InvalidOperation",
    "InvalidParameter",
    "InvalidTable",
    "MissingParameter",
    "PivotedTableSchema",
    "RepresentativePeriodFormat",
    "RepresentativePeriodTime",
    "Store",
    "TableAlreadyExists",
    "TableNotStored",
    "TableSchema",
    "TimeBaseModel",
    "TimeBasedDataAdjustment",
)

__version__ = metadata.metadata("chronify")["Version"]
