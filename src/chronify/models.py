import re
from typing import Any, Optional

import duckdb.typing
import pandas as pd
from duckdb.typing import DuckDBPyType
from pydantic import Field, field_validator, model_validator
from sqlalchemy import BigInteger, Boolean, DateTime, Double, Integer, String
from typing_extensions import Annotated

from chronify.base_models import ChronifyBaseModel
from chronify.exceptions import InvalidParameter
from chronify.time_configs import TimeConfig


REGEX_NAME_REQUIREMENT = re.compile(r"^\w+$")


class TableSchemaBase(ChronifyBaseModel):
    """Base model for table schema."""

    time_config: TimeConfig
    time_array_id_columns: Annotated[
        list[str],
        Field(
            description="Columns in the table that uniquely identify time arrays. "
            "These could be geographic identifiers, such as county and state, or an integer ID. "
            "Can be None if the table is pivoted and each pivoted column denotes a time array. "
            "Should not include time columns."
        ),
    ]

    @field_validator("time_config")
    @classmethod
    def check_time_config(cls, time_config: TimeConfig) -> TimeConfig:
        for column in time_config.list_time_columns():
            _check_name(column)
        return time_config

    @field_validator("time_array_id_columns")
    @classmethod
    def check_columns(cls, columns: list[str]) -> list[str]:
        for column in columns:
            _check_name(column)
        return columns

    def list_columns(self) -> list[str]:
        """Return the column names in the schema."""
        return self.time_array_id_columns + self.time_config.list_time_columns()


class TableSchema(TableSchemaBase):
    """Defines the schema for a time series table stored in the database."""

    name: Annotated[
        str,
        Field(description="Name of the table or view in the database.", frozen=True),
    ]
    value_column: Annotated[str, Field(description="Column in the table that contain values.")]

    @field_validator("name")
    @classmethod
    def check_name(cls, name: str) -> str:
        _check_name(name)
        return name

    @field_validator("value_column")
    @classmethod
    def check_column(cls, column: str) -> str:
        _check_name(column)
        return column

    def list_columns(self) -> list[str]:
        return super().list_columns() + [self.value_column]


class PivotedTableSchema(TableSchemaBase):
    """Defines the schema for an input table with pivoted format."""

    pivoted_dimension_name: str = Field(
        description="Use this name for the column representing the pivoted dimension during "
        "an unpivot operation.",
    )
    value_columns: list[str] = Field(description="Columns in the table that contain values.")
    time_array_id_columns: list[str] = []

    @field_validator("value_columns")
    @classmethod
    def check_column(cls, value_columns: str) -> str:
        for column in value_columns:
            _check_name(column)
        return value_columns

    @field_validator("time_array_id_columns")
    @classmethod
    def check_time_array_id_columns(cls, value: list[str]) -> list[str]:
        if value:
            msg = f"PivotedTableSchema doesn't yet support time_array_id_columns: {value}"
            raise ValueError(msg)
        return value

    def list_columns(self) -> list[str]:
        return super().list_columns() + self.value_columns


# TODO: print example tables here.

_COLUMN_TYPES = {
    "bool": Boolean,
    "datetime": DateTime,
    "float": Double,
    "int": Integer,
    "bigint": BigInteger,
    "str": String,
}

_DB_TYPES = {x for x in _COLUMN_TYPES.values()}

_DUCKDB_TYPES_TO_SQLALCHEMY_TYPES = {
    duckdb.typing.BIGINT.id: BigInteger,  # type: ignore
    duckdb.typing.BOOLEAN.id: Boolean,  # type: ignore
    duckdb.typing.DOUBLE.id: Double,  # type: ignore
    duckdb.typing.INTEGER.id: Integer,  # type: ignore
    duckdb.typing.VARCHAR.id: String,  # type: ignore
    # Note: timestamp requires special handling because of timezone in sqlalchemy.
}


def get_sqlalchemy_type_from_duckdb(duckdb_type: DuckDBPyType) -> Any:
    """Return the sqlalchemy type for a duckdb type."""
    match duckdb_type:
        case duckdb.typing.TIMESTAMP_TZ:  # type: ignore
            sqlalchemy_type = DateTime(timezone=True)
        case (
            duckdb.typing.TIMESTAMP  # type: ignore
            | duckdb.typing.TIMESTAMP_MS  # type: ignore
            | duckdb.typing.TIMESTAMP_NS  # type: ignore
            | duckdb.typing.TIMESTAMP_S  # type: ignore
        ):
            sqlalchemy_type = DateTime(timezone=False)
        case _:
            cls = _DUCKDB_TYPES_TO_SQLALCHEMY_TYPES.get(duckdb_type.id)
            if cls is None:
                msg = f"There is no sqlalchemy mapping for {duckdb_type=}"
                raise InvalidParameter(msg)
            sqlalchemy_type = cls()

    return sqlalchemy_type


def get_duckdb_type_from_sqlalchemy(sqlalchemy_type: Any) -> DuckDBPyType:
    """Return the duckdb type for a sqlalchemy type."""
    if isinstance(sqlalchemy_type, DateTime):
        duckdb_type = (
            duckdb.typing.TIMESTAMP_TZ  # type: ignore
            if sqlalchemy_type.timezone
            else duckdb.typing.TIMESTAMP  # type: ignore
        )
    elif isinstance(sqlalchemy_type, BigInteger):
        duckdb_type = duckdb.typing.BIGINT  # type: ignore
    elif isinstance(sqlalchemy_type, Boolean):
        duckdb_type = duckdb.typing.BOOLEAN  # type: ignore
    elif isinstance(sqlalchemy_type, Double):
        duckdb_type = duckdb.typing.DOUBLE  # type: ignore
    elif isinstance(sqlalchemy_type, Integer):
        duckdb_type = duckdb.typing.INTEGER  # type: ignore
    elif isinstance(sqlalchemy_type, String):
        duckdb_type = duckdb.typing.VARCHAR  # type: ignore
    else:
        msg = f"There is no duckdb mapping for {sqlalchemy_type=}"
        raise InvalidParameter(msg)

    return duckdb_type  # type: ignore


def get_duckdb_types_from_pandas(df: pd.DataFrame) -> list[DuckDBPyType]:
    """Return a list of DuckDB types from a pandas dataframe."""
    # This seems least-prone to error, but is not exactly the most efficient.
    short_df = df.head(1)  # noqa: F841
    return duckdb.sql("select * from short_df").dtypes


class ColumnDType(ChronifyBaseModel):
    """Defines the dtype of a column."""

    name: str
    dtype: Any

    @model_validator(mode="before")
    @classmethod
    def fix_data_type(cls, data: dict[str, Any]) -> dict[str, Any]:
        dtype = data.get("dtype")
        if dtype is None or any(map(lambda x: isinstance(dtype, x), _DB_TYPES)):
            return data

        if isinstance(dtype, str):
            val = _COLUMN_TYPES.get(dtype)
            if val is None:
                options = sorted(_COLUMN_TYPES.keys()) + list(_DB_TYPES)
                msg = f"{dtype=} must be one of {options}"
                raise ValueError(msg)
            data["dtype"] = val()
        else:
            msg = f"dtype is an unsupported type: {type(dtype)}. It must be a str or type."
            raise ValueError(msg)
        return data


class CsvTableSchema(TableSchemaBase):
    """Defines the schema of data in a CSV file."""

    pivoted_dimension_name: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Only set if the table is pivoted. Use this name for the column "
            "representing that dimension when unpivoting.",
        ),
    ]
    column_dtypes: Annotated[
        Optional[list[ColumnDType]],
        Field(
            default=None,
            description="Column types. Will try to infer types of any column not listed.",
        ),
    ]
    value_columns: Annotated[
        list[str], Field(description="Columns in the table that contain values.")
    ]
    time_array_id_columns: list[str] = Field(
        default=[],
        description="Columns in the table that uniquely identify time arrays. "
        "These could be geographic identifiers, such as county and state, or an integer ID. "
        "Can be empty if the table is pivoted and each pivoted column denotes a time array. "
        "Should not include time columns.",
    )

    @field_validator("value_columns")
    @classmethod
    def check_columns(cls, columns: list[str]) -> list[str]:
        for column in columns:
            _check_name(column)
        return columns

    def list_columns(self) -> list[str]:
        return super().list_columns() + self.value_columns


class CsvTableSchemaSingleTimeArrayPivotedByComponent(CsvTableSchema):
    """Defines the schema of data in a CSV file where there is a single time array for each of
    multiple components and those components are pivoted columns."""


def _check_name(name: str) -> None:
    if not REGEX_NAME_REQUIREMENT.search(name):
        msg = f"A name can only have alphanumeric characters: {name=}"
        raise ValueError(msg)
