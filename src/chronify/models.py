import re
from typing import Any, Optional, Type

import duckdb.typing
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import BigInteger, Boolean, DateTime, Double, Integer, String
from typing_extensions import Annotated

from chronify.exceptions import InvalidParameter
from chronify.time_configs import TimeConfig


def make_model_config(**kwargs) -> ConfigDict:
    """Return a Pydantic config"""
    return ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        validate_default=True,
        extra="forbid",
        use_enum_values=False,
        arbitrary_types_allowed=True,
        populate_by_name=True,
        **kwargs,  # type: ignore
    )


class ChronifyBaseModel(BaseModel):
    """Base model for all TSS data models"""

    model_config = make_model_config()


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
        for column in time_config.time_columns:
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
        return self.time_array_id_columns + self.time_config.time_columns


class TableSchema(TableSchemaBase):
    """Defines the schema for a time series table stored in the database."""

    name: Annotated[
        str, Field(description="Name of the table or view in the database.", frozen=True)
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
    duckdb.typing.TIMESTAMP.id: DateTime,  # type: ignore
    duckdb.typing.VARCHAR.id: String,  # type: ignore
}

_SQLALCHEMY_TYPES_TO_DUCKDB_TYPES = {v: k for k, v in _DUCKDB_TYPES_TO_SQLALCHEMY_TYPES.items()}


def get_sqlalchemy_type_from_duckdb(duckdb_type: duckdb.typing.DuckDBPyType) -> Type:  # type: ignore
    """Return the sqlalchemy type for a duckdb type."""
    sqlalchemy_type = _DUCKDB_TYPES_TO_SQLALCHEMY_TYPES.get(duckdb_type.id)
    if sqlalchemy_type is None:
        msg = f"There is no sqlalchemy mapping for {duckdb_type=}"
        raise InvalidParameter(msg)
    return sqlalchemy_type


def get_duckdb_type_from_sqlalchemy(sqlalchemy_type) -> str:
    """Return the duckdb type for a sqlalchemy type."""
    duckdb_type = _SQLALCHEMY_TYPES_TO_DUCKDB_TYPES.get(sqlalchemy_type)
    if duckdb_type is None:
        msg = f"There is no duckdb mapping for {sqlalchemy_type=}"
        raise InvalidParameter(msg)
    return duckdb_type.upper()


class ColumnDType(ChronifyBaseModel):
    """Defines the dtype of a column."""

    name: str
    dtype: Type

    @model_validator(mode="before")
    @classmethod
    def fix_data_type(cls, data: dict[str, Any]) -> dict[str, Any]:
        dtype = data.get("dtype")
        if dtype is None or dtype in _DB_TYPES:
            return data

        if isinstance(dtype, str):
            val = _COLUMN_TYPES.get(dtype)
            if val is None:
                options = sorted(_COLUMN_TYPES.keys()) + list(_DB_TYPES)
                msg = f"{dtype=} must be one of {options}"
                raise ValueError(msg)
            data["dtype"] = val
        else:
            msg = "dtype is an unsupported type: {type(dtype)}. It must be a str or type."
            raise ValueError(msg)
        return data


class PivotedFormatMetadata(ChronifyBaseModel):
    """Defines metadata for a pivoted table."""

    pivoted_dimension_name: str


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
        list[str], Field(description="Column in the table that contain values.")
    ]

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
