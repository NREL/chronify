import re

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Annotated

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


class TSSBaseModel(BaseModel):
    """Base model for all TSS data models"""

    model_config = make_model_config()


REGEX_NAME_REQUIREMENT = re.compile(r"^\w+$")


class TableSchema(TSSBaseModel):
    """Defines the schema for a time series table."""

    name: Annotated[
        str, Field(description="Name of the table or view in the database.", frozen=True)
    ]
    time_config: TimeConfig
    time_array_id_columns: Annotated[
        list[str],
        Field(
            description="Columns in the table that uniquely identify time arrays. "
            "These could be geographic identifiers, such as county and state, or an integer ID. "
            "Should not include time columns."
        ),
    ]
    value_columns: Annotated[
        list[str], Field(description="Columns in the table that contain values.")
    ]
    is_pivoted: Annotated[
        bool,
        Field(
            default=False,
            description="Set to True if the value columns are a pivoted dimension.",
        ),
    ]

    @field_validator("name")
    @classmethod
    def check_name(cls, name: str) -> str:
        _check_name(name)
        return name

    @field_validator("time_config")
    @classmethod
    def check_time_config(cls, time_config: TimeConfig) -> TimeConfig:
        for column in time_config.time_columns:
            _check_name(column)
        return time_config

    @field_validator("time_array_id_columns", "value_columns")
    @classmethod
    def check_columns(cls, columns: list[str]) -> list[str]:
        for column in columns:
            _check_name(column)
        return columns


# TODO: print example tables here.


def _check_name(name: str) -> None:
    if not REGEX_NAME_REQUIREMENT.search(name):
        msg = f"A name can only have alphanumeric characters: {name=}"
        raise ValueError(msg)
