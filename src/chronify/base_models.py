from typing import Any

from pydantic import BaseModel, ConfigDict


def make_model_config(**kwargs: Any) -> Any:
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
    """Base model for all chronify data models"""

    model_config = make_model_config()
