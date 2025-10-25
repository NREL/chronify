import pytest
from sqlalchemy import BigInteger, Boolean, DateTime, Double, Integer, String

from chronify.models import ColumnDType, _check_name
from chronify.exceptions import InvalidValue


def test_column_dtypes() -> None:
    ColumnDType(name="col1", dtype=Integer())
    for dtype in (BigInteger, Boolean, DateTime, Double, String):
        ColumnDType(name="col1", dtype=dtype())

    for string_type in ("int", "bigint", "bool", "datetime", "float", "str"):
        ColumnDType(name="col1", dtype=string_type)

    with pytest.raises(InvalidValue):
        ColumnDType(name="col1", dtype="invalid")


def test_invalid_column_name() -> None:
    with pytest.raises(InvalidValue):
        _check_name(name="invalid - name")
