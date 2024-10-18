import pytest
from sqlalchemy import Integer

from chronify.models import ColumnDType, _check_name


def test_column_dtypes():
    ColumnDType(name="col1", dtype=Integer)
    for string_type in ("int", "bigint", "bool", "datetime", "float", "str"):
        ColumnDType(name="col1", dtype=string_type)

    with pytest.raises(ValueError):
        ColumnDType(name="col1", dtype="invalid")


def test_invalid_column_name():
    with pytest.raises(ValueError):
        _check_name(name="invalid - name")