from typing import Any, Generator

import pytest
from sqlalchemy import Engine, create_engine


ENGINES: dict[str, dict[str, Any]] = {
    "duckdb": {"url": "duckdb:///:memory:", "connect_args": {}, "kwargs": {}},
    "sqlite": {"url": "sqlite:///:memory:", "connect_args": {}, "kwargs": {}},
}


@pytest.fixture
def create_duckdb_engine() -> Engine:
    return create_engine("duckdb:///:memory:")


@pytest.fixture(params=ENGINES.keys())
def iter_engines(request) -> Generator[Engine, None, None]:
    engine = ENGINES[request.param]
    yield create_engine(engine["url"], *engine["connect_args"], **engine["kwargs"])
