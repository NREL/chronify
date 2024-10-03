from datetime import datetime, timedelta

from sqlalchemy import Double

from chronify.models import ColumnDType, CsvTableSchema, TableSchema
from chronify.store import Store
from chronify.time import TimeIntervalType
from chronify.time_configs import DatetimeRange


def test_ingest_csv():
    time_config = DatetimeRange(
        start=datetime(year=2020, month=1, day=1),
        resolution=timedelta(hours=1),
        length=8784,
        time_interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_columns=["timestamp"],
    )

    src_schema = CsvTableSchema(
        time_config=time_config,
        column_dtypes=[
            ColumnDType(name="gen1", dtype=Double),
            ColumnDType(name="gen2", dtype=Double),
            ColumnDType(name="gen3", dtype=Double),
        ],
        value_columns=["gen1", "gen2", "gen3"],
        pivoted_dimension_name="generator",
        time_array_id_columns=[],
    )
    dst_schema = TableSchema(
        name="generators",
        time_config=time_config,
        time_array_id_columns=["generator"],
        value_column="value",
    )
    store = Store()
    store.ingest_from_csv("tests/data/gen.csv", src_schema, dst_schema)
