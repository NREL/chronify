import pandas as pd
import pytest
import numpy as np
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from typing import Any
from sqlalchemy import Engine, MetaData


from chronify.datetime_range_generator import DatetimeRangeGenerator
from chronify.time_configs import IndexTimeRange, TimeBasedDataAdjustment
from chronify.index_time_range_generator import IndexTimeRangeGenerator
from chronify.models import TableSchema

from chronify.sqlalchemy.functions import read_database, write_database
from chronify.time_series_mapper import map_time
from chronify.time_configs import DatetimeRange
from chronify.time import DaylightSavingsDataAdjustment, TimeIntervalType


def generate_indextime_data(time_config: IndexTimeRange) -> np.ndarray:  # type: ignore
    return np.fromiter(IndexTimeRangeGenerator(time_config).iter_timestamps(), int)


def generate_indextime_dataframe(
    schema: TableSchema, time_col_values: dict[str, list]
) -> pd.DataFrame:
    assert isinstance(schema.time_config, IndexTimeRange)
    assert set(time_col_values.keys()) == set(schema.time_array_id_columns)

    n_time_arrays = max(len(v) for v in time_col_values.values())

    t_idxs = generate_indextime_data(schema.time_config)
    n_idxs = t_idxs.shape[0]

    df = pd.DataFrame({schema.time_config.time_column: np.tile(t_idxs, n_time_arrays)})

    for x in schema.time_array_id_columns:
        values = time_col_values[x]
        n_values = len(values)

        df[x] = sum(
            ([time_col_values[x][i % n_values]] * n_idxs for i in range(n_time_arrays)), []
        )
    df[schema.value_column] = np.arange(len(df))
    return df


def _inc_per_year(year: int, resolution: timedelta, tzinfo: ZoneInfo | None = None):
    dt = datetime(year, 1, 1, 0, 0, tzinfo=tzinfo)
    inc = 0
    while dt.year == year:
        dt += resolution
        inc += 1
    return inc


def gen_index_time_schema(
    year: int,
    tzinfo: ZoneInfo | None,
    interval_type: TimeIntervalType,
    name: str,
    resolution: timedelta = timedelta(hours=1),
    add_tz_col: bool = False,
) -> TableSchema:
    start_timestamp = datetime(year=year, month=1, day=1, tzinfo=tzinfo)
    start = 0
    resolution = timedelta(hours=1)
    length = _inc_per_year(year, resolution)
    schema = TableSchema(
        name=name,
        time_config=IndexTimeRange(
            start_timestamp=start_timestamp,
            start=start,
            resolution=resolution,
            length=length,
            interval_type=interval_type,
            time_column="index_time",
        ),
        time_array_id_columns=["id"] + (["time_zone"] if add_tz_col else []),
        value_column="value",
    )
    return schema


def gen_datetime_schema(
    year: int,
    tzinfo: ZoneInfo | None,
    interval_type: TimeIntervalType,
    name: str,
    resolution: timedelta = timedelta(hours=1),
    add_tz_col: bool = False,
) -> TableSchema:
    start = datetime(year=year, month=1, day=1, tzinfo=tzinfo)
    resolution = timedelta(hours=1)
    length = _inc_per_year(year, resolution)
    schema = TableSchema(
        name=name,
        time_config=DatetimeRange(
            start=start,
            resolution=resolution,
            length=length,
            interval_type=interval_type,
            time_column="timestamp",
        ),
        time_array_id_columns=["id"] + (["time_zone"] if add_tz_col else []),
        value_column="value",
    )
    return schema


def ingest_data(
    engine: Engine,
    df: pd.DataFrame,
    schema: TableSchema,
) -> None:
    metadata = MetaData()
    with engine.begin() as conn:
        write_database(df, conn, schema.name, [schema.time_config], if_table_exists="replace")
    metadata.reflect(engine, views=True)


def run_test_with_error(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
    to_schema: TableSchema,
    error: tuple[Any, str],
    **kwargs,
) -> None:
    metadata = MetaData()
    ingest_data(engine, df, from_schema)
    with pytest.raises(error[0], match=error[1]):
        map_time(engine, metadata, from_schema, to_schema, check_mapped_timestamps=False, **kwargs)


def get_mapped_results(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
    to_schema: TableSchema,
    **kwargs,
) -> pd.DataFrame:
    metadata = MetaData()
    ingest_data(engine, df, from_schema)
    map_time(engine, metadata, from_schema, to_schema, check_mapped_timestamps=False, **kwargs)

    with engine.connect() as conn:
        query = f"select * from {to_schema.name}"
        queried = read_database(query, conn, to_schema.time_config)
    queried = queried.sort_values(by=to_schema.time_array_id_columns).reset_index(drop=True)

    return queried


@pytest.mark.parametrize(
    "tzinfo", [ZoneInfo("US/Eastern")]
)  # [ZoneInfo("UTC"), ZoneInfo("US/Eastern"), None])
def test_index_mapping_simple(
    iter_engines: Engine,
    tzinfo,
):
    """
    No wrapping, and no TimeBasedDataAdjustment
    """
    from_schema = gen_index_time_schema(
        year=2020,
        tzinfo=tzinfo,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        name="input_data",
        resolution=timedelta(hours=1),
        add_tz_col=False,
    )

    id_values = {"id": [1, 2, 3]}
    df = generate_indextime_dataframe(from_schema, id_values)

    to_schema = gen_datetime_schema(
        year=2020,
        tzinfo=ZoneInfo("US/Pacific"),
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        name="simple_output",
        resolution=timedelta(hours=1),
        add_tz_col=False,
    )
    queried = get_mapped_results(iter_engines, df, from_schema, to_schema, wrap_time_allowed=True)
    queried = queried.sort_values(by=["id", "timestamp"], ignore_index=True)

    assert np.array_equal(
        np.sort(queried["value"]), np.arange(queried.shape[0])
    ), "No value should have been dropped"

    assert not np.array_equal(queried["value"], np.arange(queried.shape[0]))

    expected_timestamps = pd.to_datetime(
        DatetimeRangeGenerator(to_schema.time_config).list_timestamps()
    )  # type: ignore

    for _, group in queried.groupby(list(id_values.keys()))[to_schema.time_config.time_column]:
        assert np.array_equal(
            group, expected_timestamps
        )  # each timearray should have all expected timestamps


def test_index_mapping_data_adjusments(
    iter_engines: Engine,
):
    """
    Test the implementation of TimeBasedDataAdjustments

    main case:
        industrial time, dropping an hour in spring, and duplicating an hour in the fall
    """
    from_schema = gen_index_time_schema(
        year=2020,
        tzinfo=None,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        name="input_data",
        resolution=timedelta(hours=1),
        add_tz_col=False,
    )

    id_values = {"id": [1, 2, 3]}
    df = generate_indextime_dataframe(from_schema, id_values)

    to_schema = gen_datetime_schema(
        year=2020,
        tzinfo=ZoneInfo("US/Mountain"),
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        name="simple_output",
        resolution=timedelta(hours=1),
        add_tz_col=False,
    )

    map_time_kwargs = {
        "time_based_data_adjustment": TimeBasedDataAdjustment(
            daylight_saving_adjustment=DaylightSavingsDataAdjustment.DUPLICATE
        )
    }
    queried = get_mapped_results(iter_engines, df, from_schema, to_schema, **map_time_kwargs)
    queried = queried.sort_values(by=["id", "timestamp"]).reset_index(drop=True)

    expected_timestamps = pd.to_datetime(
        DatetimeRangeGenerator(to_schema.time_config).list_timestamps()
    )

    for _, group in queried.groupby(list(id_values.keys())):
        # each timearray should have all expected timestamps
        assert np.array_equal(group[to_schema.time_config.time_column], expected_timestamps)

        values = group["value"].to_numpy()
        values_diff = values[1:] - values[:-1]

        assert np.sum(values_diff == 2) == 1, "One value should have been dropped"
        assert np.sum(values_diff == 0) == 1, "One value should have been duplicated"


def test_index_mapping_time_alignment(
    iter_engines: Engine,
):
    """
    Test the implementation of shift and wrap time, Only test that needs index->intermediate_dt->final_dt

    cases:
        - different period types?
        - time zone conversion and wrapping to one year
    """

    # TODO: test with time_zone column, where mapped ts need to be wrapped
    from_schema = gen_index_time_schema(
        year=2020,
        tzinfo=None,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        name="input_data",
        resolution=timedelta(hours=1),
        add_tz_col=True,
    )

    # should None be tested as a source time_zone?
    id_values = {"id": [1, 2, 3], "time_zone": ["US/Eastern", "US/Pacific"]}
    df = generate_indextime_dataframe(from_schema, id_values)

    to_schema = gen_datetime_schema(
        year=2020,
        tzinfo=ZoneInfo("US/Mountain"),
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        name="simple_output",
        resolution=timedelta(hours=1),
        add_tz_col=True,
    )

    map_kwargs = {"align_to_project": False}

    # TODO: should fail for align_to_project=False
    # run_test_with_error(iter_engines, df, from_schema, to_schema, Exception(), **map_kwargs)

    map_kwargs = {"align_to_project": True}

    queried = get_mapped_results(iter_engines, df, from_schema, to_schema, **map_kwargs)
    queried = queried.sort_values(by=["id", "timestamp"]).reset_index(drop=True)
