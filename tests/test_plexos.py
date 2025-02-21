from pathlib import Path
from chronify.plexos_time_series import PlexosTimeSeriesFormats, PlexosTimeSeriesParser
from tempfile import NamedTemporaryFile
import pytest


def temp_csv_file(data: str):
    with NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(data)
        tmp_file.flush()
        return Path(tmp_file.name)


@pytest.fixture
def time_series_NMDH():
    hours = ",".join((str(x) for x in range(1, 25)))
    load1 = ",".join((str(x) for x in range(25, 49)))
    load2 = ",".join((str(x) for x in range(49, 73)))
    return temp_csv_file(f"name,month,day,{hours}\nGeneration,1,1,{load1}\nGeneration,1,2,{load2}")


@pytest.fixture
def time_series_NYMDH():
    hours = ",".join((str(x) for x in range(1, 25)))
    load1 = ",".join((str(x) for x in range(25, 49)))
    load2 = ",".join((str(x) for x in range(49, 73)))
    return temp_csv_file(
        f"name,year,month,day,{hours}\ntest_generator,2025,1,1,{load1}\ntest_generator,2025,1,2,{load2}"
    )


@pytest.fixture
def time_series_NYMDPV():
    header = "name,year,month,day,period,value\n"
    data = "test_generator,2025,1,1,H1-5,100\ntest_generator,2025,1,1,H6-12,200\ntest_generator,2025,1,1,H13-24,300"
    return temp_csv_file(header + data)


"""
case PlexosTimeSeriesFormats.TS_NMDH:
    pass
case PlexosTimeSeriesFormats.TS_NYMDH:
    pass
case PlexosTimeSeriesFormats.TS_NYMDPV:
    pass
"""


def test_NMDH_parser(time_series_NMDH, iter_engines):
    parser = PlexosTimeSeriesParser(iter_engines)
    parser.ingest_to_datetime(time_series_NMDH)


def test_NYMDH_parser(time_series_NYMDH, iter_engines):
    parser = PlexosTimeSeriesParser(iter_engines)
    parser.ingest_to_datetime(time_series_NYMDH)


def test_NYMDPV_parser(time_series_NYMDPV, iter_engines):
    parser = PlexosTimeSeriesParser(iter_engines)
    parser.ingest_to_datetime(time_series_NYMDPV)


def test_plexos():
    columns = ["name", "year", "month", "day", "period", "value"]
    _tmp = PlexosTimeSeriesFormats.from_columns(columns)
