from collections.abc import Iterable
from enum import Enum
from pathlib import Path
import pandas as pd
from pandas.core.dtypes.dtypes import re
from sqlalchemy import Engine
from datetime import datetime


from chronify.store import Store
from chronify import exceptions

INT_COLS = ["year", "month", "day"] + list(str(x) for x in range(1, 25))
COLUMN_DTYPES: dict[str, str] = {
    "name": "str",
    "period": "str",
    "value": "float64",
    **{name: "int32" for name in INT_COLS},
}


class PlexosTimeSeriesFormats(Enum):
    TS_NYMDPV = ("name", "year", "month", "day", "period", "value")
    TS_NYMDH = (
        "name",
        "year",
        "month",
        "day",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
    )
    TS_NMDH = (
        "name",
        "month",
        "day",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
    )

    @classmethod
    def from_columns(cls, columns: Iterable[str]) -> "PlexosTimeSeriesFormats":
        column_set = set(columns)
        for selected_enum in cls.__members__.values():
            if column_set == set(selected_enum.value):
                return selected_enum

        msg = f"No format for columns: {columns}"
        raise exceptions.InvalidInput(msg)


class PlexosTimeSeriesParser:
    def __init__(self, engine: Engine | None = None) -> None:
        self._store = Store(engine=engine)

    @staticmethod
    def _check_input_format(data_file: Path) -> None:
        valid_extensions = [".csv"]
        if data_file.suffix not in valid_extensions:
            msg = f"{data_file.name} does not have a file extension in the supported extensions: {valid_extensions}"
            raise exceptions.InvalidInput(msg)

    @staticmethod
    def _read_data_file(data_file: Path) -> pd.DataFrame:
        return pd.read_csv(data_file, header=0, dtype=COLUMN_DTYPES)

    @staticmethod
    def _convert_to_datetime(data: pd.DataFrame) -> pd.DataFrame:
        plexos_format = PlexosTimeSeriesFormats.from_columns(data.columns)
        match plexos_format:
            case PlexosTimeSeriesFormats.TS_NMDH:
                df = NMDH_to_datetime(data)
            case PlexosTimeSeriesFormats.TS_NYMDH:
                df = NYMDH_to_datetime(data)
            case PlexosTimeSeriesFormats.TS_NYMDPV:
                df = NYMDPV_to_datetime(data)

        return df

    def ingest_to_datetime(self, data_file: Path):
        """
        Given a file of plexos time series data, convert the time format to datetime timestamps
        and ingest into database
        """
        self._check_input_format(data_file)
        df = self._read_data_file(data_file)
        df = self._convert_to_datetime(df)

        self._store
        # Create a CSVTableSchema (not pivoted?)
        # Create a DatetimeRange TableSchema to store in df


# Parsers
def NMDH_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    # No explicit datetime representation? needs a target year?
    pass


def NYMDH_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.melt(
        id_vars=["name", "month", "day", "year"],
        value_vars=[str(x) for x in range(1, 25)],
        var_name="hour",
    )
    df["hour"] = df["hour"].astype(int) - 1  # convert from 1-24 to 0-23 hour format

    df["datetime"] = df.apply(row_YMDH_to_datetime, axis=1)
    df = df.drop(columns=["month", "day", "year", "hour"])
    return df


def NYMDPV_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    period_mapping = generate_period_mapping(df.loc[:, "period"])
    if sorted(period_mapping.columns) != ["hour", "period"]:
        msg = "Expecting only hour periods like H1-5"
        raise exceptions.InvalidInput(msg)

    df = df.merge(period_mapping, on="period")
    df["hour"] = df["hour"] - 1
    df["datetime"] = df.apply(row_YMDH_to_datetime, axis=1)
    df = df.drop(columns=["month", "day", "year", "hour", "period"])
    return df


def row_YMDH_to_datetime(row):
    ymdh = row[["year", "month", "day", "hour"]].to_list()
    return datetime_from_YMDH(*ymdh)


def datetime_from_YMDH(year: int, month: int, day: int, hour: int) -> datetime:
    """return a datetime object from a tuple (year, month, day, hour)"""
    return datetime(year=year, month=month, day=day, hour=hour)


def generate_period_mapping(periods: pd.Series) -> pd.DataFrame:
    unique_periods = periods.unique()
    mappings = []
    for period_str in unique_periods:
        period_type, period_vals = parse_period(period_str)
        mappings.append(pd.DataFrame({"period": period_str, period_type: period_vals}))

    return pd.concat(mappings)


def parse_period(period: str) -> tuple[str, list[int]]:
    match = re.match(r"([HMD])(\d{1,2}-\d{1,2})", period)
    if match is None:
        msg = f"Cannot parse Period: {period}, expecting format like H1-4, D10-12, or M1-3"
        raise exceptions.InvalidInput(msg)
    match_map = {"H": "hour", "M": "month", "D": "day"}
    val_type, val = match.groups()
    start, end = (int(v) for v in val.split("-"))
    vals = list(range(start, end + 1))
    return match_map[val_type], vals
