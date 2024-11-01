import pandas as pd
import numpy as np
import pytest
import duckdb

import chronify.duckdb.functions as ddbf

conn = duckdb.connect(config={"TimeZone": "US/Mountain"})  # US/Mountain, UTC
conn_tz = conn.execute("select * from duckdb_settings() where name='TimeZone';").fetchall()[0][1]
print(f"Duckdb connection TimeZone = {conn_tz}")


@pytest.fixture
def generate_data():
    ## [1] Create data
    # load data
    dfd = pd.DataFrame(
        {
            "id": np.repeat(1, 12 * 24 * 7),
            "month": np.repeat(range(1, 13), 24 * 7),
            "dow": np.tile(
                np.repeat(range(7), 24), 12
            ),  # 0: Monday, 6: Sunday, ~ pyspark.weekday(), duckdb.isodow()-1, pd.day_of_week
            "hour": np.tile(range(24), 12 * 7),
        }
    )
    dfd["is_weekday"] = False
    dfd.loc[dfd["dow"] < 5, "is_weekday"] = True
    dfd["value"] = dfd["month"] * 1000 + dfd["dow"] * 100 + dfd["hour"]

    dfd = pd.concat([dfd, dfd.assign(id=2).assign(value=dfd.value * 2)], axis=0)

    # project time
    dft = pd.DataFrame(
        {
            "tid": range(8760),
            "timestamp": pd.date_range(
                start="2018-01-01", periods=8760, freq="h", tz="US/Eastern"
            ),
        }
    )

    # mapping data
    dfg = pd.DataFrame(
        {
            "id": [1, 2],
            "geography": ["IL", "CO"],
            "timezone": ["US/Central", "US/Mountain"],
        }
    )

    dfm = []
    for idx, row in dfg.iterrows():
        dfgt = dft.copy()
        dfgt["timestamp_tmp"] = dfgt["timestamp"].dt.tz_convert(row["timezone"])
        dfgt["month"] = dfgt["timestamp_tmp"].dt.month
        dfgt["dow"] = dfgt["timestamp_tmp"].dt.day_of_week
        dfgt["hour"] = dfgt["timestamp_tmp"].dt.hour
        dfgt["timezone"] = row["timezone"]
        dfm.append(dfgt.drop(columns=["timestamp_tmp"]))
    dfm = pd.concat(dfm, axis=0, ignore_index=True)

    ## [3] Convert to DUCKDB
    ddfd = duckdb.sql("SELECT * FROM dfd").set_alias("ddfd")
    ddfg = duckdb.sql("SELECT * FROM dfg").set_alias("ddfg")
    ddfm = duckdb.sql("SELECT * FROM dfm").set_alias("ddfm")

    return ddfd, ddfg, ddfm


def test_join(generate_data):
    breakpoint()
    df = ddbf.join(generate_data[0], generate_data[1], ["id"])
    ddbf.join(df, generate_data[2], ["month", "dow", "hour", "timezone"])
