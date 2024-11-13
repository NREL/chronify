import pandas as pd
import numpy as np

import duckdb


conn = duckdb.connect(config={"TimeZone": "US/Mountain"})  # US/Mountain, UTC
conn_tz = conn.execute("select * from duckdb_settings() where name='TimeZone';").fetchall()[0][1]
print(f"Duckdb connection TimeZone = {conn_tz}")

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
        "timestamp": pd.date_range(start="2018-01-01", periods=8760, freq="h", tz="US/Eastern"),
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

## [2] Create mapping in pd.dataframe
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
ddfd = conn.sql("SELECT * FROM dfd").set_alias("ddfd")
ddfg = conn.sql("SELECT * FROM dfg").set_alias("ddfg")
ddfm = conn.sql("SELECT * FROM dfm").set_alias("ddfm")


def get_join_statement(left_df, right_df, keys: list):
    stmts = [f"{left_df.alias}.{key}={right_df.alias}.{key}" for key in keys]
    return " and ".join(stmts)


def get_select_after_join_statement(left_df, right_df, keys: list):
    left_cols = [f"{left_df.alias}.{x}" for x in left_df.columns]
    right_cols = [x for x in right_df.columns if x not in keys]
    return ", ".join(left_cols + right_cols)


## [4] Apply mapping in DUCKDB
breakpoint()
keys = ["id"]
join_stmt = get_join_statement(ddfd, ddfg, keys)
select_stmt = get_select_after_join_statement(ddfd, ddfg, keys)
ddfdg = ddfd.join(ddfg, join_stmt).select(select_stmt).set_alias("ddfdg")

keys = ["month", "dow", "hour", "timezone"]
join_stmt = get_join_statement(ddfdg, ddfm, keys)
select_stmt = get_select_after_join_statement(ddfdg, ddfm, keys)
ddf = ddfdg.join(ddfm, join_stmt).select(select_stmt).set_alias("ddf")

# downselect to final columns
columns = ["id", "geography", "timezone", "tid", "timestamp", "value"]
select_stmt = ", ".join(columns)
ddf = ddf.select(select_stmt)

## [5] Check final ddf
sfi = 1656  # 2AM gap is between 1657 and 1658 in the project timezone
fbi = 7367  # duplicating 1AM are 7368 and 7369 in the project timezone
filter_stmt = f"tid BETWEEN {sfi} AND {sfi+4} OR tid BETWEEN {fbi} AND {fbi+4}"
ddf.filter(filter_stmt).show()

# if creating time mapping functions in Duckdb, convert project time to geography timezone and cast as tz-naive time, then extract time attributes
"""
ddft2 = ddf.select("*, TIMEZONE('US/Pacific', timestamp) AS tmp_timestamp")
select_stmt2 = "*, HOUR(tmp_timestamp) AS hour, MONTH(tmp_timestamp) AS month, ISODOW(tmp_timestamp)-1 AS dow"
ddft2.select(select_stmt2).filter(filter_stmt).show()
"""
