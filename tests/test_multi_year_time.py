import pandas as pd


def index_time_data() -> pd.DataFrame:
    df1 = pd.DataFrame(
        {
            "year": 2019,
            "index_time": range(8760),
            "value": range(8760),
        }
    )
    df2 = pd.DataFrame(
        {
            "year": 2020,
            "index_time": range(8784),
            "value": range(8784),
        }
    )
    df = pd.concat([df1, df2], ignore_index=True)
    return df


def test_multi_year_index_time() -> None:
    # df = index_time_data()
    pass
