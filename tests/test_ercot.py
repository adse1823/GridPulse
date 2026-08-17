from unittest.mock import patch

import pandas as pd


def _load_df(n: int = 24) -> pd.DataFrame:
    return pd.DataFrame({
        "Time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "Load": [40000.0 + i * 50 for i in range(n)],
    })


def _fuel_df(n: int = 24) -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "Interval Start": times,
        "Wind": [5000.0] * n,
        "Solar": [1000.0] * n,
        "Natural Gas": [20000.0] * n,
    })


@patch("ingest.ercot._ISO")
def test_fetch_demand_shape(mock_iso):
    mock_iso.get_load.return_value = _load_df(24)
    from ingest.ercot import fetch_demand
    df = fetch_demand("2024-01-01", "2024-01-01")
    assert set(df.columns) == {"timestamp", "region", "demand_mw"}
    assert df["region"].eq("ERCO").all()
    assert str(df["timestamp"].dt.tz) == "UTC"


@patch("ingest.ercot._ISO")
def test_fetch_generation_melts_fuels(mock_iso):
    mock_iso.get_fuel_mix.return_value = _fuel_df(4)
    from ingest.ercot import fetch_generation
    df = fetch_generation("2024-01-01", "2024-01-01")
    assert set(df.columns) == {"timestamp", "region", "fuel_type", "generation_mw"}
    assert set(df["fuel_type"].unique()) == {"Wind", "Solar", "Natural Gas"}
    assert df["region"].eq("ERCO").all()
