import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def _mock_response(data: list[dict]) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {"response": {"data": data}}
    return mock


def _demand_rows(n: int = 4) -> list[dict]:
    times = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return [{"period": t.strftime("%Y-%m-%dT%H"), "value": 40000 + i * 100} for i, t in enumerate(times)]


def _gen_rows(n: int = 4) -> list[dict]:
    times = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return [{"period": t.strftime("%Y-%m-%dT%H"), "fueltype": "WND", "value": 5000 + i} for i, t in enumerate(times)]


@patch("ingest.eia.requests.get")
@patch("ingest.eia._api_key", return_value="test-key")
def test_fetch_demand_shape(mock_key, mock_get):
    mock_get.side_effect = [_mock_response(_demand_rows(4)), _mock_response([])]
    from ingest.eia import fetch_demand
    df = fetch_demand("2024-01-01", "2024-01-01")
    assert set(df.columns) == {"timestamp", "region", "demand_mw"}
    assert len(df) == 4
    assert df["region"].eq("ERCO").all()
    assert str(df["timestamp"].dt.tz) == "UTC"


@patch("ingest.eia.requests.get")
@patch("ingest.eia._api_key", return_value="test-key")
def test_fetch_generation_shape(mock_key, mock_get):
    mock_get.side_effect = [_mock_response(_gen_rows(4)), _mock_response([])]
    from ingest.eia import fetch_generation
    df = fetch_generation("2024-01-01", "2024-01-01")
    assert set(df.columns) == {"timestamp", "region", "fuel_type", "generation_mw"}
    assert len(df) == 4
    assert df["fuel_type"].eq("WND").all()


@patch("ingest.eia._api_key", return_value="")
def test_missing_key_raises(mock_key):
    mock_key.side_effect = EnvironmentError("EIA_API_KEY not set")
    from ingest.eia import fetch_demand
    with pytest.raises(EnvironmentError):
        fetch_demand("2024-01-01", "2024-01-01")
