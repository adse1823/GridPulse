from unittest.mock import MagicMock, patch

import pandas as pd

from ingest.weather import ERCOT_POINTS, fetch_forecast, fetch_historical


def _mock_response(lat: float, lon: float, n: int = 4) -> MagicMock:
    times = pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "hourly": {
            "time": times,
            "temperature_2m": [20.0] * n,
            "wind_speed_10m": [5.0] * n,
            "shortwave_radiation": [100.0] * n,
        }
    }
    return mock


@patch("ingest.weather.requests.get")
def test_fetch_historical_shape(mock_get):
    mock_get.side_effect = [_mock_response(pt["lat"], pt["lon"]) for pt in ERCOT_POINTS]
    df = fetch_historical("2024-01-01", "2024-01-01")
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {
        "timestamp", "latitude", "longitude",
        "temperature_c", "wind_speed_10m_ms", "shortwave_radiation", "is_forecast",
    }
    assert len(df) == 4 * len(ERCOT_POINTS)
    assert df["is_forecast"].eq(False).all()


@patch("ingest.weather.requests.get")
def test_fetch_forecast_shape(mock_get):
    mock_get.side_effect = [_mock_response(pt["lat"], pt["lon"], n=48) for pt in ERCOT_POINTS]
    df = fetch_forecast()
    assert len(df) == 48 * len(ERCOT_POINTS)
    assert df["is_forecast"].eq(True).all()


@patch("ingest.weather.requests.get")
def test_timestamps_are_utc(mock_get):
    mock_get.side_effect = [_mock_response(pt["lat"], pt["lon"]) for pt in ERCOT_POINTS]
    df = fetch_historical("2024-01-01", "2024-01-01")
    assert str(df["timestamp"].dt.tz) == "UTC"
