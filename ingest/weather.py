import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST = "https://api.open-meteo.com/v1/forecast"
_VARS = "temperature_2m,wind_speed_10m,shortwave_radiation"

# Representative grid points per region.
# Chosen to cover major load centres and key generation corridors.
REGION_POINTS: dict[str, list[dict]] = {
    "ERCO": [
        {"lat": 32.78, "lon": -96.80},  # Dallas — largest load centre
        {"lat": 29.76, "lon": -95.37},  # Houston
        {"lat": 29.42, "lon": -98.49},  # San Antonio
        {"lat": 32.45, "lon": -99.73},  # Abilene — west TX wind corridor
    ],
    "CISO": [
        {"lat": 34.05, "lon": -118.24},  # Los Angeles — largest load centre
        {"lat": 37.77, "lon": -122.42},  # San Francisco — Bay Area + coastal wind
        {"lat": 38.58, "lon": -121.49},  # Sacramento — Central Valley, hot summers
        {"lat": 33.83, "lon": -116.54},  # Palm Springs — desert solar + peak A/C
    ],
    "PJM": [
        {"lat": 41.85, "lon": -87.65},  # Chicago — western anchor, largest load
        {"lat": 39.95, "lon": -75.17},  # Philadelphia — eastern load centre
        {"lat": 39.96, "lon": -82.99},  # Columbus — central Ohio
        {"lat": 37.54, "lon": -77.43},  # Richmond — Virginia/Southeast anchor
    ],
    "NYIS": [
        {"lat": 40.71, "lon": -74.01},  # New York City — dominant load centre
        {"lat": 42.65, "lon": -73.75},  # Albany — upstate NY
        {"lat": 42.89, "lon": -78.87},  # Buffalo — western NY
        {"lat": 43.10, "lon": -75.23},  # Utica — central NY
    ],
}

REGION_TZ = {
    "ERCO": "US/Central",
    "CISO": "US/Pacific",
    "PJM":  "US/Eastern",
    "NYIS": "US/Eastern",
}

# Keep old name as alias so existing imports don't break
ERCOT_POINTS = REGION_POINTS["ERCO"]


def _session() -> requests.Session:
    """Session with automatic retry on connection errors and 5xx responses."""
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,        # waits 2, 4, 8, 16, 32 s between attempts
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _get(session: requests.Session, url: str, params: dict, timeout: int) -> requests.Response:
    """GET with an extra application-level retry for RemoteDisconnected errors."""
    for attempt in range(1, 4):
        try:
            r = session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.exceptions.ConnectionError as exc:
            if attempt == 3:
                raise
            wait = 10 * attempt
            print(f"  [weather] connection error (attempt {attempt}/3), retrying in {wait}s: {exc}")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _parse(r: requests.Response, is_forecast: bool) -> pd.DataFrame:
    h = r.json()["hourly"]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(h["time"], utc=True),
        "temperature_c": h["temperature_2m"],
        "wind_speed_10m_ms": h["wind_speed_10m"],
        "shortwave_radiation": h["shortwave_radiation"],
        "is_forecast": is_forecast,
    })


def fetch_historical(start: str, end: str, region: str = "ERCO") -> pd.DataFrame:
    """Hourly historical weather for the given region. start/end: 'YYYY-MM-DD'"""
    points = REGION_POINTS[region]
    session = _session()
    frames = []
    for pt in points:
        r = _get(session, _ARCHIVE, {
            "latitude": pt["lat"],
            "longitude": pt["lon"],
            "start_date": start,
            "end_date": end,
            "hourly": _VARS,
            "timezone": "UTC",
        }, timeout=90)
        df = _parse(r, is_forecast=False)
        df["latitude"] = pt["lat"]
        df["longitude"] = pt["lon"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)[
        ["timestamp", "latitude", "longitude", "temperature_c",
         "wind_speed_10m_ms", "shortwave_radiation", "is_forecast"]
    ]


def fetch_forecast(region: str = "ERCO") -> pd.DataFrame:
    """48-hour weather forecast for the given region."""
    points = REGION_POINTS[region]
    session = _session()
    frames = []
    for pt in points:
        r = _get(session, _FORECAST, {
            "latitude": pt["lat"],
            "longitude": pt["lon"],
            "hourly": _VARS,
            "forecast_days": 2,
            "timezone": "UTC",
        }, timeout=30)
        df = _parse(r, is_forecast=True)
        df["latitude"] = pt["lat"]
        df["longitude"] = pt["lon"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)[
        ["timestamp", "latitude", "longitude", "temperature_c",
         "wind_speed_10m_ms", "shortwave_radiation", "is_forecast"]
    ]
