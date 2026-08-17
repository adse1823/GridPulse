import pandas as pd
import requests

_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST = "https://api.open-meteo.com/v1/forecast"
_VARS = "temperature_2m,wind_speed_10m,shortwave_radiation"

# Representative ERCOT load centres + west TX wind corridor
ERCOT_POINTS = [
    {"lat": 32.78, "lon": -96.80},  # Dallas — largest load centre
    {"lat": 29.76, "lon": -95.37},  # Houston
    {"lat": 29.42, "lon": -98.49},  # San Antonio
    {"lat": 32.45, "lon": -99.73},  # Abilene — west TX wind corridor
]


def _parse(r: requests.Response, is_forecast: bool) -> pd.DataFrame:
    h = r.json()["hourly"]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(h["time"], utc=True),
        "temperature_c": h["temperature_2m"],
        "wind_speed_10m_ms": h["wind_speed_10m"],
        "shortwave_radiation": h["shortwave_radiation"],
        "is_forecast": is_forecast,
    })


def fetch_historical(start: str, end: str) -> pd.DataFrame:
    """Hourly historical weather for ERCOT points. start/end: 'YYYY-MM-DD'"""
    frames = []
    for pt in ERCOT_POINTS:
        r = requests.get(
            _ARCHIVE,
            params={
                "latitude": pt["lat"],
                "longitude": pt["lon"],
                "start_date": start,
                "end_date": end,
                "hourly": _VARS,
                "timezone": "UTC",
            },
            timeout=60,
        )
        r.raise_for_status()
        df = _parse(r, is_forecast=False)
        df["latitude"] = pt["lat"]
        df["longitude"] = pt["lon"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)[
        ["timestamp", "latitude", "longitude", "temperature_c",
         "wind_speed_10m_ms", "shortwave_radiation", "is_forecast"]
    ]


def fetch_forecast() -> pd.DataFrame:
    """48-hour weather forecast for ERCOT points."""
    frames = []
    for pt in ERCOT_POINTS:
        r = requests.get(
            _FORECAST,
            params={
                "latitude": pt["lat"],
                "longitude": pt["lon"],
                "hourly": _VARS,
                "forecast_days": 2,
                "timezone": "UTC",
            },
            timeout=30,
        )
        r.raise_for_status()
        df = _parse(r, is_forecast=True)
        df["latitude"] = pt["lat"]
        df["longitude"] = pt["lon"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)[
        ["timestamp", "latitude", "longitude", "temperature_c",
         "wind_speed_10m_ms", "shortwave_radiation", "is_forecast"]
    ]
