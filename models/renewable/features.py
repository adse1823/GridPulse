import duckdb
import numpy as np
import pandas as pd

from ingest.weather import REGION_POINTS, REGION_TZ

_TRAIN_END = "2023-07-31"
_VAL_END = "2024-01-31"
_SOLAR_TRAIN_START = "2023-01-01"  # solar capacity grew too fast; 2022 data misleads the model

WIND_FEATURE_COLS = [
    "wind_speed_10m_ms", "wind_speed_sq", "wind_speed_cu",
    "hour_sin", "hour_cos", "month",
    "wind_lag_24", "wind_lag_168",
]
SOLAR_FEATURE_COLS = [
    "shortwave_radiation", "temperature_c",
    "hour_sin", "hour_cos", "month",
]
WIND_TARGET = "wind_mw"
SOLAR_TARGET = "solar_mw"


def _weather_cond(region: str) -> str:
    pts = REGION_POINTS[region]
    return " OR ".join(
        f"(latitude = {p['lat']} AND longitude = {p['lon']})" for p in pts
    )


def _load_generation(conn: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    return conn.execute(f"""
        SELECT
            timestamp,
            SUM(CASE WHEN fuel_type = 'WND' THEN generation_mw END) AS wind_mw,
            SUM(CASE WHEN fuel_type = 'SUN' THEN generation_mw END) AS solar_mw
        FROM generation
        WHERE region = '{region}'
          AND fuel_type IN ('WND', 'SUN')
        GROUP BY timestamp
        ORDER BY timestamp
    """).df()


def _load_weather(conn: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    cond = _weather_cond(region)
    return conn.execute(f"""
        SELECT timestamp,
               AVG(temperature_c)       AS temperature_c,
               AVG(wind_speed_10m_ms)   AS wind_speed_10m_ms,
               AVG(shortwave_radiation) AS shortwave_radiation
        FROM weather
        WHERE is_forecast = FALSE AND ({cond})
        GROUP BY timestamp
        ORDER BY timestamp
    """).df()


def build_features(db_path: str = "gridpulse.duckdb", region: str = "ERCO") -> pd.DataFrame:
    conn = duckdb.connect(db_path, read_only=True)
    gen = _load_generation(conn, region)
    weather = _load_weather(conn, region)
    conn.close()

    gen["timestamp"] = pd.to_datetime(gen["timestamp"], utc=True)
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True)

    df = gen.merge(weather, on="timestamp", how="left")
    df = df.sort_values("timestamp").reset_index(drop=True)

    tz = REGION_TZ[region]
    local = df["timestamp"].dt.tz_convert(tz)
    df["hour"] = local.dt.hour
    df["month"] = local.dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["wind_speed_sq"] = df["wind_speed_10m_ms"] ** 2
    df["wind_speed_cu"] = df["wind_speed_10m_ms"] ** 3

    df["wind_lag_24"] = df["wind_mw"].shift(24)
    df["wind_lag_168"] = df["wind_mw"].shift(168)
    df["solar_lag_24"] = df["solar_mw"].shift(24)
    df["solar_lag_168"] = df["solar_mw"].shift(168)

    wind_required = WIND_FEATURE_COLS + [WIND_TARGET]
    solar_required = SOLAR_FEATURE_COLS + [SOLAR_TARGET]
    df = df.dropna(subset=list(set(wind_required + solar_required))).reset_index(drop=True)

    return df


def split(
    df: pd.DataFrame, region: str = "ERCO"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tz = REGION_TZ[region]
    local_date = df["timestamp"].dt.tz_convert(tz).dt.date.astype(str)
    train = df[local_date <= _TRAIN_END].copy()
    val = df[(local_date > _TRAIN_END) & (local_date <= _VAL_END)].copy()
    test = df[local_date > _VAL_END].copy()
    return train, val, test


def split_solar(
    df: pd.DataFrame, region: str = "ERCO"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Solar-specific split: training starts 2023-01-01 to avoid low-capacity 2022 data."""
    tz = REGION_TZ[region]
    local_date = df["timestamp"].dt.tz_convert(tz).dt.date.astype(str)
    train = df[(local_date >= _SOLAR_TRAIN_START) & (local_date <= _TRAIN_END)].copy()
    val = df[(local_date > _TRAIN_END) & (local_date <= _VAL_END)].copy()
    test = df[local_date > _VAL_END].copy()
    return train, val, test
