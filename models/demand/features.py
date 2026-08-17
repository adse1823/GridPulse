import duckdb
import holidays
import numpy as np
import pandas as pd

_TRAIN_END = "2023-07-31"
_VAL_END = "2024-01-31"

FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_holiday",
    "hour_sin", "hour_cos",
    "demand_lag_24", "demand_lag_48", "demand_lag_168",
    "temperature_c", "temperature_c_sq", "wind_speed_10m_ms",
]
TARGET_COL = "demand_mw"


def _load_demand(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute("""
        SELECT timestamp, demand_mw
        FROM demand
        WHERE region = 'ERCO'
        ORDER BY timestamp
    """).df()


def _load_weather(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute("""
        SELECT timestamp,
               AVG(temperature_c)       AS temperature_c,
               AVG(wind_speed_10m_ms)   AS wind_speed_10m_ms
        FROM weather
        WHERE is_forecast = FALSE
        GROUP BY timestamp
        ORDER BY timestamp
    """).df()


def build_features(db_path: str = "gridpulse.duckdb") -> pd.DataFrame:
    conn = duckdb.connect(db_path, read_only=True)
    demand = _load_demand(conn)
    weather = _load_weather(conn)
    conn.close()

    demand["timestamp"] = pd.to_datetime(demand["timestamp"], utc=True)
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True)

    df = demand.merge(weather, on="timestamp", how="left")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # calendar
    us_holidays = holidays.US(years=range(2022, 2026))
    local = df["timestamp"].dt.tz_convert("US/Central")
    df["hour"] = local.dt.hour
    df["day_of_week"] = local.dt.dayofweek
    df["month"] = local.dt.month
    df["is_holiday"] = local.dt.date.apply(lambda d: int(d in us_holidays))
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # lag features
    df["demand_lag_24"] = df["demand_mw"].shift(24)
    df["demand_lag_48"] = df["demand_mw"].shift(48)
    df["demand_lag_168"] = df["demand_mw"].shift(168)

    # weather transforms
    df["temperature_c_sq"] = df["temperature_c"] ** 2

    # drop rows where lags are NaN (first 168 hours)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)

    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    local_date = df["timestamp"].dt.tz_convert("US/Central").dt.date.astype(str)
    train = df[local_date <= _TRAIN_END].copy()
    val = df[(local_date > _TRAIN_END) & (local_date <= _VAL_END)].copy()
    test = df[local_date > _VAL_END].copy()
    return train, val, test
