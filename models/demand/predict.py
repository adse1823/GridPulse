import os
import pickle

import duckdb
import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd
from tensorflow import keras

from .features import FEATURE_COLS

ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "..", "models", "artifacts")


def _load_forecast_weather(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute("""
        SELECT timestamp,
               AVG(temperature_c)     AS temperature_c,
               AVG(wind_speed_10m_ms) AS wind_speed_10m_ms
        FROM weather
        WHERE is_forecast = TRUE
        GROUP BY timestamp
        ORDER BY timestamp
    """).df()


def _load_recent_demand(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute("""
        SELECT timestamp, demand_mw
        FROM demand
        WHERE region = 'ERCO'
        ORDER BY timestamp DESC
        LIMIT 200
    """).df()


def predict(db_path: str = "gridpulse.duckdb") -> pd.DataFrame:
    conn = duckdb.connect(db_path, read_only=True)
    weather = _load_forecast_weather(conn)
    recent = _load_recent_demand(conn)
    conn.close()

    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True)
    recent["timestamp"] = pd.to_datetime(recent["timestamp"], utc=True)
    recent = recent.sort_values("timestamp").set_index("timestamp")

    us_holidays = holidays.US(years=range(2024, 2027))

    rows = []
    for _, row in weather.iterrows():
        ts = row["timestamp"]
        local = ts.tz_convert("US/Central")
        hour = local.hour

        def _lag(h):
            key = ts - pd.Timedelta(hours=h)
            return recent["demand_mw"].get(key, np.nan)

        rows.append({
            "timestamp": ts,
            "hour": hour,
            "day_of_week": local.dayofweek,
            "month": local.month,
            "is_holiday": int(local.date() in us_holidays),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "demand_lag_24": _lag(24),
            "demand_lag_48": _lag(48),
            "demand_lag_168": _lag(168),
            "temperature_c": row["temperature_c"],
            "temperature_c_sq": row["temperature_c"] ** 2,
            "wind_speed_10m_ms": row["wind_speed_10m_ms"],
        })

    df = pd.DataFrame(rows)
    missing = df[FEATURE_COLS].isna().any(axis=1).sum()
    if missing:
        print(f"  [demand predict] {missing} rows have missing lag values — filling with mean")
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].mean())

    X = df[FEATURE_COLS].values

    with open(os.path.join(ARTIFACTS, "demand_val_mae.pkl"), "rb") as f:
        meta = pickle.load(f)

    if meta["winner"] == "lightgbm":
        model = lgb.Booster(model_file=os.path.join(ARTIFACTS, "demand_model.lgb"))
        preds = model.predict(X)
    else:
        model = keras.models.load_model(os.path.join(ARTIFACTS, "demand_model.keras"))
        with open(os.path.join(ARTIFACTS, "demand_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        preds = model.predict(scaler.transform(X), verbose=0).flatten()

    return pd.DataFrame({"timestamp": df["timestamp"], "demand_forecast_mw": preds})
