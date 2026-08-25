import os
import pickle

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from tensorflow import keras

from ingest.weather import REGION_TZ

from .features import WIND_FEATURE_COLS, _weather_cond

ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "..", "models", "artifacts")


def _load_forecast_weather(conn: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    cond = _weather_cond(region)
    return conn.execute(f"""
        SELECT timestamp,
               AVG(temperature_c)       AS temperature_c,
               AVG(wind_speed_10m_ms)   AS wind_speed_10m_ms,
               AVG(shortwave_radiation) AS shortwave_radiation
        FROM weather
        WHERE is_forecast = TRUE AND ({cond})
        GROUP BY timestamp
        ORDER BY timestamp
    """).df()


def _load_recent_generation(conn: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    return conn.execute(f"""
        SELECT timestamp,
               SUM(CASE WHEN fuel_type = 'WND' THEN generation_mw END) AS wind_mw,
               SUM(CASE WHEN fuel_type = 'SUN' THEN generation_mw END) AS solar_mw
        FROM generation
        WHERE region = '{region}'
          AND fuel_type IN ('WND', 'SUN')
        GROUP BY timestamp
        ORDER BY timestamp DESC
        LIMIT 200
    """).df()


def predict(db_path: str = "gridpulse.duckdb", region: str = "ERCO") -> pd.DataFrame:
    conn = duckdb.connect(db_path, read_only=True)
    weather = _load_forecast_weather(conn, region)
    recent = _load_recent_generation(conn, region)
    conn.close()

    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True)
    recent["timestamp"] = pd.to_datetime(recent["timestamp"], utc=True)
    recent = recent.sort_values("timestamp").set_index("timestamp")

    tz = REGION_TZ[region]

    rows = []
    for _, row in weather.iterrows():
        ts = row["timestamp"]
        local = ts.tz_convert(tz)
        hour = local.hour

        def _lag(col, h):
            key = ts - pd.Timedelta(hours=h)
            return recent[col].get(key, np.nan)

        ws = row["wind_speed_10m_ms"]
        rows.append({
            "timestamp": ts,
            "hour": hour,
            "month": local.month,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "wind_speed_10m_ms": ws,
            "wind_speed_sq": ws ** 2,
            "wind_speed_cu": ws ** 3,
            "wind_lag_24": _lag("wind_mw", 24),
            "wind_lag_168": _lag("wind_mw", 168),
            "solar_lag_168": _lag("solar_mw", 168),
        })

    df = pd.DataFrame(rows)

    # --- wind forecast ---
    wind_feats = df[WIND_FEATURE_COLS].copy()
    missing = wind_feats.isna().any(axis=1).sum()
    if missing:
        print(f"  [wind predict] {missing} rows have missing lag values"
              " -- filling with column mean")
        wind_feats = wind_feats.fillna(wind_feats.mean())

    with open(os.path.join(ARTIFACTS, f"wind_val_mae_{region}.pkl"), "rb") as f:
        wind_meta = pickle.load(f)

    if wind_meta["winner"] == "lightgbm":
        wind_model = lgb.Booster(model_file=os.path.join(ARTIFACTS, f"wind_model_{region}.lgb"))
        wind_preds = wind_model.predict(wind_feats.values)
    else:
        wind_model = keras.models.load_model(os.path.join(ARTIFACTS, f"wind_model_{region}.keras"))
        with open(os.path.join(ARTIFACTS, f"wind_scaler_{region}.pkl"), "rb") as f:
            scaler = pickle.load(f)
        wind_preds = wind_model.predict(scaler.transform(wind_feats.values), verbose=0).flatten()

    wind_preds = np.clip(wind_preds, 0, None)

    # solar: naive lag_168 (concept drift makes a model unreliable across all regions)
    solar_preds = df["solar_lag_168"].fillna(df["solar_lag_168"].mean()).values
    solar_preds = np.clip(solar_preds, 0, None)

    return pd.DataFrame({
        "timestamp": df["timestamp"],
        "wind_forecast_mw": wind_preds,
        "solar_forecast_mw": solar_preds,
    })
