import os
import pickle

import lightgbm as lgb
import numpy as np
from tensorflow import keras

from .features import (
    SOLAR_FEATURE_COLS,
    SOLAR_TARGET,
    WIND_FEATURE_COLS,
    WIND_TARGET,
    build_features,
    split,
)

ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "..", "models", "artifacts")

# Approximate ERCOT installed capacity for normalised MAE
_WIND_CAPACITY_MW = 40_000
_SOLAR_CAPACITY_MW = 20_000


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _evaluate_one(name: str, feature_cols: list, target_col: str,
                  test_df: object, lag_col: str, capacity_mw: int) -> None:
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    with open(os.path.join(ARTIFACTS, f"{name}_val_mae.pkl"), "rb") as f:
        meta = pickle.load(f)

    print(f"\n--- {name.upper()} ---")
    print(f"{'Model':<12} {'Test MAE (MW)':>14} {'Normalised MAE':>15}")
    print("-" * 44)

    lgb_path = os.path.join(ARTIFACTS, f"{name}_model.lgb")
    if os.path.exists(lgb_path):
        pred = lgb.Booster(model_file=lgb_path).predict(X_test)
        mae = _mae(y_test, pred)
        tag = " <-- winner" if meta["winner"] == "lightgbm" else ""
        print(f"{'lightgbm':<12} {mae:>14,.0f} {mae/capacity_mw:>14.1%}{tag}")

    keras_path = os.path.join(ARTIFACTS, f"{name}_model.keras")
    if os.path.exists(keras_path):
        m = keras.models.load_model(keras_path)
        with open(os.path.join(ARTIFACTS, f"{name}_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        pred = m.predict(scaler.transform(X_test), verbose=0).flatten()
        mae = _mae(y_test, pred)
        tag = " <-- winner" if meta["winner"] == "keras" else ""
        print(f"{'keras':<12} {mae:>14,.0f} {mae/capacity_mw:>14.1%}{tag}")

    naive_mae = _mae(y_test, test_df[lag_col].values)
    norm = naive_mae / capacity_mw
    print(f"{'naive (lag168)':<12} {naive_mae:>14,.0f} {norm:>14.1%}  (same hour last week)")


def evaluate(db_path: str = "gridpulse.duckdb") -> None:
    df = build_features(db_path)
    _, _, test_df = split(df)

    _evaluate_one(
        "wind", WIND_FEATURE_COLS, WIND_TARGET, test_df, "wind_lag_168", _WIND_CAPACITY_MW
    )
    _evaluate_one(
        "solar", SOLAR_FEATURE_COLS, SOLAR_TARGET, test_df, "solar_lag_168", _SOLAR_CAPACITY_MW
    )
    print("\n  (solar retrained on 2023+ only to account for capacity growth)")


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "gridpulse.duckdb"
    evaluate(db)
