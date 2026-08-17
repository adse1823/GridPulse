import os
import pickle

import lightgbm as lgb
import numpy as np
from tensorflow import keras

from .features import FEATURE_COLS, TARGET_COL, build_features, split

ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "..", "models", "artifacts")


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def evaluate(db_path: str = "gridpulse.duckdb") -> dict:
    df = build_features(db_path)
    _, _, test_df = split(df)
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values

    with open(os.path.join(ARTIFACTS, "demand_val_mae.pkl"), "rb") as f:
        meta = pickle.load(f)

    results = {}

    lgb_path = os.path.join(ARTIFACTS, "demand_model.lgb")
    if os.path.exists(lgb_path):
        m = lgb.Booster(model_file=lgb_path)
        pred = m.predict(X_test)
        results["lightgbm"] = {"mae": _mae(y_test, pred), "mape": _mape(y_test, pred)}

    keras_path = os.path.join(ARTIFACTS, "demand_model.keras")
    if os.path.exists(keras_path):
        m = keras.models.load_model(keras_path)
        with open(os.path.join(ARTIFACTS, "demand_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        pred = m.predict(scaler.transform(X_test), verbose=0).flatten()
        results["keras"] = {"mae": _mae(y_test, pred), "mape": _mape(y_test, pred)}

    print(f"\n{'Model':<12} {'Test MAE (MW)':>14} {'Test MAPE (%)':>14}")
    print("-" * 42)
    for name, r in results.items():
        tag = " <-- winner" if name == meta["winner"] else ""
        print(f"{name:<12} {r['mae']:>14,.0f} {r['mape']:>13.1f}%{tag}")

    naive_mae = _mae(y_test, test_df["demand_lag_168"].values)
    print(f"{'naive (lag168)':<12} {naive_mae:>14,.0f}  (same hour last week)")

    return results


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "gridpulse.duckdb"
    evaluate(db)
