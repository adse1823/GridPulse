import os
import pickle

import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from .features import (
    SOLAR_FEATURE_COLS,
    SOLAR_TARGET,
    WIND_FEATURE_COLS,
    WIND_TARGET,
    build_features,
    split,
    split_solar,
)

ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "..", "models", "artifacts")


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _train_lgb(X_train, y_train, X_val, y_val) -> lgb.Booster:
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    params = {
        "objective": "mae",
        "metric": "mae",
        "num_leaves": 128,
        "learning_rate": 0.05,
        "min_child_samples": 50,
        "verbosity": -1,
    }
    return lgb.train(
        params, dtrain, num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )


def _train_keras(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v = scaler.transform(X_val)
    model = keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=(X_tr.shape[1],)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    model.fit(
        X_tr, y_train,
        validation_data=(X_v, y_val),
        epochs=100, batch_size=512,
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0,
    )
    return model, scaler


def _train_one(name: str, feature_cols: list, target_col: str,
               train_df, val_df, region: str) -> dict:
    X_tr = train_df[feature_cols].values
    y_tr = train_df[target_col].values
    X_v = val_df[feature_cols].values
    y_v = val_df[target_col].values

    print(f"  Training LightGBM for {name} ({region}) ...")
    lgb_model = _train_lgb(X_tr, y_tr, X_v, y_v)
    lgb_mae = _mae(y_v, lgb_model.predict(X_v))
    print(f"    LightGBM val MAE: {lgb_mae:,.0f} MW")

    print(f"  Training Keras for {name} ({region}) ...")
    keras_model, scaler = _train_keras(X_tr, y_tr, X_v, y_v)
    keras_mae = _mae(y_v, keras_model.predict(scaler.transform(X_v), verbose=0).flatten())
    print(f"    Keras     val MAE: {keras_mae:,.0f} MW")

    os.makedirs(ARTIFACTS, exist_ok=True)

    if lgb_mae <= keras_mae:
        winner = "lightgbm"
        lgb_model.save_model(os.path.join(ARTIFACTS, f"{name}_model_{region}.lgb"))
    else:
        winner = "keras"
        keras_model.save(os.path.join(ARTIFACTS, f"{name}_model_{region}.keras"))
        with open(os.path.join(ARTIFACTS, f"{name}_scaler_{region}.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    meta = {"lightgbm": lgb_mae, "keras": keras_mae, "winner": winner}
    with open(os.path.join(ARTIFACTS, f"{name}_val_mae_{region}.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"  Winner ({name}/{region}): {winner}  (LightGBM {lgb_mae:,.0f} vs Keras {keras_mae:,.0f} MW)\n")
    return meta


def train(db_path: str = "gridpulse.duckdb", region: str = "ERCO") -> dict:
    print(f"Building renewable feature matrix for {region} ...")
    df = build_features(db_path, region)

    wind_train, val_df, _ = split(df, region)
    print(f"  wind  -- train={len(wind_train):,}  val={len(val_df):,}")
    wind_meta = _train_one("wind", WIND_FEATURE_COLS, WIND_TARGET, wind_train, val_df, region)

    solar_train, solar_val, _ = split_solar(df, region)
    print(f"  solar -- train={len(solar_train):,}  val={len(solar_val):,}  (2023+ only)")
    solar_meta = _train_one("solar", SOLAR_FEATURE_COLS, SOLAR_TARGET, solar_train, solar_val, region)

    return {"wind": wind_meta, "solar": solar_meta}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("db", nargs="?", default="gridpulse.duckdb")
    p.add_argument("--region", default="ERCO")
    args = p.parse_args()
    train(args.db, args.region)
