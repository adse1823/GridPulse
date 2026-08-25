import os
import pickle

import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from .features import FEATURE_COLS, TARGET_COL, build_features, split

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
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def _train_keras(X_train, y_train, X_val, y_val) -> tuple:
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
        epochs=100,
        batch_size=512,
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0,
    )
    return model, scaler


def train(db_path: str = "gridpulse.duckdb", region: str = "ERCO") -> dict:
    print(f"Building demand feature matrix for {region} ...")
    df = build_features(db_path, region)
    train_df, val_df, test_df = split(df, region)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df[TARGET_COL].values

    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    print("Training LightGBM ...")
    lgb_model = _train_lgb(X_train, y_train, X_val, y_val)
    lgb_val_mae = _mae(y_val, lgb_model.predict(X_val))
    print(f"  LightGBM val MAE: {lgb_val_mae:,.0f} MW")

    print("Training Keras ...")
    keras_model, scaler = _train_keras(X_train, y_train, X_val, y_val)
    keras_val_mae = _mae(y_val, keras_model.predict(scaler.transform(X_val), verbose=0).flatten())
    print(f"  Keras     val MAE: {keras_val_mae:,.0f} MW")

    os.makedirs(ARTIFACTS, exist_ok=True)

    if lgb_val_mae <= keras_val_mae:
        winner = "lightgbm"
        lgb_model.save_model(os.path.join(ARTIFACTS, f"demand_model_{region}.lgb"))
    else:
        winner = "keras"
        keras_model.save(os.path.join(ARTIFACTS, f"demand_model_{region}.keras"))
        with open(os.path.join(ARTIFACTS, f"demand_scaler_{region}.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    with open(os.path.join(ARTIFACTS, f"demand_val_mae_{region}.pkl"), "wb") as f:
        pickle.dump({"lightgbm": lgb_val_mae, "keras": keras_val_mae, "winner": winner}, f)

    print(f"\nWinner ({region}): {winner}  (LightGBM {lgb_val_mae:,.0f} vs Keras {keras_val_mae:,.0f} MW)")
    return {"winner": winner, "lgb_val_mae": lgb_val_mae, "keras_val_mae": keras_val_mae}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("db", nargs="?", default="gridpulse.duckdb")
    p.add_argument("--region", default="ERCO")
    args = p.parse_args()
    train(args.db, args.region)
