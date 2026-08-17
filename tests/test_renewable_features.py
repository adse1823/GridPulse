import numpy as np
import pandas as pd

from models.renewable.features import SOLAR_FEATURE_COLS, WIND_FEATURE_COLS, split, split_solar


def _make_df(start: str, periods: int) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    all_cols = list(set(WIND_FEATURE_COLS + SOLAR_FEATURE_COLS))
    df = pd.DataFrame({"timestamp": ts})
    for col in all_cols:
        df[col] = rng.random(periods)
    df["wind_mw"] = rng.random(periods) * 15_000
    df["solar_mw"] = rng.random(periods) * 8_000
    df["wind_lag_168"] = df["wind_mw"].shift(168).fillna(0)
    df["solar_lag_168"] = df["solar_mw"].shift(168).fillna(0)
    return df


def test_split_solar_train_starts_2023():
    df = _make_df("2022-01-01", 26_000)
    train, _, _ = split_solar(df)
    train_start = train["timestamp"].dt.tz_convert("US/Central").dt.date.astype(str).min()
    assert train_start >= "2023-01-01"


def test_split_solar_excludes_2022_from_train():
    df = _make_df("2022-01-01", 26_000)
    train, _, _ = split_solar(df)
    dates = train["timestamp"].dt.tz_convert("US/Central").dt.year
    assert (dates == 2022).sum() == 0


def test_split_and_split_solar_same_test_set():
    df = _make_df("2022-01-01", 26_000)
    _, _, test_full = split(df)
    _, _, test_solar = split_solar(df)
    assert set(test_full["timestamp"]) == set(test_solar["timestamp"])


def test_wind_features_include_power_curve():
    assert "wind_speed_sq" in WIND_FEATURE_COLS
    assert "wind_speed_cu" in WIND_FEATURE_COLS


def test_solar_features_no_lags():
    assert "solar_lag_24" not in SOLAR_FEATURE_COLS
    assert "solar_lag_168" not in SOLAR_FEATURE_COLS
