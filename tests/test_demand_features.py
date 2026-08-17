import numpy as np
import pandas as pd

from models.demand.features import FEATURE_COLS, TARGET_COL, split


def _make_df(start: str, periods: int) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"timestamp": ts})
    for col in FEATURE_COLS:
        df[col] = rng.random(periods)
    df[TARGET_COL] = rng.random(periods) * 50_000
    return df


def test_split_sizes_sum_to_total():
    df = _make_df("2022-01-01", 26_000)
    train, val, test = split(df)
    assert len(train) + len(val) + len(test) == len(df)


def test_split_no_overlap():
    df = _make_df("2022-01-01", 26_000)
    train, val, test = split(df)
    train_ts = set(train["timestamp"])
    val_ts = set(val["timestamp"])
    test_ts = set(test["timestamp"])
    assert train_ts.isdisjoint(val_ts)
    assert train_ts.isdisjoint(test_ts)
    assert val_ts.isdisjoint(test_ts)


def test_split_chronological_order():
    df = _make_df("2022-01-01", 26_000)
    train, val, test = split(df)
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()


def test_split_train_ends_before_val():
    df = _make_df("2022-01-01", 26_000)
    train, val, _ = split(df)
    train_end = train["timestamp"].dt.tz_convert("US/Central").dt.date.astype(str).max()
    val_start = val["timestamp"].dt.tz_convert("US/Central").dt.date.astype(str).min()
    assert train_end <= "2023-07-31"
    assert val_start > "2023-07-31"


def test_feature_cols_count():
    assert len(FEATURE_COLS) == 12
