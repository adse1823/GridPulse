import pandas as pd
import pytest
from ingest.schema import init_db
from ingest.load import _upsert


def _demand_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "region": "ERCO",
        "demand_mw": [40000.0 + i * 100 for i in range(n)],
    })


def test_upsert_inserts_rows(tmp_path):
    conn = init_db(str(tmp_path / "test.duckdb"))
    df = _demand_df(3)
    _upsert(conn, "demand", df, ["timestamp", "region"])
    count = conn.execute("SELECT COUNT(*) FROM demand").fetchone()[0]
    assert count == 3
    conn.close()


def test_upsert_no_duplicates(tmp_path):
    conn = init_db(str(tmp_path / "test.duckdb"))
    df = _demand_df(3)
    _upsert(conn, "demand", df, ["timestamp", "region"])
    _upsert(conn, "demand", df, ["timestamp", "region"])
    count = conn.execute("SELECT COUNT(*) FROM demand").fetchone()[0]
    assert count == 3  # idempotent


def test_upsert_updates_value(tmp_path):
    conn = init_db(str(tmp_path / "test.duckdb"))
    df = _demand_df(1)
    _upsert(conn, "demand", df, ["timestamp", "region"])

    updated = df.copy()
    updated["demand_mw"] = 99999.0
    _upsert(conn, "demand", updated, ["timestamp", "region"])

    val = conn.execute("SELECT demand_mw FROM demand").fetchone()[0]
    assert val == 99999.0
    conn.close()
