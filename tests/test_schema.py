import pytest
import duckdb
from ingest.schema import init_db


def test_init_db_creates_tables(tmp_path):
    db = str(tmp_path / "test.duckdb")
    conn = init_db(db)
    tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    assert {"demand", "generation", "weather"} == tables
    conn.close()


def test_init_db_idempotent(tmp_path):
    db = str(tmp_path / "test.duckdb")
    init_db(db).close()
    # second call must not raise
    conn = init_db(db)
    conn.close()


def test_demand_schema(tmp_path):
    conn = init_db(str(tmp_path / "test.duckdb"))
    cols = {row[0] for row in conn.execute("DESCRIBE demand").fetchall()}
    assert cols == {"timestamp", "region", "demand_mw"}
    conn.close()


def test_generation_schema(tmp_path):
    conn = init_db(str(tmp_path / "test.duckdb"))
    cols = {row[0] for row in conn.execute("DESCRIBE generation").fetchall()}
    assert cols == {"timestamp", "region", "fuel_type", "generation_mw"}
    conn.close()


def test_weather_schema(tmp_path):
    conn = init_db(str(tmp_path / "test.duckdb"))
    cols = {row[0] for row in conn.execute("DESCRIBE weather").fetchall()}
    assert cols == {"timestamp", "latitude", "longitude", "temperature_c",
                    "wind_speed_10m_ms", "shortwave_radiation", "is_forecast"}
    conn.close()
