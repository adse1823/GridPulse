import pandas as pd
import duckdb
from .schema import init_db
from .eia import fetch_demand, fetch_generation
from .weather import fetch_historical, fetch_forecast


def _upsert(conn: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, conflict_cols: list[str]) -> int:
    tmp = f"_tmp_{table}"
    conn.execute(f"CREATE TEMP TABLE IF NOT EXISTS {tmp} AS SELECT * FROM df LIMIT 0")
    conn.execute(f"DELETE FROM {tmp}")
    conn.execute(f"INSERT INTO {tmp} SELECT * FROM df")
    conflict = ", ".join(conflict_cols)
    cols = ", ".join(df.columns)
    set_clause = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in df.columns if c not in conflict_cols
    )
    conn.execute(f"""
        INSERT INTO {table} ({cols})
        SELECT {cols} FROM {tmp}
        ON CONFLICT ({conflict}) DO UPDATE SET {set_clause}
    """)
    return len(df)


def run_historical(start: str, end: str, db_path: str = "gridpulse.duckdb") -> dict:
    conn = init_db(db_path)

    print(f"[demand]     fetching {start} -> {end} ...")
    n_demand = _upsert(conn, "demand", fetch_demand(start, end), ["timestamp", "region"])
    print(f"[demand]     {n_demand} rows")

    print(f"[generation] fetching {start} -> {end} ...")
    n_gen = _upsert(conn, "generation", fetch_generation(start, end), ["timestamp", "region", "fuel_type"])
    print(f"[generation] {n_gen} rows")

    print("[weather]    fetching historical ...")
    n_wx = _upsert(conn, "weather", fetch_historical(start, end), ["timestamp", "latitude", "longitude"])
    print(f"[weather]    {n_wx} rows")

    conn.close()
    return {"demand": n_demand, "generation": n_gen, "weather": n_wx}


def run_forecast_weather(db_path: str = "gridpulse.duckdb") -> dict:
    conn = init_db(db_path)

    print("[weather]    fetching 48h forecast ...")
    n_wx = _upsert(conn, "weather", fetch_forecast(), ["timestamp", "latitude", "longitude"])
    print(f"[weather]    {n_wx} rows")

    conn.close()
    return {"weather": n_wx}
