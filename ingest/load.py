import duckdb
import pandas as pd

from .eia import fetch_demand, fetch_generation
from .schema import init_db
from .weather import REGION_POINTS, fetch_forecast, fetch_historical

ALL_REGIONS = list(REGION_POINTS.keys())  # ["ERCO", "CISO", "PJM", "NYIS"]


def _upsert(
    conn: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, conflict_cols: list[str]
) -> int:
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


def run_historical(
    start: str,
    end: str,
    db_path: str = "gridpulse.duckdb",
    regions: list[str] | None = None,
) -> dict:
    if regions is None:
        regions = ["ERCO"]
    conn = init_db(db_path)
    totals: dict[str, int] = {"demand": 0, "generation": 0, "weather": 0}

    for region in regions:
        print(f"\n[{region}] -- fetching {start} -> {end}")

        print("  [demand]     fetching ...")
        n = _upsert(conn, "demand", fetch_demand(start, end, region), ["timestamp", "region"])
        print(f"  [demand]     {n} rows")
        totals["demand"] += n

        print("  [generation] fetching ...")
        n = _upsert(
            conn, "generation",
            fetch_generation(start, end, region),
            ["timestamp", "region", "fuel_type"],
        )
        print(f"  [generation] {n} rows")
        totals["generation"] += n

        print("  [weather]    fetching ...")
        n = _upsert(
            conn, "weather",
            fetch_historical(start, end, region),
            ["timestamp", "latitude", "longitude"],
        )
        print(f"  [weather]    {n} rows")
        totals["weather"] += n

    conn.close()
    return totals


def run_forecast_weather(
    db_path: str = "gridpulse.duckdb",
    regions: list[str] | None = None,
) -> dict:
    if regions is None:
        regions = ["ERCO"]
    conn = init_db(db_path)
    total_wx = 0

    for region in regions:
        print(f"[{region}] fetching 48h forecast ...")
        n = _upsert(
            conn, "weather",
            fetch_forecast(region),
            ["timestamp", "latitude", "longitude"],
        )
        print(f"  {n} rows")
        total_wx += n

    conn.close()
    return {"weather": total_wx}
