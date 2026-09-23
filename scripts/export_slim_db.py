"""
Export a slim inference-only duckdb for HF Spaces deployment.

The full gridpulse.duckdb contains years of historical data used for training.
At inference time the pipeline only needs:
  - weather:     forecast rows (is_forecast = TRUE)
  - demand:      last 200 rows per region (covers all lag features up to lag_168)
  - generation:  last 200 timestamps per region for WND + SUN fuel types

Run after every forecast refresh:
    python scripts/export_slim_db.py
"""

import argparse
import os

import pandas as pd
import duckdb

from ingest.schema import init_db

REGIONS = ["ERCO", "CISO", "PJM", "NYIS"]


def export(src: str, dst: str) -> None:
    if os.path.exists(dst):
        os.remove(dst)

    src_conn = duckdb.connect(src, read_only=True)

    # --- extract slim slices ---

    weather_df = src_conn.execute(
        "SELECT * FROM weather WHERE is_forecast = TRUE"
    ).df()

    demand_parts = [
        src_conn.execute(f"""
            SELECT * FROM demand
            WHERE region = '{r}'
            ORDER BY timestamp DESC
            LIMIT 200
        """).df()
        for r in REGIONS
    ]
    demand_df = pd.concat(demand_parts, ignore_index=True)

    gen_parts = [
        src_conn.execute(f"""
            SELECT * FROM generation
            WHERE region = '{r}'
              AND fuel_type IN ('WND', 'SUN')
              AND timestamp IN (
                  SELECT DISTINCT timestamp FROM generation
                  WHERE region = '{r}' AND fuel_type IN ('WND', 'SUN')
                  ORDER BY timestamp DESC
                  LIMIT 200
              )
        """).df()
        for r in REGIONS
    ]
    gen_df = pd.concat(gen_parts, ignore_index=True)

    src_conn.close()

    # --- write slim db ---
    dst_conn = init_db(dst)
    dst_conn.execute("INSERT INTO weather SELECT * FROM weather_df")
    dst_conn.execute("INSERT INTO demand SELECT * FROM demand_df")
    dst_conn.execute("INSERT INTO generation SELECT * FROM gen_df")
    dst_conn.close()

    src_mb = os.path.getsize(src) / 1e6
    dst_mb = os.path.getsize(dst) / 1e6
    print(f"Full db:  {src_mb:.1f} MB  ({src})")
    print(f"Slim db:  {dst_mb:.1f} MB  ({dst})")
    print(f"Rows — weather: {len(weather_df)}  demand: {len(demand_df)}  generation: {len(gen_df)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="gridpulse.duckdb")
    p.add_argument("--dst", default="gridpulse_slim.duckdb")
    args = p.parse_args()
    export(args.src, args.dst)
