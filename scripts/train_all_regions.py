"""
Pull historical data and train demand + renewable models for each region.

Usage:
  # Ingest new regions and train all (default: CISO PJM NYIS)
  python scripts/train_all_regions.py

  # Skip ingest if data already in DB
  python scripts/train_all_regions.py --skip-ingest

  # Train specific regions only
  python scripts/train_all_regions.py --regions CISO PJM
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ingest.load import run_historical
from models.demand.train import train as train_demand
from models.renewable.train import train as train_renewable

ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts")


def migrate_erco_artifacts() -> None:
    """Rename pre-region ERCO artifacts to the new *_ERCO.* naming convention."""
    renames = [
        ("demand_model.lgb",   "demand_model_ERCO.lgb"),
        ("demand_model.keras", "demand_model_ERCO.keras"),
        ("demand_scaler.pkl",  "demand_scaler_ERCO.pkl"),
        ("demand_val_mae.pkl", "demand_val_mae_ERCO.pkl"),
        ("wind_model.lgb",     "wind_model_ERCO.lgb"),
        ("wind_model.keras",   "wind_model_ERCO.keras"),
        ("wind_scaler.pkl",    "wind_scaler_ERCO.pkl"),
        ("wind_val_mae.pkl",   "wind_val_mae_ERCO.pkl"),
        ("solar_model.lgb",    "solar_model_ERCO.lgb"),
        ("solar_model.keras",  "solar_model_ERCO.keras"),
        ("solar_val_mae.pkl",  "solar_val_mae_ERCO.pkl"),
    ]
    migrated = 0
    for old, new in renames:
        src = os.path.join(ARTIFACTS, old)
        dst = os.path.join(ARTIFACTS, new)
        if os.path.exists(src) and not os.path.exists(dst):
            os.rename(src, dst)
            print(f"  {old} -> {new}")
            migrated += 1
    if migrated == 0:
        print("  nothing to migrate (already done or artifacts missing)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="gridpulse.duckdb")
    p.add_argument("--regions", nargs="+", default=["CISO", "PJM", "NYIS"],
                   metavar="REGION")
    p.add_argument("--start", default="2022-01-01", metavar="YYYY-MM-DD")
    p.add_argument("--end",   default="2024-12-31", metavar="YYYY-MM-DD")
    p.add_argument("--skip-ingest", action="store_true",
                   help="Skip data pull (use if data already in DB)")
    p.add_argument("--skip-erco-migration", action="store_true",
                   help="Skip renaming old ERCO artifact files")
    args = p.parse_args()

    if not args.skip_erco_migration:
        print("\n=== Migrating ERCO artifacts ===")
        migrate_erco_artifacts()

    results = {}
    for region in args.regions:
        print(f"\n{'='*40}")
        print(f"Region: {region}")
        print(f"{'='*40}")

        if not args.skip_ingest:
            print(f"\n[1/3] Ingesting {region} ({args.start} -> {args.end}) ...")
            counts = run_historical(args.start, args.end, args.db, [region])
            print(f"  demand={counts['demand']:,}  generation={counts['generation']:,}  "
                  f"weather={counts['weather']:,} rows")
        else:
            print("[1/3] Skipping ingest")

        print("\n[2/3] Training demand model ...")
        demand_meta = train_demand(args.db, region)

        print("\n[3/3] Training renewable models ...")
        renewable_meta = train_renewable(args.db, region)

        results[region] = {"demand": demand_meta, "renewable": renewable_meta}

    print(f"\n{'='*40}")
    print("SUMMARY")
    print(f"{'='*40}")
    for region, meta in results.items():
        d = meta["demand"]
        w = meta["renewable"]["wind"]
        s = meta["renewable"]["solar"]
        print(f"{region}:")
        demand_mae = min(d['lgb_val_mae'], d['keras_val_mae'])
        print(f"  demand  winner={d['winner']}  val_mae={demand_mae:,.0f} MW")
        print(f"  wind    winner={w['winner']}  val_mae={min(w['lightgbm'], w['keras']):,.0f} MW")
        print(f"  solar   winner={s['winner']}  val_mae={min(s['lightgbm'], s['keras']):,.0f} MW")


if __name__ == "__main__":
    main()
