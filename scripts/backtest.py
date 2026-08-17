"""
Backtest the risk aggregator against a known historical event.

Uses actual EIA demand + generation as stand-ins for model forecasts,
then runs through the aggregator to show which hours would be flagged AT RISK.

Default window: Feb 8–17, 2021 (Texas winter storm Uri — the blackout event).

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --start 2021-02-08 --end 2021-02-17
"""

import argparse
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ingest.eia import fetch_demand, fetch_generation
from agents.risk_aggregator.aggregator import aggregate
from agents.reporting.report import generate


_BLACKOUT_CONTEXT = """
CONTEXT — Feb 2021 Texas Winter Storm Uri
  Feb 10: temperatures begin dropping, demand surges
  Feb 11: ERCOT issues conservation appeal
  Feb 12: rolling blackouts begin (~3am CST)
  Feb 13-17: sustained blackouts, ~4.5 million homes without power
  Peak demand: ~69 GW  |  Supply trough: ~45 GW (gas plants froze)
"""


def run(start: str, end: str) -> None:
    print(f"Fetching EIA actuals for {start} → {end} ...")

    demand_raw = fetch_demand(start, end)
    gen_raw = fetch_generation(start, end)

    # Build demand_df in aggregator format
    demand_df = demand_raw[["timestamp", "demand_mw"]].rename(
        columns={"demand_mw": "demand_forecast_mw"}
    )

    # Build renewable_df: pivot WND and SUN from generation table
    wind = (
        gen_raw[gen_raw["fuel_type"] == "WND"][["timestamp", "generation_mw"]]
        .rename(columns={"generation_mw": "wind_forecast_mw"})
    )
    solar = (
        gen_raw[gen_raw["fuel_type"] == "SUN"][["timestamp", "generation_mw"]]
        .rename(columns={"generation_mw": "solar_forecast_mw"})
    )
    renewable_df = wind.merge(solar, on="timestamp", how="outer").fillna(0)

    print(f"  {len(demand_df)} demand hours, {len(renewable_df)} renewable hours\n")

    risk_df = aggregate(demand_df, renewable_df)

    at_risk = risk_df["at_risk"].sum()
    total = len(risk_df)
    print(f"AT-RISK HOURS: {at_risk} of {total} ({at_risk/total*100:.0f}%)\n")

    # Show the worst hours
    worst = (
        risk_df[risk_df["at_risk"]]
        .nlargest(10, "shortfall_mw")[
            ["timestamp", "demand_forecast_mw", "wind_forecast_mw",
             "solar_forecast_mw", "total_supply_mw", "shortfall_mw"]
        ]
    )
    if not worst.empty:
        worst = worst.copy()
        worst["timestamp"] = worst["timestamp"].dt.tz_convert("US/Central")
        print("TOP 10 WORST HOURS:")
        print(worst.to_string(index=False))
        print()

    print(generate(risk_df))

    if start <= "2021-02-17" and end >= "2021-02-10":
        print(_BLACKOUT_CONTEXT)


def main() -> None:
    parser = argparse.ArgumentParser(description="GridPulse historical backtest")
    parser.add_argument("--start", default="2021-02-08", metavar="YYYY-MM-DD")
    parser.add_argument("--end", default="2021-02-17", metavar="YYYY-MM-DD")
    args = parser.parse_args()
    run(args.start, args.end)


if __name__ == "__main__":
    main()
