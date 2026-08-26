import pandas as pd

from ingest.weather import REGION_TZ

# Fixed seasonal dispatchable capacity per region (MW)
# Conservative: installed dispatchable capacity minus ~15% typical planned outage rate
# Winter is lower for ERCO because cold snaps trip unweatherized gas plants (Feb 2021)
_REGION_DISPATCHABLE_MW: dict[str, dict[str, int]] = {
    "ERCO": {"summer": 46_000, "winter": 42_000, "shoulder": 50_000},
    "CISO": {"summer": 38_000, "winter": 35_000, "shoulder": 40_000},
    "PJM":  {"summer": 130_000, "winter": 120_000, "shoulder": 135_000},
    "NYIS": {"summer": 26_000, "winter": 24_000, "shoulder": 28_000},
}

_SAFETY_MARGIN_MW = 3_000


def _season(month: int) -> str:
    if month in (6, 7, 8, 9):
        return "summer"
    if month in (11, 12, 1, 2):
        return "winter"
    return "shoulder"


def aggregate(
    demand_df: pd.DataFrame,
    renewable_df: pd.DataFrame,
    region: str = "ERCO",
) -> pd.DataFrame:
    """
    demand_df:    columns [timestamp, demand_forecast_mw]
    renewable_df: columns [timestamp, wind_forecast_mw, solar_forecast_mw]

    Returns a per-hour DataFrame with supply components, shortfall, and AT_RISK flag.
    """
    dispatchable = _REGION_DISPATCHABLE_MW[region]
    tz = REGION_TZ[region]

    df = demand_df.merge(renewable_df, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    local = df["timestamp"].dt.tz_convert(tz)
    df["dispatchable_mw"] = local.dt.month.apply(
        lambda m: dispatchable[_season(m)]
    )
    df["renewable_mw"] = df["wind_forecast_mw"] + df["solar_forecast_mw"]
    df["total_supply_mw"] = df["renewable_mw"] + df["dispatchable_mw"]
    df["shortfall_mw"] = df["demand_forecast_mw"] - df["total_supply_mw"]
    df["at_risk"] = df["shortfall_mw"] > _SAFETY_MARGIN_MW

    return df[[
        "timestamp",
        "demand_forecast_mw",
        "wind_forecast_mw",
        "solar_forecast_mw",
        "renewable_mw",
        "dispatchable_mw",
        "total_supply_mw",
        "shortfall_mw",
        "at_risk",
    ]]


def run(db_path: str = "gridpulse.duckdb", region: str = "ERCO") -> pd.DataFrame:
    from models.demand.predict import predict as demand_predict
    from models.renewable.predict import predict as renewable_predict

    print(f"Running demand forecast ({region}) ...")
    demand_df = demand_predict(db_path, region)
    print(f"  {len(demand_df)} hours forecast")

    print(f"Running renewable forecast ({region}) ...")
    renewable_df = renewable_predict(db_path, region)
    print(f"  {len(renewable_df)} hours forecast")

    print("Aggregating risk ...")
    risk_df = aggregate(demand_df, renewable_df, region)
    at_risk = risk_df["at_risk"].sum()
    print(f"  {at_risk} of {len(risk_df)} hours flagged AT RISK")

    return risk_df


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "gridpulse.duckdb"
    df = run(db)
    cols = ["timestamp", "demand_forecast_mw", "total_supply_mw", "shortfall_mw", "at_risk"]
    print(df[cols].to_string(index=False))
