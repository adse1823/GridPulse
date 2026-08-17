import gridstatus
import pandas as pd

_ISO = gridstatus.Ercot()
_REGION = "ERCO"


def fetch_demand(start: str, end: str) -> pd.DataFrame:
    """Hourly ERCOT demand. start/end: 'YYYY-MM-DD'"""
    raw = _ISO.get_load(start=start, end=end, verbose=False)
    df = raw[["Time", "Load"]].rename(columns={"Time": "timestamp", "Load": "demand_mw"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")
    df["region"] = _REGION
    # gridstatus returns 5-min intervals — resample to hourly mean
    df = (
        df.set_index("timestamp")
        .groupby("region")["demand_mw"]
        .resample("h")
        .mean()
        .reset_index()
    )
    return df[["timestamp", "region", "demand_mw"]]


def fetch_generation(start: str, end: str) -> pd.DataFrame:
    """Hourly ERCOT generation by fuel type. start/end: 'YYYY-MM-DD'"""
    raw = _ISO.get_fuel_mix(start=start, end=end, verbose=False)

    # gridstatus returns wide format: one column per fuel type
    fuel_cols = [c for c in raw.columns if c not in ("Time", "Interval Start", "Interval End")]
    id_col = "Interval Start" if "Interval Start" in raw.columns else "Time"

    df = raw[[id_col] + fuel_cols].rename(columns={id_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.melt(id_vars="timestamp", var_name="fuel_type", value_name="generation_mw")
    df["generation_mw"] = pd.to_numeric(df["generation_mw"], errors="coerce")
    df["region"] = _REGION

    # resample to hourly if sub-hourly
    df = (
        df.set_index("timestamp")
        .groupby(["region", "fuel_type"])["generation_mw"]
        .resample("h")
        .mean()
        .reset_index()
    )
    return df[["timestamp", "region", "fuel_type", "generation_mw"]]
