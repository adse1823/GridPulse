import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_BASE = "https://api.eia.gov/v2/electricity/rto"
_REGION = "ERCO"
_PAGE = 5000


def _api_key() -> str:
    key = os.environ.get("EIA_API_KEY", "")
    if not key:
        raise EnvironmentError("EIA_API_KEY not set — add it to .env")
    return key


def _paginate(url: str, params: dict) -> list[dict]:
    rows, offset = [], 0
    while True:
        r = requests.get(url, params={**params, "offset": offset}, timeout=30)
        r.raise_for_status()
        data = r.json()["response"]["data"]
        if not data:
            break
        rows.extend(data)
        if len(data) < _PAGE:
            break
        offset += len(data)
    return rows


def fetch_demand(start: str, end: str) -> pd.DataFrame:
    """Hourly ERCOT demand from EIA. start/end: 'YYYY-MM-DD'"""
    rows = _paginate(
        f"{_BASE}/region-data/data/",
        {
            "api_key": _api_key(),
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": _REGION,
            "facets[type][]": "D",
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": _PAGE,
        },
    )
    df = pd.DataFrame(rows)[["period", "value"]].rename(
        columns={"period": "timestamp", "value": "demand_mw"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")
    df["region"] = _REGION
    return df[["timestamp", "region", "demand_mw"]]


def fetch_generation(start: str, end: str) -> pd.DataFrame:
    """Hourly ERCOT generation by fuel type from EIA. start/end: 'YYYY-MM-DD'"""
    rows = _paginate(
        f"{_BASE}/fuel-type-data/data/",
        {
            "api_key": _api_key(),
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": _REGION,
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": _PAGE,
        },
    )
    df = pd.DataFrame(rows)[["period", "fueltype", "value"]].rename(
        columns={"period": "timestamp", "fueltype": "fuel_type", "value": "generation_mw"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["generation_mw"] = pd.to_numeric(df["generation_mw"], errors="coerce")
    df["region"] = _REGION
    return df[["timestamp", "region", "fuel_type", "generation_mw"]]
