import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

_BASE = "https://api.eia.gov/v2/electricity/rto"
_PAGE = 5000


def _api_key() -> str:
    key = os.environ.get("EIA_API_KEY", "")
    if not key:
        raise EnvironmentError("EIA_API_KEY not set — add it to .env")
    return key


def _paginate(url: str, params: dict) -> list[dict]:
    rows, offset = [], 0
    while True:
        r = requests.get(url, params={**params, "offset": offset}, timeout=120)
        r.raise_for_status()
        data = r.json()["response"]["data"]
        if not data:
            break
        rows.extend(data)
        if len(data) < _PAGE:
            break
        offset += len(data)
    return rows


def fetch_demand(start: str, end: str, region: str = "ERCO") -> pd.DataFrame:
    """Hourly demand from EIA for the given region. start/end: 'YYYY-MM-DD'"""
    rows = _paginate(
        f"{_BASE}/region-data/data/",
        {
            "api_key": _api_key(),
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region,
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
    df["region"] = region
    return df[["timestamp", "region", "demand_mw"]]


def fetch_generation(start: str, end: str, region: str = "ERCO") -> pd.DataFrame:
    """Hourly generation by fuel type from EIA for the given region. start/end: 'YYYY-MM-DD'"""
    rows = _paginate(
        f"{_BASE}/fuel-type-data/data/",
        {
            "api_key": _api_key(),
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region,
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
    df["region"] = region
    return df[["timestamp", "region", "fuel_type", "generation_mw"]]
