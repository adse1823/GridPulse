import os

import anthropic
import pandas as pd
from dotenv import load_dotenv

from agents.rag.retrieve import format_passages, retrieve

load_dotenv()

_SYSTEM = """\
You are a grid reliability analyst. You are given a 48-hour supply/demand risk \
forecast for a US electricity grid region and a set of passages from historical \
grid incident reports.

Your job: write a concise plain-language risk briefing (3–5 sentences). It must:
- State the risk window (dates/hours) and peak shortfall in MW
- Name the primary driver (e.g. cold snap + low wind, heat dome + low solar)
- Reference any relevant historical precedent from the provided passages
- Avoid jargon the operator doesn't already know

Do not add bullet points, headers, or formatting. Plain prose only.\
"""

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 400


def _build_risk_summary(risk_df: pd.DataFrame, region: str) -> str:
    at_risk = risk_df[risk_df["at_risk"]]
    if at_risk.empty:
        return f"No AT RISK hours in the 48-hour forecast for {region}."

    peak_idx = risk_df["shortfall_mw"].idxmax()
    peak = risk_df.loc[peak_idx]
    peak_ts = peak["timestamp"].tz_convert("US/Central").strftime("%Y-%m-%d %H:%M CST")

    wind_pct = peak["wind_forecast_mw"] / peak["demand_forecast_mw"] * 100
    solar_pct = peak["solar_forecast_mw"] / peak["demand_forecast_mw"] * 100

    drivers = []
    if peak["demand_forecast_mw"] > 55_000:
        drivers.append("high demand")
    if wind_pct < 15:
        drivers.append("low wind output")
    if solar_pct < 5:
        drivers.append("low solar output")
    driver_str = " and ".join(drivers) if drivers else "demand exceeding supply"

    return (
        f"Region: {region}\n"
        f"AT RISK hours: {len(at_risk)} of {len(risk_df)}\n"
        f"Peak shortfall: {peak['shortfall_mw']:,.0f} MW at {peak_ts}\n"
        f"  Demand forecast:     {peak['demand_forecast_mw']:,.0f} MW\n"
        f"  Wind forecast:       {peak['wind_forecast_mw']:,.0f} MW ({wind_pct:.0f}% of demand)\n"
        f"  Solar forecast:      {peak['solar_forecast_mw']:,.0f} MW ({solar_pct:.0f}% of demand)\n"
        f"  Dispatchable:        {peak['dispatchable_mw']:,.0f} MW\n"
        f"  Total supply:        {peak['total_supply_mw']:,.0f} MW\n"
        f"Primary driver: {driver_str}"
    )


def _build_query(risk_df: pd.DataFrame) -> str:
    peak_idx = risk_df["shortfall_mw"].idxmax()
    peak = risk_df.loc[peak_idx]
    wind_pct = peak["wind_forecast_mw"] / peak["demand_forecast_mw"] * 100
    solar_pct = peak["solar_forecast_mw"] / peak["demand_forecast_mw"] * 100

    parts = ["grid supply shortfall electricity demand"]
    if wind_pct < 15:
        parts.append("wind generation collapse low wind")
    if solar_pct < 5:
        parts.append("low solar generation")
    if peak["demand_forecast_mw"] > 55_000:
        parts.append("high electricity demand peak load")
    return " ".join(parts)


def generate_llm(risk_df: pd.DataFrame, region: str = "ERCO") -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment.")

    risk_summary = _build_risk_summary(risk_df, region)

    if risk_df["at_risk"].sum() == 0:
        return risk_summary

    query = _build_query(risk_df)
    passages = retrieve(query, k=5)
    context = format_passages(passages)

    user_msg = (
        f"RISK FORECAST:\n{risk_summary}\n\n"
        f"RELEVANT HISTORICAL PASSAGES:\n{context}\n\n"
        "Write the risk briefing."
    )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text
