import pandas as pd


def compute_metrics(risk_df: pd.DataFrame, region: str) -> dict:
    """
    Compute headroom metrics for a single region from its risk DataFrame.

    Headroom = total_supply - demand_forecast (positive = surplus, negative = deficit).
    """
    headroom = risk_df["total_supply_mw"] - risk_df["demand_forecast_mw"]
    total_hours = len(risk_df)
    at_risk_hours = int(risk_df["at_risk"].sum())

    return {
        "region": region,
        "median_headroom_mw": float(headroom.median()),
        "min_headroom_mw": float(headroom.min()),
        "pct_hours_ok": round((total_hours - at_risk_hours) / total_hours * 100, 1),
        "at_risk_hours": at_risk_hours,
        "total_hours": total_hours,
    }


def rank_regions(region_metrics: list[dict]) -> list[dict]:
    """Sort regions by median headroom descending (most comfortable first)."""
    return sorted(region_metrics, key=lambda x: -x["median_headroom_mw"])


def format_ranking(ranking: list[dict]) -> str:
    lines = [
        "=== GridPulse Headroom Ranking ===",
        f"{'Rank':<5} {'Region':<6} {'Median Headroom':>16} {'Min Headroom':>13} "
        f"{'Hours OK':>9} {'AT RISK':>8}",
        "-" * 62,
    ]
    for i, r in enumerate(ranking, start=1):
        lines.append(
            f"{i:<5} {r['region']:<6} "
            f"{r['median_headroom_mw']:>+15,.0f} MW "
            f"{r['min_headroom_mw']:>+12,.0f} MW "
            f"{r['pct_hours_ok']:>8.0f}% "
            f"{r['at_risk_hours']:>5}/{r['total_hours']}"
        )
    lines.append("==================================")
    return "\n".join(lines)
