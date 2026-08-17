import pandas as pd


def _fmt_ts(ts: pd.Timestamp) -> str:
    local = ts.tz_convert("US/Central")
    return local.strftime("%Y-%m-%d %H:%M CST")


def _group_consecutive(risk_df: pd.DataFrame) -> list[dict]:
    """Group consecutive AT RISK hours into windows."""
    at_risk = risk_df[risk_df["at_risk"]].copy()
    if at_risk.empty:
        return []

    windows = []
    group_start = None
    prev_ts = None
    group_rows = []

    for _, row in at_risk.iterrows():
        ts = row["timestamp"]
        if prev_ts is None or (ts - prev_ts) > pd.Timedelta(hours=1):
            if group_start is not None:
                windows.append(_summarise(group_rows))
            group_start = ts
            group_rows = [row]
        else:
            group_rows.append(row)
        prev_ts = ts

    if group_rows:
        windows.append(_summarise(group_rows))

    return windows


def _summarise(rows: list) -> dict:
    df = pd.DataFrame(rows)
    peak_idx = df["shortfall_mw"].idxmax()
    peak_row = df.loc[peak_idx]

    wind_pct = peak_row["wind_forecast_mw"] / peak_row["demand_forecast_mw"] * 100
    solar_pct = peak_row["solar_forecast_mw"] / peak_row["demand_forecast_mw"] * 100

    drivers = []
    if peak_row["demand_forecast_mw"] > 55_000:
        drivers.append("high demand")
    if wind_pct < 15:
        drivers.append("low wind")
    if solar_pct < 5:
        drivers.append("low solar")
    driver_str = " + ".join(drivers) if drivers else "demand exceeds supply"

    return {
        "start": df["timestamp"].iloc[0],
        "end": df["timestamp"].iloc[-1],
        "hours": len(df),
        "peak_shortfall_mw": peak_row["shortfall_mw"],
        "driver": driver_str,
    }


def generate(risk_df: pd.DataFrame) -> str:
    now = pd.Timestamp.now(tz="US/Central")
    window_start = risk_df["timestamp"].min()
    window_end = risk_df["timestamp"].max()
    at_risk_count = risk_df["at_risk"].sum()
    total = len(risk_df)

    lines = [
        "=== GridPulse Risk Report ===",
        f"Generated:        {now.strftime('%Y-%m-%d %H:%M CST')}",
        f"Forecast window:  {_fmt_ts(window_start)} → {_fmt_ts(window_end)}",
        "",
    ]

    windows = _group_consecutive(risk_df)

    if not windows:
        lines.append(f"ALL CLEAR — all {total} hours have sufficient margin.")
    else:
        lines.append(f"AT-RISK HOURS ({at_risk_count} of {total}):")
        for w in windows:
            start_str = _fmt_ts(w["start"])
            end_str = _fmt_ts(w["end"])
            lines.append(
                f"  {start_str} – {end_str}  |  "
                f"Peak shortfall: {w['peak_shortfall_mw']:,.0f} MW  |  "
                f"Driver: {w['driver']}"
            )
        if at_risk_count < total:
            lines.append("")
            lines.append("All other hours: sufficient margin.")

    lines.append("=============================")
    return "\n".join(lines)


def run(db_path: str = "gridpulse.duckdb") -> str:
    from agents.risk_aggregator.aggregator import run as aggregate_run
    risk_df = aggregate_run(db_path)
    report = generate(risk_df)
    print(report)
    return report


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "gridpulse.duckdb"
    run(db)
