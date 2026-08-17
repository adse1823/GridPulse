import pandas as pd

from agents.reporting.report import _group_consecutive, generate


def _risk_df(timestamps, at_risk, shortfall_mw=None, demand_mw=50_000,
             wind_mw=8_000, solar_mw=2_000):
    n = len(timestamps)
    if shortfall_mw is None:
        shortfall_mw = [5_000 if r else -10_000 for r in at_risk]
    return pd.DataFrame({
        "timestamp": timestamps,
        "demand_forecast_mw": [demand_mw] * n,
        "wind_forecast_mw": [wind_mw] * n,
        "solar_forecast_mw": [solar_mw] * n,
        "renewable_mw": [wind_mw + solar_mw] * n,
        "dispatchable_mw": [46_000] * n,
        "total_supply_mw": [46_000 + wind_mw + solar_mw] * n,
        "shortfall_mw": shortfall_mw,
        "at_risk": at_risk,
    })


def _ts(n: int = 4, start: str = "2024-08-15 00:00"):
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


# --- generate ---

def test_generate_no_risk():
    ts = _ts(4)
    df = _risk_df(ts, [False] * 4)
    report = generate(df)
    assert "ALL CLEAR" in report
    assert "AT-RISK HOURS" not in report


def test_generate_has_header():
    ts = _ts(4)
    df = _risk_df(ts, [False] * 4)
    report = generate(df)
    assert "GridPulse Risk Report" in report
    assert "Forecast window" in report


def test_generate_at_risk_hours_shown():
    ts = _ts(4)
    df = _risk_df(ts, [True, True, False, False])
    report = generate(df)
    assert "AT-RISK HOURS" in report
    assert "2 of 4" in report


def test_generate_all_ok_footer():
    ts = _ts(4)
    df = _risk_df(ts, [True, True, False, False])
    report = generate(df)
    assert "All other hours: sufficient margin" in report


def test_generate_no_footer_when_all_at_risk():
    ts = _ts(2)
    df = _risk_df(ts, [True, True])
    report = generate(df)
    # "All other hours" only appears when some hours are OK
    assert "All other hours" not in report


# --- _group_consecutive ---

def test_consecutive_hours_grouped():
    ts = _ts(3)
    df = _risk_df(ts, [True, True, True])
    windows = _group_consecutive(df)
    assert len(windows) == 1
    assert windows[0]["hours"] == 3


def test_non_consecutive_hours_split():
    ts = pd.DatetimeIndex([
        pd.Timestamp("2024-08-15 00:00", tz="UTC"),
        pd.Timestamp("2024-08-15 01:00", tz="UTC"),
        pd.Timestamp("2024-08-15 06:00", tz="UTC"),  # gap
        pd.Timestamp("2024-08-15 07:00", tz="UTC"),
    ])
    df = _risk_df(ts, [True, True, True, True])
    windows = _group_consecutive(df)
    assert len(windows) == 2


def test_no_at_risk_returns_empty():
    ts = _ts(4)
    df = _risk_df(ts, [False] * 4)
    windows = _group_consecutive(df)
    assert windows == []


def test_window_peak_shortfall_is_max():
    ts = _ts(3)
    df = _risk_df(ts, [True, True, True], shortfall_mw=[3_500, 9_000, 4_200])
    windows = _group_consecutive(df)
    assert windows[0]["peak_shortfall_mw"] == 9_000
