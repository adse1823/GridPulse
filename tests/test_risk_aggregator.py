import pandas as pd

from agents.risk_aggregator.aggregator import _dispatchable_MW, _season, aggregate


def _demand_df(timestamps, demand_mw):
    return pd.DataFrame({"timestamp": timestamps, "demand_forecast_mw": demand_mw})


def _renewable_df(timestamps, wind_mw, solar_mw):
    return pd.DataFrame({
        "timestamp": timestamps,
        "wind_forecast_mw": wind_mw,
        "solar_forecast_mw": solar_mw,
    })


def _make_ts(n: int = 4, month: int = 7):
    return pd.date_range(f"2024-{month:02d}-15 00:00", periods=n, freq="h", tz="UTC")


# --- season logic ---

def test_season_summer():
    for m in (6, 7, 8, 9):
        assert _season(m) == "summer"


def test_season_winter():
    for m in (11, 12, 1, 2):
        assert _season(m) == "winter"


def test_season_shoulder():
    for m in (3, 4, 5, 10):
        assert _season(m) == "shoulder"


def test_dispatchable_summer_lower_than_shoulder():
    assert _dispatchable_MW(7) < _dispatchable_MW(4)


def test_dispatchable_winter_lowest():
    assert _dispatchable_MW(1) < _dispatchable_MW(7)


# --- aggregate logic ---

def test_aggregate_columns():
    ts = _make_ts()
    df = aggregate(_demand_df(ts, [50_000] * 4), _renewable_df(ts, [8_000] * 4, [2_000] * 4))
    expected = {
        "timestamp", "demand_forecast_mw", "wind_forecast_mw", "solar_forecast_mw",
        "renewable_mw", "dispatchable_mw", "total_supply_mw", "shortfall_mw", "at_risk",
    }
    assert set(df.columns) == expected


def test_aggregate_at_risk_when_shortfall_exceeds_margin():
    ts = _make_ts()
    # demand=60GW, wind=5GW, solar=1GW, dispatch=46GW → supply=52GW, shortfall=8GW → AT RISK
    df = aggregate(
        _demand_df(ts, [60_000] * 4),
        _renewable_df(ts, [5_000] * 4, [1_000] * 4),
    )
    assert df["at_risk"].all()


def test_aggregate_ok_when_supply_sufficient():
    ts = _make_ts()
    # demand=40GW, wind=12GW, solar=4GW, dispatch=46GW → supply=62GW, shortfall=-22GW → OK
    df = aggregate(
        _demand_df(ts, [40_000] * 4),
        _renewable_df(ts, [12_000] * 4, [4_000] * 4),
    )
    assert not df["at_risk"].any()


def test_aggregate_shortfall_arithmetic():
    ts = _make_ts(n=1, month=7)  # summer → 46,000 dispatchable
    demand = 60_000
    wind = 5_000
    solar = 1_000
    expected_shortfall = demand - (wind + solar + 46_000)  # 8,000 MW

    df = aggregate(
        _demand_df(ts, [demand]),
        _renewable_df(ts, [wind], [solar]),
    )
    assert abs(df["shortfall_mw"].iloc[0] - expected_shortfall) < 1


def test_aggregate_renewable_sum():
    ts = _make_ts()
    df = aggregate(
        _demand_df(ts, [50_000] * 4),
        _renewable_df(ts, [8_000] * 4, [3_000] * 4),
    )
    assert (df["renewable_mw"] == 11_000).all()


def test_aggregate_boundary_exactly_at_margin():
    ts = _make_ts(n=1, month=7)  # summer → 46,000 MW
    # shortfall = exactly 3,000 → NOT at risk (must be strictly greater)
    demand = 46_000 + 8_000 + 2_000 + 3_000  # supply=56,000, shortfall=3,000
    df = aggregate(
        _demand_df(ts, [demand]),
        _renewable_df(ts, [8_000], [2_000]),
    )
    assert not df["at_risk"].iloc[0]
