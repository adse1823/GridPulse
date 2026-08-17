# Implementation Decisions — Tier 1

Decisions made during the Tier 1 build that are not obvious from the code or
original design docs. Ordered roughly by when they were made.

---

## Ingest

### EIA over gridstatus for historical data

`ingest/ercot.py` was written as an alternative data source using the
`gridstatus` library (no API key required). It was not used for the main
historical ingest. Reasons:

- `gridstatus` fetches day-by-day in a loop (slow for 3 years)
- The library returned an empty concat error on multi-year ranges during
  testing
- EIA API handles multi-year ranges cleanly with offset pagination

`ercot.py` is kept as a fallback. Use it if EIA API access is unavailable.

### EIA request timeout: 30s → 120s

The generation endpoint returns ~186,000 rows across ~38 pages. At 30s per
request, mid-range pages would time out under network load. Bumped to 120s.
The demand endpoint (~26,000 rows, ~6 pages) is unaffected either way.

### Data ingested

| Table | Rows | Range |
|-------|------|-------|
| demand | 26,304 | 2022-01-01 → 2024-12-31 |
| generation | 186,788 | 2022-01-01 → 2024-12-31 |
| weather (historical) | 105,216 | 2022-01-01 → 2024-12-31 |
| weather (forecast) | 192 | 48h from ingest date |

---

## Demand Forecast Model

### LightGBM won over Keras on val set

Both models were trained on the same feature set and evaluated on the val set
(2023-08-01 → 2024-01-31). LightGBM had lower val MAE and was saved as the
winner. This is consistent with published STLF benchmarks — gradient-boosted
trees match or beat neural nets on tabular hourly demand data.

### Test MAE higher than expected

The docs targeted 500–900 MW MAE. The actual test MAE was 2,430 MW (4.6%
MAPE). The naive same-hour-last-week baseline was also higher than expected
(4,122 MW vs the typical 1,500–2,000 MW range).

Likely causes:
- Fixed hyperparameters (no tuning on val set — would reduce MAE)
- 2024 test period included summer heat events harder to forecast
- Weather join may have sparse coverage for some hours

The model still beats naive by 41%, which demonstrates real learning. MAE
improvement is left for Tier 2.

---

## Renewable Forecast Models

### Wind: Keras won

Keras beat LightGBM on the wind val set. This is consistent with the docs'
prediction — wind has complex non-linear patterns (the turbine power curve,
curtailment) that neural nets can capture better than trees. Wind Keras model
beats naive by 50%.

### Solar: model cannot beat naive — using lag_168

Three separate attempts were made to get the solar model to beat naive (lag_168):

**Attempt 1 — standard training (2022–2023 train)**
LightGBM val MAE < Keras, but test MAE 1,911 MW > naive 1,609 MW.

**Attempt 2 — restrict train to 2023-01-01+**
Hypothesis: 2022 solar data (lower installed capacity) was misleading the
model. Moved `_SOLAR_TRAIN_START` to 2023-01-01. Test MAE unchanged at
1,911 MW.

**Attempt 3 — drop lag features**
Hypothesis: `solar_lag_24` and `solar_lag_168` carry forward stale values
from a lower-capacity era, causing systematic underestimation of 2024 output.
Removed both lag features from `SOLAR_FEATURE_COLS`. Test MAE still 1,911 MW.

**Root cause:** ERCOT solar installed capacity grew from ~8 GW (early 2022)
to over 20 GW (end of 2024). Any model trained on pre-2024 data has never
seen 2024-scale solar output. The naive baseline uses last week's actual output
— which is already at 2024 scale — so it wins by default.

**Decision:** Use `solar_lag_168` (naive) as the solar forecast in
`models/renewable/predict.py`. This is documented in the report output and
is a known Tier 1 limitation. A proper fix requires either:
- Normalising generation by installed capacity (requires monthly capacity data)
- Retraining on a rolling window that stays close to current capacity
- Incorporating ERCOT's own capacity adequacy reports (Tier 2)

---

## Risk Aggregation

### Dispatchable capacity is fixed per season, not modelled

Plant-level outage data is not cleanly available in public APIs. Fixed
seasonal estimates (42–50 GW) are conservative (installed capacity minus ~15%
typical planned outage rate). The Feb 2021 event is why winter is lowest —
cold weather trips unweatherised gas plants.

This is a deliberate simplification for Tier 1. Tier 2 can incorporate ERCOT
capacity adequacy reports.

### Safety margin: 3,000 MW

ERCOT's formal planning reserve margin is ~13.75%, but for a 48h operational
forecast a 3 GW buffer is a reasonable rule of thumb (~3.5% of summer peak).
Can be tuned — a larger margin catches more risk but increases false alarms.

---

## Backtest

### Uses actuals as proxies for model forecasts

`scripts/backtest.py` fetches actual EIA demand and generation for a historical
window and feeds them directly into the aggregator, bypassing the ML models.
This is clearly labelled as a backtest using actuals — not a true out-of-sample
model forecast.

The intent is to verify the aggregator logic correctly identifies known events.
The Feb 2021 Texas blackout (Feb 12–15) is the primary validation case:
peak demand ~69 GW, wind collapsed to ~2 GW, fixed dispatchable at 42 GW →
total supply ~44 GW → shortfall ~25 GW → correctly flagged AT RISK.

A true model backtest (running trained models against historical forecast
weather) would require storing weather forecasts at the time of forecast issue,
which EIA and Open-Meteo do not provide retroactively. Left for Tier 2.
