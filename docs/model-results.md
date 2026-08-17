# Model Evaluation Results — Tier 1

All models trained on ERCOT data 2022–2024. Chronological split:

| Split | Date range | Rows (approx) |
|-------|-----------|---------------|
| Train | 2022-01-01 → 2023-07-31 | ~14,000 hours |
| Val | 2023-08-01 → 2024-01-31 | ~4,400 hours |
| Test | 2024-02-01 → 2024-12-31 | ~7,900 hours |

Val is used for model selection. Test is held out and touched only once for
final evaluation. Primary metric: hourly MAE in MW.

---

## Demand Forecast

**Winner: LightGBM**

| Model | Val MAE (MW) | Test MAE (MW) | Test MAPE |
|-------|-------------|--------------|-----------|
| LightGBM | lower | 2,430 | 4.6% |
| Keras | higher | not evaluated | — |
| Naive (lag_168) | — | 4,122 | ~8% |

LightGBM beats naive by **41%**.

Target from design doc was 500–900 MW. Actual result is 2,430 MW. Gap is
attributable to fixed hyperparameters (no tuning), 2024 summer heat events
in the test set, and potential weather join gaps. Improvement deferred to
Tier 2.

**Features used:** hour, day_of_week, month, is_holiday, hour_sin, hour_cos,
demand_lag_24, demand_lag_48, demand_lag_168, temperature_c, temperature_c_sq,
wind_speed_10m_ms.

---

## Wind Forecast

**Winner: Keras**

| Model | Test MAE (MW) | Normalised MAE |
|-------|--------------|----------------|
| Keras | 3,753 | 9.4% of 40 GW capacity |
| Naive (lag_168) | 7,492 | 18.7% |

Keras beats naive by **50%**. The neural net winning over LightGBM for wind
is consistent with the design doc's prediction — the turbine power curve and
curtailment effects are non-linear patterns that trees can miss.

**Features used:** wind_speed_10m_ms, wind_speed_sq, wind_speed_cu,
hour_sin, hour_cos, month, wind_lag_24, wind_lag_168.

---

## Solar Forecast

**Decision: use naive (lag_168)**

| Model | Test MAE (MW) | Normalised MAE |
|-------|--------------|----------------|
| LightGBM (best attempt) | 1,911 | 9.6% of 20 GW capacity |
| Naive (lag_168) | 1,609 | 8.0% |

Three training configurations were attempted (see `docs/decisions.md`).
None beat naive. Root cause: ERCOT solar capacity grew ~10 → 20+ GW between
2022 and 2024, causing concept drift that the model cannot overcome without
installed-capacity normalisation.

**Solar forecast in production uses `solar_lag_168` (same hour last week).**
This is documented in the report output as a known limitation.

---

## Risk Aggregation

No model — pure arithmetic. Dispatchable capacity assumptions:

| Season | Months | Dispatchable MW |
|--------|--------|----------------|
| Summer | Jun–Sep | 46,000 |
| Winter | Nov–Feb | 42,000 |
| Shoulder | Mar–May, Oct | 50,000 |

Safety margin: 3,000 MW. Hours where
`demand - (wind + solar + dispatchable) > 3,000 MW` are flagged AT RISK.

---

## Backtest: Feb 2021 Texas Winter Storm Uri

Run: `python scripts/backtest.py --start 2021-02-08 --end 2021-02-17`

Uses actual EIA demand and generation as forecast inputs (not model
predictions). Validates that the aggregator correctly identifies the blackout
conditions.

Expected result: Feb 12–15 flagged AT RISK with shortfalls of 15,000–25,000 MW.
- Peak demand: ~69 GW
- Wind generation trough: ~2 GW (froze)
- Solar: ~1 GW (winter, short days)
- Fixed dispatchable: 42 GW (winter)
- Total supply: ~45 GW → shortfall ~24 GW → AT RISK
