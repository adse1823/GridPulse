# Renewable Forecast Model

## Problem

Predict hourly ERCOT wind + solar generation (MW) for each hour in the next
24–48 hours, given weather forecasts available at prediction time.

This is structurally the same problem as demand forecasting but driven by
different physics. Wind and solar are not dispatchable — they generate whatever
the weather allows. The grid operator cannot turn them up when demand rises.
This is what makes them the uncertain side of the supply equation.

---

## Why separate models for wind and solar

Wind and solar have different physical drivers:

| | Wind | Solar |
|-|------|-------|
| Primary driver | Wind speed | Solar irradiance (GHI) |
| Pattern | Roughly continuous, peaks overnight in Texas | Strictly zero at night, peaks midday |
| Seasonality | Stronger in spring/fall in west TX | Stronger in summer |
| Variability | High — can drop from 15GW to 2GW in hours | More predictable shape, but clouds add noise |
| ERCOT installed capacity (2024) | ~40 GW | ~20 GW |

We train one model for wind output and one for solar output. In Tier 1 we sum
them for the "renewables" supply number in the risk step.

---

## Wind Generation Model

### Features

| Feature | Description |
|---------|-------------|
| `wind_speed_10m_ms` | Avg wind speed across 4 ERCOT points |
| `wind_speed_sq` | Wind speed squared — power output is roughly cubic in wind speed at low speeds |
| `wind_speed_cu` | Wind speed cubed — captures the cubic power curve region |
| `hour_sin`, `hour_cos` | Time of day (minor effect, captures maintenance patterns) |
| `month` | Season (west TX wind is strongest spring/fall) |
| `wind_lag_24` | Same hour yesterday — autocorrelation |
| `wind_lag_168` | Same hour last week |

**Why wind speed squared and cubed:**
Wind turbine power output follows a power curve — roughly cubic in wind speed
between cut-in (~3 m/s) and rated speed (~12 m/s), then flat above rated speed.
Including `wind_speed^2` and `wind_speed^3` as explicit features helps
tree models approximate this curve without needing to learn it from interaction
terms.

### Models

Same two-model approach as demand:
1. **LightGBM** — baseline
2. **Keras feedforward** — same architecture as demand model

The NN may actually compete better here than for demand. Wind has more complex
non-linear patterns (the power curve, curtailment effects) that trees can miss.
We evaluate honestly on the test set and pick the winner.

---

## Solar Generation Model

### Features

| Feature | Description |
|---------|-------------|
| `shortwave_radiation` | Avg solar irradiance across 4 ERCOT points (W/m²) |
| `temperature_c` | Panel efficiency drops at high temperatures |
| `hour_sin`, `hour_cos` | Time of day — solar is zero at night by definition |
| `month` | Seasonal variation in day length and sun angle |
| `solar_lag_24` | Same hour yesterday |
| `solar_lag_168` | Same hour last week |

**Note on nighttime hours:** solar generation is exactly 0 between sunset and
sunrise. The model will learn this naturally from irradiance=0, but we can also
zero out predictions when `shortwave_radiation < 5 W/m²` as a hard constraint
post-prediction.

### Models

Same LightGBM vs Keras comparison. For solar, LightGBM tends to be strong
because irradiance is a near-linear predictor of output (after accounting for
temperature derating) — trees handle this well.

---

## Data: where generation by fuel comes from

The `generation` table in DuckDB contains hourly generation by fuel type for
ERCOT, pulled from EIA. The fuel type codes from EIA are:

| EIA code | Fuel |
|----------|------|
| `SUN` | Solar |
| `WND` | Wind |
| `NG` | Natural gas |
| `NUC` | Nuclear |
| `COL` | Coal |
| `WAT` | Hydro |
| `OTH` | Other |

For the renewable models, we use `SUN` as the target for solar and `WND` as
the target for wind.

---

## Train / Val / Test Split

Same as demand model — chronological:

| Split | Date range |
|-------|-----------|
| Train | 2022-01-01 → 2023-07-31 |
| Val | 2023-08-01 → 2024-01-31 |
| Test | 2024-02-01 → 2024-12-31 |

---

## Evaluation metric

**Hourly MAE in MW** — same as demand, for consistency.

We also report **normalized MAE** (MAE / installed capacity) to compare wind
and solar on equal footing despite different capacity scales.

**What good looks like for ERCOT:**

| | Naive baseline | Expected from this project |
|-|---------------|---------------------------|
| Wind MAE | ~2,000–3,000 MW | ~800–1,500 MW |
| Solar MAE | ~800–1,200 MW | ~300–600 MW |

Solar is more predictable than wind, hence lower MAE.

---

## Files

```
models/renewable/
├── features.py   — build wind + solar feature matrices from DuckDB
├── train.py      — train wind model + solar model (LightGBM + Keras each)
└── evaluate.py   — MAE comparison table for wind and solar
```
