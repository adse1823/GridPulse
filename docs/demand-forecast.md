# Demand Forecast Model

## Problem

Predict hourly ERCOT electricity demand (MW) for each hour in the next 24–48
hours, given weather forecasts and calendar information available at prediction
time.

This is a well-studied problem called **Short-Term Load Forecasting (STLF)**.
It is not a novel ML problem — grid operators have been doing it for decades.
The goal is to implement a clean, correct version that is honest about its
accuracy.

---

## Why demand is forecastable

Electricity demand follows strong, predictable patterns:

- **Daily cycle:** demand peaks in late afternoon (people home, AC/heat running),
  troughs at 3–4am
- **Weekly cycle:** weekdays are higher than weekends
- **Seasonal cycle:** summer peaks (AC) and winter peaks (heating) in Texas
- **Temperature sensitivity:** ERCOT demand has a strong non-linear relationship
  with temperature — below ~55°F and above ~75°F, demand rises sharply
- **Holiday effect:** demand drops on federal holidays (less commercial/industrial
  load)

These patterns make demand learnable from historical data + weather forecasts.

---

## Features

All features are constructed from data already in DuckDB.

### Calendar features

| Feature | Description | Why it matters |
|---------|-------------|----------------|
| `hour` | Hour of day (0–23) | Captures daily demand cycle |
| `day_of_week` | Day of week (0=Mon, 6=Sun) | Weekday vs weekend pattern |
| `month` | Month (1–12) | Seasonal variation |
| `is_holiday` | US federal holiday (0/1) | Demand drop on holidays |
| `hour_sin`, `hour_cos` | Sine/cosine encoding of hour | Circular encoding preserves midnight continuity |

### Lag features (demand)

Lagged demand captures autocorrelation — the strongest predictor of demand at
hour H is demand at hour H-24 and H-168 (same hour yesterday and last week).

| Feature | Lag | Why it matters |
|---------|-----|----------------|
| `demand_lag_24` | 24 hours | Same hour yesterday |
| `demand_lag_168` | 168 hours (7 days) | Same hour last week |
| `demand_lag_48` | 48 hours | Same hour two days ago |

**Important:** lag features are only valid at prediction time if we actually
have data from 24/48/168 hours ago. For a real forecast, `demand_lag_24` uses
the most recent actual demand, which we have.

### Weather features

Averaged across the 4 ERCOT grid points in the weather table (Dallas, Houston,
San Antonio, Abilene).

| Feature | Description | Why it matters |
|---------|-------------|----------------|
| `temperature_c` | 2m air temperature | Strongest single predictor of demand |
| `temperature_c_sq` | Temperature squared | Captures non-linear U-shape relationship |
| `wind_speed_10m_ms` | Wind speed | Minor direct effect on demand (evaporative cooling) |

Weather features at prediction time use the Open-Meteo 48h forecast, not
actuals — this is what makes the forecast genuinely useful.

---

## Train / Val / Test Split

**Chronological only — no random shuffle.** Shuffling time series data leaks
future information into training.

| Split | Date range | Rows (approx) |
|-------|-----------|---------------|
| Train | 2022-01-01 → 2023-07-31 | ~14,000 hours |
| Val | 2023-08-01 → 2024-01-31 | ~4,400 hours |
| Test | 2024-02-01 → 2024-12-31 | ~7,900 hours |

Val is used for hyperparameter tuning. Test is held out until final evaluation
— touching it to tune is data leakage.

---

## Models

### Model 1: LightGBM (baseline)

LightGBM is a gradient-boosted tree ensemble. For tabular time series problems
like STLF, it is frequently the strongest model — multiple academic benchmarks
show it matching or beating neural nets for hourly load forecasting.

**Why start here:**
- No normalization required
- Handles non-linear feature interactions (temperature U-shape) natively
- Fast to train (~seconds on 3 years of hourly data)
- Interpretable via feature importance

**Hyperparameters to tune on val set:**
- `num_leaves` (32–256)
- `learning_rate` (0.01–0.1)
- `min_child_samples` (20–100)
- `n_estimators` (100–1000, with early stopping on val MAE)

### Model 2: Keras feedforward neural network

A simple dense network. Only earns its place if it beats LightGBM on the
held-out test set.

**Architecture:**
```
Input (n_features,)
  -> Dense(128, relu) -> Dropout(0.2)
  -> Dense(64, relu)  -> Dropout(0.2)
  -> Dense(1)         # linear output — predicting MW directly
```

**Training details:**
- Loss: MAE (same as eval metric — consistent)
- Optimizer: Adam, lr=1e-3
- Early stopping on val MAE, patience=10 epochs
- Input features normalized to zero mean / unit variance (required for NN,
  not for LightGBM)
- Batch size: 512

**Why a simple architecture:**
ERCOT hourly demand has well-behaved patterns. Deep networks add training
instability without accuracy gains for this problem shape. Start simple.

---

## Evaluation metric

**Hourly MAE (Mean Absolute Error) in MW**

```
MAE = mean(|actual_demand - predicted_demand|)
```

Why MAE over RMSE: MAE is more interpretable ("off by X MW on average"), and
RMSE over-penalizes outliers which are often legitimate extreme weather events
— penalizing the model for missing them distorts training.

**What good looks like for ERCOT:**

| Benchmark | MAE |
|-----------|-----|
| Naive (same hour last week) | ~1,500–2,000 MW |
| Industry day-ahead (ERCOT internal) | ~300–600 MW |
| Expected from this project | ~500–900 MW |

We also report MAPE (mean absolute percentage error) for comparability with
published benchmarks, but MAE is the decision metric.

---

## Model selection rule

Train both models. Evaluate both on the held-out test set (2024-02-01 onward).
Choose whichever has lower test MAE. If LightGBM wins or ties, use LightGBM —
simpler is better when accuracy is equal.

The chosen model's weights/artifacts are saved to `models/artifacts/` and used
by the risk aggregation step.

---

## Files

```
models/demand/
├── features.py   — load DuckDB, build feature matrix, handle lag joins
├── train.py      — train LightGBM + Keras, save winner to artifacts/
└── evaluate.py   — compute MAE/MAPE on test set, print comparison table
```
