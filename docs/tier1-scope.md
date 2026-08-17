# Tier 1 Scope

## What we are building

A pipeline that answers one question:

> **For the next 24–48 hours, is ERCOT at risk of a supply shortfall?**

A shortfall means forecasted electricity demand exceeds the supply that can
physically be dispatched. When this happens, operators face rolling blackouts
or emergency imports. The Feb 2021 Texas event is the canonical example —
demand spiked from a cold snap while wind turbines froze and gas plants
tripped offline.

The pipeline has four steps:

```
Ingest -> Demand Forecast -> Renewable Forecast -> Risk Aggregation -> Report
```

Tier 1 implements all four in their simplest working form. No agents, no LLM,
no cloud. A script you can run locally that produces a risk flag and a plain-
text explanation.

---

## Region: ERCOT (Texas)

ERCOT operates the grid for ~90% of Texas (~26 million customers). We use it
for Tier 1 because:

- The Feb 2021 blackouts are the most documented US grid shortfall in recent
  history — gives the project a concrete, testable narrative
- ERCOT is electrically isolated (not interconnected with neighboring grids),
  making the supply/demand math cleaner than PJM or MISO
- EIA provides clean hourly historical demand + generation data going back to
  2015 via free API

ERCOT's grid at a glance (approximate 2024 figures):

| Metric | Value |
|--------|-------|
| Peak demand (summer) | ~85 GW |
| Peak demand (winter) | ~65 GW |
| Installed wind capacity | ~40 GW |
| Installed solar capacity | ~20 GW |
| Dispatchable (gas + nuclear + coal) | ~55 GW |
| Annual demand | ~430 TWh |

---

## What Tier 1 produces

**Input:** today's date

**Output:**
```
2024-01-15 18:00 CST — AT RISK
  Demand forecast:  58,200 MW
  Supply forecast:  54,800 MW  (wind 8,400 + solar 200 + dispatchable 46,200)
  Shortfall:        3,400 MW
  Driver:           Cold snap + low wind. Similar to Feb 2021 pattern.

2024-01-15 19:00 CST — AT RISK
  ...

2024-01-16 14:00 CST — OK
  Demand forecast:  48,100 MW
  Supply forecast:  61,300 MW
  Margin:           13,200 MW
```

The report is a plain template string in Tier 1 (no LLM). The LLM narrative
comes in Tier 2.

---

## What Tier 1 is NOT

| Out of scope for Tier 1 | Why |
|-------------------------|-----|
| Multiple regions | Need one working end-to-end first |
| LLM/RAG reporting | Adds complexity before core is proven |
| LangGraph agents | Orchestration layer comes after models work |
| FastAPI / dashboard | Serving layer is Tier 2 |
| Airflow / Kubernetes | Scheduling is Tier 3 |
| Real-time data feeds | Historical + 48h forecast is enough for Tier 1 |
| Substation-level analysis | Needs distribution grid topology data not publicly available |

---

## Definition of done (Tier 1)

- [x] EIA + weather data ingested into DuckDB for 2022–2024
- [x] Demand forecast model trained, evaluated — LightGBM vs Keras, winner chosen
- [x] Renewable forecast model trained, evaluated — same process
- [x] Risk aggregation logic produces correct at-risk flags
- [x] Plain-text report generated from flags
- [x] pytest coverage for all four components
- [x] GitHub Actions CI: lint (ruff) + tests pass on push

**Completed: 2026-08-16. See `docs/decisions.md` and `docs/model-results.md`.**
