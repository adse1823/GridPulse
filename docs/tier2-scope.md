# Tier 2 Scope

## What was built

Tier 2 agentizes the Tier 1 pipeline, extends it to four regions, adds an
LLM reporting layer grounded via RAG, and wraps everything in a FastAPI
service and Streamlit dashboard.

**Completed: 2026-08-25.**

---

## Regions

| Region | Code | Coverage |
|--------|------|----------|
| ERCOT (Texas) | `ERCO` | ~90% of Texas, ~26M customers |
| CAISO (California) | `CISO` | ~80% of California |
| PJM (Mid-Atlantic) | `PJM` | 13 states + DC, ~65M customers |
| NYISO (New York) | `NYIS` | New York state |

All four regions share the same pipeline. Models are trained and saved
separately per region (`demand_model_ERCO.lgb`, `demand_model_CISO.lgb`, etc).

---

## Graph architecture

Two LangGraph `StateGraph` objects:

### `agents/graph.py` — single-region risk report

```
START → demand_forecast  ──┐
      → renewable_forecast ┘→ risk_aggregate → generate_report → END
                                             → llm_report      → END
```

**State:** `db_path`, `region`, `demand_df`, `renewable_df`, `risk_df`,
`report` (plain text), `llm_narrative` (Claude output).

- `demand_forecast` and `renewable_forecast` run in parallel (independent).
- `generate_report` (template) and `llm_report` (Claude) run in parallel
  after `risk_aggregate`. Plain-text report is always available even if the
  LLM call fails.

### `agents/headroom_graph.py` — multi-region headroom ranking

```
START → rank_all → END
```

`rank_all` loops over all requested regions, runs demand + renewable +
aggregate for each, computes headroom metrics, and returns a sorted ranking.

**State:** `db_path`, `regions`, `ranking` (list of metric dicts),
`ranking_table` (formatted text).

---

## Components

### Ingest

No schema changes from Tier 1. Same three tables (`demand`, `generation`,
`weather`). Extended to all four regions via the `region` parameter on
`run_historical()` and `run_forecast_weather()`.

EIA ingest hardened for large region datasets:
- Page size reduced from 5,000 → 2,000 rows
- urllib3 retry adapter (5 retries, exponential backoff)
- Application-level retry loop specifically for 504 gateway timeouts

### Models

All model code is region-parameterised. Each region gets its own artifact
files. `split()` and `split_solar()` now use the correct local timezone per
region (`REGION_TZ`) instead of hardcoded `US/Central`.

Solar still uses naive `lag_168` for all regions — the concept drift problem
(rapidly growing installed capacity outpacing training data) is universal, not
ERCOT-specific.

### Risk aggregation

`aggregate()` now accepts a `region` parameter. Dispatchable capacity is
looked up from a per-region seasonal table:

| Region | Summer | Winter | Shoulder |
|--------|--------|--------|----------|
| ERCO | 46 GW | 42 GW | 50 GW |
| CISO | 38 GW | 35 GW | 40 GW |
| PJM | 130 GW | 120 GW | 135 GW |
| NYIS | 26 GW | 24 GW | 28 GW |

Safety margin remains 3,000 MW for all regions.

### RAG corpus

Documents chunked (~400 words, 80-word overlap), embedded with
`all-MiniLM-L6-v2` (local, no API key), stored in Chroma at `data/chroma/`.

| Document | Status |
|----------|--------|
| UT Austin Feb 2021 Texas Blackout Timeline | Auto-downloaded |
| NERC 2021–2022 Winter Reliability Assessment | Auto-downloaded |
| FERC/NERC Feb 2021 Texas Outages Report | Manual — URL serves HTML |

To build the corpus: `python -m agents.rag.ingest`

To add the FERC report: download the PDF manually, save to
`docs/corpus/ferc_nerc_2021_texas_outages.pdf`, re-run ingest.

### LLM reporting

One `claude-sonnet-4-6` call per pipeline run. Only fires when AT RISK hours
exist. Prompt includes the structured risk summary (peak shortfall, driver,
timestamps) and the top-5 retrieved RAG passages. Returns 3–5 sentences of
plain prose.

Requires: `ANTHROPIC_API_KEY` in `.env`.

### FastAPI

Base URL: `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/report` | POST | Run single-region risk pipeline |
| `/headroom` | GET | Run multi-region headroom ranking |
| `/health` | GET | Liveness check |
| `/docs` | GET | Interactive Swagger UI |

Start: `uvicorn api.main:app --reload`

### Streamlit dashboard

Two tabs:

**Risk Report** — pill buttons to select region, then Run Report:
- Summary metrics (AT RISK hours, peak shortfall, status)
- Demand vs supply line chart with AT RISK hours shaded
- Stacked area supply breakdown (wind / solar / dispatchable vs demand)
- LLM narrative (collapsed if Chroma not built or key not set)
- Expandable full text report and per-hour table

**Headroom Ranking** — multi-select regions, then Run Ranking:
- Bar chart of median spare capacity per region
- Ranked summary table

Start: `streamlit run dashboard/app.py`

---

## Running order (first time)

```bash
# 1. Ingest historical data for all regions (one-time, ~15 min)
gridpulse-ingest historical --start 2022-01-01 --end 2024-12-31

# 2. Pull 48h weather forecast (run daily before each report)
gridpulse-ingest forecast

# 3. Train models for CISO, PJM, NYIS (~20-40 min)
python scripts/train_all_regions.py --skip-ingest

# 4. Build RAG vector store (one-time, downloads 2 PDFs)
python -m agents.rag.ingest

# 5. Single-region risk report
python -m agents.graph gridpulse.duckdb ERCO

# 6. Multi-region headroom ranking
python -m agents.headroom_graph gridpulse.duckdb

# 7. REST API
uvicorn api.main:app --reload

# 8. Dashboard
streamlit run dashboard/app.py
```

---

## Known limitations carried forward

| Limitation | Location | Fix |
|------------|----------|-----|
| Solar uses naive lag_168 | `models/renewable/predict.py` | Normalise by installed capacity |
| Dispatchable capacity is fixed seasonal estimate | `agents/risk_aggregator/aggregator.py` | Incorporate capacity adequacy reports |
| FERC report requires manual PDF download | `docs/corpus/` | Find direct PDF link |
| No auth on FastAPI endpoints | `api/main.py` | Add API key header middleware before any non-local deployment |
