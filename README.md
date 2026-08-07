# GridWatch

*(placeholder name — rename freely)*

A grid shortfall early-warning system: forecasts electricity demand and renewable
generation 24–48 hours ahead, compares them against known dispatchable capacity,
and flags periods at risk of a supply shortfall — the same mechanism behind real
events like the Texas Feb 2021 blackouts and California heat-wave rolling
blackouts. It also ranks tracked regions by spare capacity margin, surfacing
where new demand could be added with the most headroom.

## Why this project

Electricity demand and supply must balance in real time — grid-scale storage is
still limited, so operators schedule generation against *forecasted*, not actual,
demand. Unlike a problem with a closed-form physical answer, demand and renewable
output depend on weather, calendar effects, and behavior with no equation that
predicts them from first principles. That's a genuine, well-established
time-series ML problem, not a place ML is bolted on for its own sake — the
project was scoped by first asking, for each piece, whether the ML/infra choice
is load-bearing or decorative. (See `docs/scope-decisions.md` for the reasoning
trail, including approaches considered and rejected.)

## How it works

1. **Ingest** — pull historical + forecast weather (Open-Meteo/NOAA) and
   historical demand/generation-mix data (EIA API, grid operator APIs) into
   DuckDB.
2. **Forecast demand** — a small feedforward TensorFlow/Keras model, trained on
   lag features (same hour yesterday/last week), forecast weather, and calendar
   features (hour, day-of-week, holiday). Benchmarked against a
   LightGBM/XGBoost baseline — the neural net only earns its complexity if it
   actually beats the tree model.
3. **Forecast renewable output** — same shape, separate model, driven by
   wind/irradiance forecasts instead of temperature.
4. **Aggregate risk** — plain arithmetic, deliberately not ML: if forecasted
   demand exceeds forecasted supply (renewables + known dispatchable capacity)
   within a safety margin, flag the hour as at-risk. Decision logic stays simple
   and auditable on top of the two learned forecasts.
5. **Report** — an LLM call (grounded via RAG over historical grid incident
   write-ups) turns a flagged risk into a plain-language explanation, e.g.
   *"Tomorrow 5–8pm: demand forecast 42GW vs. supply forecast 38GW, driven by a
   cold snap and low wind — resembles the pattern behind the Feb 2021 Texas
   event."*
6. **Rank headroom** *(Tier 2)* — the flip side of step 4: instead of flagging
   where supply might fall short, rank tracked regions by how much spare
   margin they typically carry (forecasted supply consistently and comfortably
   above forecasted demand). Reuses the same per-region demand/renewable
   forecasts — no new model, just a different comparison. Requires the
   ingestion/forecast steps to run across multiple tracked regions/Balancing
   Authorities rather than just one, since ranking needs something to rank
   *across*.

Agents: `demand-forecast` → `renewable-forecast` → `risk-aggregator` →
`headroom-ranker` → `reporting`, orchestrated with LangGraph. Each owns one job
and a defined input/output contract, not shared global state.

## Data sources

| Source | Provides | Link |
|---|---|---|
| EIA API | Historical/near-real-time demand, generation mix by region | https://www.eia.gov/opendata/ |
| Grid operators (CAISO/ERCOT/PJM/NYISO) | Regional demand, generation, price | public dashboards/APIs per operator |
| Open-Meteo / NOAA | Historical + forecast weather | https://open-meteo.com/ |

## Roadmap

**Tier 1 — core pipeline (build first, must fully work before anything else)**
- [ ] EIA + weather ingestion → DuckDB
- [ ] Demand forecast model (Keras) + LightGBM baseline
- [ ] Renewable forecast model (Keras)
- [ ] Risk-aggregation logic
- [ ] Reporting as plain template string (no LLM yet)
- [ ] pytest coverage
- [ ] GitHub Actions CI (lint + test on push)

**Tier 2 — agentize + LLM/RAG**
- [ ] Split into LangGraph-orchestrated agents
- [ ] Extend ingestion/forecasting to multiple tracked regions/Balancing
      Authorities (needed for headroom ranking to have something to rank across)
- [ ] Headroom-ranker agent — ranks regions by spare capacity margin (Option A:
      reuses existing forecasts, regional aggregation only — not
      substation/feeder-level hosting capacity, see scope-decisions doc)
- [ ] LLM-based reporting agent
- [ ] RAG over historical grid incident write-ups (HF sentence-transformer
      embeddings + Chroma)
- [ ] FastAPI serving layer
- [ ] Streamlit dashboard
- [ ] Dockerize each agent

**Tier 3 — ops, scheduling, cloud (incremental, non-blocking)**
- [ ] Airflow DAG for the ingest → forecast → aggregate → report pipeline
- [ ] Kubernetes CronJobs (local cluster first, then cloud)
- [ ] AWS deployment (SageMaker and/or Lambda)
- [ ] Terraform provisioning
- [ ] MLflow experiment tracking
- [ ] Evidently AI drift detection on the demand model
- [ ] SHAP explanations feeding the LLM narrative
- [ ] Grafana + Prometheus (+ CloudWatch) monitoring
- [ ] CI/CD scheduled retrain, gated on held-out performance vs. current model

**Deliberately out of scope** — see scope-decisions doc for why: Snowflake,
Kafka/Redpanda, Spark, Flink, AWS Glue, dbt (no genuine need at this data
volume/velocity); PyTorch (redundant with TF for this problem); Azure ML
(duplicate cloud platform); FinBERT, LIME, true A/B testing (don't fit this
problem's shape). PyTorch Geometric / GNN-based cascading-failure modeling on
grid topology is a real, separate future project — not a phase of this one.
Same applies to true substation/feeder-level hosting-capacity analysis
("where can a new data center physically connect") — that needs distribution
grid topology data that isn't public at a usable national scale (a few states
publish utility-specific hosting-capacity maps, inconsistent coverage/format).
The headroom-ranker in Tier 2 answers a narrower, buildable question — which
*regions* have the most spare margin, not which physical grid nodes do.

## Repo structure (planned)

```
GridWatch/
├── ingest/            # EIA + weather pulls, DuckDB loading
├── models/
│   ├── demand/
│   └── renewable/
├── agents/
│   ├── demand_forecast/
│   ├── renewable_forecast/
│   ├── risk_aggregator/
│   ├── headroom_ranker/   # Tier 2
│   └── reporting/
├── api/                # FastAPI serving layer (Tier 2)
├── dashboard/          # Streamlit app (Tier 2)
├── infra/              # Docker, k8s manifests, Terraform (Tier 2-3)
├── tests/
└── docs/
    └── scope-decisions.md
```

## Status

Planning complete. Tier 1 not yet started.
