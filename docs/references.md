# External References

Sources used for research, corpus building, and implementation decisions.

---

## EIA API

| Resource | URL | Used for |
|----------|-----|----------|
| EIA API Technical Documentation | https://www.eia.gov/opendata/documentation.php | API structure, pagination, facet params |
| EIA Codes Reference (Cleanview) | https://docs.cleanview.co/api-reference/eia-codes | Confirmed RTO respondent codes |

### Confirmed EIA Respondent Codes

| Region | Code | Notes |
|--------|------|-------|
| ERCOT (Texas) | `ERCO` | Tier 1 — built |
| CAISO (California) | `CISO` | Tier 2 |
| PJM (Mid-Atlantic) | `PJM` | Tier 2 |
| NYISO (New York) | `NYIS` | Tier 2 |
| MISO (Midwest) | `MISO` | Optional Tier 2 |
| ISO New England | `ISNE` | Optional Tier 2 |

All use the same `/v2/electricity/rto/region-data/` and
`/v2/electricity/rto/fuel-type-data/` endpoints with `facets[respondent][]`
as the region filter.

---

## LangGraph

| Resource | URL | Notes |
|----------|-----|-------|
| LangGraph StateGraph Reference | https://reference.langchain.com/python/langgraph/graph/state/StateGraph | Current v1.x API — stable since 1.0 LTS |
| LangGraph Graph API Overview | https://docs.langchain.com/oss/python/langgraph/graph-api | Nodes, edges, compile, invoke |
| LangGraph Basics Tutorial | https://shafiqulai.github.io/blogs/blog_8.html | StateGraph + TypedDict pattern |

**Current version:** v1.x (stable). Key imports:
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
```

---

## RAG Corpus — Grid Incident Documents

Documents to use as the retrieval corpus for the LLM reporting agent.
Chunked and embedded with sentence-transformers, stored in Chroma.

### Primary (Tier 2)

| Document | URL | Format | Why |
|----------|-----|--------|-----|
| FERC/NERC Feb 2021 Texas Cold Weather Outages Report | https://www.ferc.gov/media/february-2021-cold-weather-outages-texas-and-south-central-united-states-ferc-nerc-and | PDF (~300 pages) | Primary incident report: demand spike, wind/gas failures, blackout mechanics |
| UT Austin Feb 2021 Texas Blackout Timeline | https://energy.utexas.edu/sites/default/files/UTAustin%20(2021)%20EventsFebruary2021TexasBlackout%2020210714.pdf | PDF | Hour-by-hour timeline — cleaner for chunking than the 300-page FERC report |
| NERC 2021–2022 Winter Reliability Assessment | https://www.nerc.com/globalassets/programs/rapa/ra/nerc_wra_2021.pdf | PDF | Broader context: other regions, seasonal risk patterns |

### To add when CAISO region is live (Tier 2)

| Document | Notes |
|----------|-------|
| CAISO 2020 heat dome report | California Aug 2020 rolling blackouts — demand record, supply shortfall |
| NERC 2023 Summer Reliability Assessment | Covers CAISO, PJM, MISO risk profiles |

---

## Other Research

| Resource | URL | Notes |
|----------|-----|-------|
| FERC RTOs and ISOs overview | https://www.ferc.gov/power-sales-and-markets/rtos-and-isos | Background on RTO structure and coverage areas |
| EIA Hourly Electric Grid Monitor | https://www.eia.gov/todayinenergy/detail.php?id=40993 | Covers ISNE, NYIS, PJM, MISO, SWPP, ERCO, CISO |
