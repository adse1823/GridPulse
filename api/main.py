import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from agents.graph import build_graph
from agents.headroom_graph import ALL_REGIONS, build_headroom_graph

_DB_PATH = os.getenv("GRIDPULSE_DB", "gridpulse.duckdb")

# Build both graphs once at startup
_graph = None
_headroom_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _headroom_graph
    _graph = build_graph()
    _headroom_graph = build_headroom_graph()
    yield


app = FastAPI(
    title="GridPulse API",
    description="Grid supply/demand risk forecasting and headroom ranking.",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------- request / response models ----------

class ReportRequest(BaseModel):
    region: str = "ERCO"
    db_path: str = _DB_PATH


class HourRisk(BaseModel):
    timestamp: str
    demand_forecast_mw: float
    total_supply_mw: float
    shortfall_mw: float
    at_risk: bool


class ReportResponse(BaseModel):
    region: str
    at_risk_hours: int
    total_hours: int
    report: str
    llm_narrative: str
    risk_table: list[HourRisk]


class RegionHeadroom(BaseModel):
    region: str
    median_headroom_mw: float
    min_headroom_mw: float
    pct_hours_ok: float
    at_risk_hours: int
    total_hours: int


class HeadroomResponse(BaseModel):
    regions: list[str]
    ranking: list[RegionHeadroom]
    ranking_table: str


# ---------- endpoints ----------

VALID_REGIONS = set(ALL_REGIONS)


@app.post("/report", response_model=ReportResponse)
def run_report(req: ReportRequest):
    if req.region not in VALID_REGIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown region '{req.region}'. Valid: {sorted(VALID_REGIONS)}",
        )

    result = _graph.invoke({"db_path": req.db_path, "region": req.region})

    risk_df = result["risk_df"]
    risk_table = [
        HourRisk(
            timestamp=row["timestamp"].isoformat(),
            demand_forecast_mw=row["demand_forecast_mw"],
            total_supply_mw=row["total_supply_mw"],
            shortfall_mw=row["shortfall_mw"],
            at_risk=bool(row["at_risk"]),
        )
        for _, row in risk_df.iterrows()
    ]

    return ReportResponse(
        region=req.region,
        at_risk_hours=int(risk_df["at_risk"].sum()),
        total_hours=len(risk_df),
        report=result.get("report", ""),
        llm_narrative=result.get("llm_narrative", ""),
        risk_table=risk_table,
    )


@app.get("/headroom", response_model=HeadroomResponse)
def run_headroom(
    regions: list[str] = Query(default=ALL_REGIONS),
    db_path: str = Query(default=_DB_PATH),
):
    invalid = [r for r in regions if r not in VALID_REGIONS]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown regions: {invalid}. Valid: {sorted(VALID_REGIONS)}",
        )

    result = _headroom_graph.invoke({"db_path": db_path, "regions": regions})

    return HeadroomResponse(
        regions=regions,
        ranking=[RegionHeadroom(**r) for r in result["ranking"]],
        ranking_table=result["ranking_table"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
