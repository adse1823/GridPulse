from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ingest.weather import REGION_POINTS

ALL_REGIONS = list(REGION_POINTS.keys())  # ["ERCO", "CISO", "PJM", "NYIS"]


class HeadroomState(TypedDict):
    db_path: str
    regions: list[str]      # regions to rank
    ranking: list[dict]     # sorted metrics, most headroom first
    ranking_table: str      # formatted text table


def _rank_all(state: HeadroomState) -> dict:
    from agents.headroom_ranker.ranker import compute_metrics, format_ranking, rank_regions
    from agents.risk_aggregator.aggregator import aggregate
    from models.demand.predict import predict as demand_predict
    from models.renewable.predict import predict as renewable_predict

    metrics = []
    for region in state["regions"]:
        print(f"[headroom] {region} ...")
        demand_df = demand_predict(state["db_path"], region)
        renewable_df = renewable_predict(state["db_path"], region)
        risk_df = aggregate(demand_df, renewable_df, region)
        m = compute_metrics(risk_df, region)
        metrics.append(m)
        print(
            f"  median headroom: {m['median_headroom_mw']:+,.0f} MW  "
            f"at_risk: {m['at_risk_hours']}/{m['total_hours']} hours"
        )

    ranking = rank_regions(metrics)
    return {"ranking": ranking, "ranking_table": format_ranking(ranking)}


def build_headroom_graph():
    g = StateGraph(HeadroomState)
    g.add_node("rank_all", _rank_all)
    g.add_edge(START, "rank_all")
    g.add_edge("rank_all", END)
    return g.compile()


headroom_graph = build_headroom_graph()


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "gridpulse.duckdb"
    regions = sys.argv[2:] if len(sys.argv) > 2 else ALL_REGIONS
    result = headroom_graph.invoke({"db_path": db, "regions": regions})
    print(result["ranking_table"])
