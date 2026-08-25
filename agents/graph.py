from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class GridPulseState(TypedDict):
    db_path: str
    region: str
    demand_df: Any       # pd.DataFrame [timestamp, demand_forecast_mw]
    renewable_df: Any    # pd.DataFrame [timestamp, wind_forecast_mw, solar_forecast_mw]
    risk_df: Any         # pd.DataFrame with at_risk column
    report: str


def _demand_forecast(state: GridPulseState) -> dict:
    from models.demand.predict import predict
    print(f"[demand_forecast] {state['region']} ...")
    df = predict(state["db_path"], state["region"])
    print(f"  {len(df)} hours")
    return {"demand_df": df}


def _renewable_forecast(state: GridPulseState) -> dict:
    from models.renewable.predict import predict
    print(f"[renewable_forecast] {state['region']} ...")
    df = predict(state["db_path"], state["region"])
    print(f"  {len(df)} hours")
    return {"renewable_df": df}


def _risk_aggregate(state: GridPulseState) -> dict:
    from agents.risk_aggregator.aggregator import aggregate
    print("[risk_aggregate] ...")
    df = aggregate(state["demand_df"], state["renewable_df"])
    print(f"  {df['at_risk'].sum()} of {len(df)} hours AT RISK")
    return {"risk_df": df}


def _generate_report(state: GridPulseState) -> dict:
    from agents.reporting.report import generate
    print("[generate_report] ...")
    return {"report": generate(state["risk_df"])}


def build_graph():
    g = StateGraph(GridPulseState)

    g.add_node("demand_forecast", _demand_forecast)
    g.add_node("renewable_forecast", _renewable_forecast)
    g.add_node("risk_aggregate", _risk_aggregate)
    g.add_node("generate_report", _generate_report)

    # demand and renewable forecasts are independent — run in parallel
    g.add_edge(START, "demand_forecast")
    g.add_edge(START, "renewable_forecast")
    g.add_edge("demand_forecast", "risk_aggregate")
    g.add_edge("renewable_forecast", "risk_aggregate")
    g.add_edge("risk_aggregate", "generate_report")
    g.add_edge("generate_report", END)

    return g.compile()


graph = build_graph()


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "gridpulse.duckdb"
    region = sys.argv[2] if len(sys.argv) > 2 else "ERCO"
    result = graph.invoke({"db_path": db, "region": region})
    print(result["report"])
