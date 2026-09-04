import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.graph import build_graph
from agents.headroom_graph import ALL_REGIONS, build_headroom_graph
from ingest.weather import REGION_TZ

st.set_page_config(
    page_title="GridPulse",
    page_icon="⚡",
    layout="wide",
)

# Build graphs once per session
@st.cache_resource
def get_graphs():
    return build_graph(), build_headroom_graph()


# ---------- sidebar ----------

st.sidebar.title("⚡ GridPulse")
db_path = st.sidebar.text_input("DuckDB path", value=os.getenv("GRIDPULSE_DB", "gridpulse.duckdb"))
tab_choice = st.sidebar.radio("View", ["Risk Report", "Headroom Ranking"])

# ---------- Risk Report tab ----------

if tab_choice == "Risk Report":
    st.title("Risk Report")
    region = st.pills("Region", ALL_REGIONS, default="ERCO", selection_mode="single")
    run = st.sidebar.button("Run Report", type="primary")

    if run:
        graph, _ = get_graphs()
        with st.spinner("Running pipeline..."):
            result = graph.invoke({"db_path": db_path, "region": region})

        risk_df = result["risk_df"]
        report = result.get("report", "")
        narrative = result.get("llm_narrative", "")

        at_risk = int(risk_df["at_risk"].sum())
        total = len(risk_df)
        peak_shortfall = risk_df["shortfall_mw"].max()

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("AT RISK Hours", f"{at_risk} / {total}")
        col2.metric("Peak Shortfall", f"{peak_shortfall:,.0f} MW")
        col3.metric(
            "Status",
            "⚠️ AT RISK" if at_risk > 0 else "✅ All Clear",
        )

        st.divider()

        # Supply vs demand chart
        local_ts = risk_df["timestamp"].dt.tz_convert(REGION_TZ[region])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=local_ts, y=risk_df["demand_forecast_mw"],
            name="Demand Forecast", line=dict(color="#ef4444", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=local_ts, y=risk_df["total_supply_mw"],
            name="Total Supply", line=dict(color="#22c55e", width=2),
        ))
        fig.add_trace(go.Bar(
            x=local_ts[risk_df["at_risk"]],
            y=risk_df.loc[risk_df["at_risk"], "demand_forecast_mw"],
            name="AT RISK hour", marker_color="rgba(239,68,68,0.15)",
            yaxis="y",
        ))
        fig.update_layout(
            title="Demand vs Supply — 48h Forecast",
            xaxis_title="Time (local)",
            yaxis_title="MW",
            legend=dict(orientation="h", y=-0.2),
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Breakdown chart
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=local_ts, y=risk_df["wind_forecast_mw"],
            name="Wind", stackgroup="supply", line=dict(width=0),
            fillcolor="rgba(59,130,246,0.6)",
        ))
        fig2.add_trace(go.Scatter(
            x=local_ts, y=risk_df["solar_forecast_mw"],
            name="Solar", stackgroup="supply", line=dict(width=0),
            fillcolor="rgba(234,179,8,0.6)",
        ))
        fig2.add_trace(go.Scatter(
            x=local_ts, y=risk_df["dispatchable_mw"],
            name="Dispatchable", stackgroup="supply", line=dict(width=0),
            fillcolor="rgba(107,114,128,0.5)",
        ))
        fig2.add_trace(go.Scatter(
            x=local_ts, y=risk_df["demand_forecast_mw"],
            name="Demand", line=dict(color="#ef4444", width=2, dash="dot"),
        ))
        fig2.update_layout(
            title="Supply Breakdown vs Demand",
            xaxis_title="Time (local)",
            yaxis_title="MW",
            legend=dict(orientation="h", y=-0.2),
            height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # LLM narrative
        if narrative:
            st.subheader("Analysis")
            st.info(narrative)

        # Plain text report
        with st.expander("Full text report"):
            st.code(report, language=None)

        # Raw table
        with st.expander("Per-hour risk table"):
            display_df = risk_df.copy()
            display_df["timestamp"] = local_ts.dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(
                display_df[[
                    "timestamp", "demand_forecast_mw", "wind_forecast_mw",
                    "solar_forecast_mw", "dispatchable_mw",
                    "total_supply_mw", "shortfall_mw", "at_risk",
                ]].style.apply(
                    lambda row: ["background-color: #fef2f2" if row["at_risk"] else ""
                                 for _ in row],
                    axis=1,
                ),
                use_container_width=True,
            )
    else:
        st.info("Select a region and click **Run Report** to generate the 48h forecast.")

# ---------- Headroom Ranking tab ----------

else:
    regions = st.sidebar.multiselect("Regions to rank", ALL_REGIONS, default=ALL_REGIONS)
    run = st.sidebar.button("Run Ranking", type="primary")

    st.title("Headroom Ranking")
    st.caption("Regions ranked by median spare capacity over the next 48 hours.")

    if run and regions:
        _, headroom_graph = get_graphs()
        with st.spinner("Running pipeline for all regions..."):
            result = headroom_graph.invoke({"db_path": db_path, "regions": regions})

        ranking = result["ranking"]
        ranking_df = pd.DataFrame(ranking)

        # Bar chart
        colors = ["#22c55e" if v >= 0 else "#ef4444"
                  for v in ranking_df["median_headroom_mw"]]
        fig = go.Figure(go.Bar(
            x=ranking_df["region"],
            y=ranking_df["median_headroom_mw"],
            marker_color=colors,
            text=ranking_df["median_headroom_mw"].apply(lambda v: f"{v:+,.0f} MW"),
            textposition="outside",
        ))
        fig.update_layout(
            title="Median Spare Capacity — 48h Forecast",
            yaxis_title="MW (positive = surplus)",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Ranking table
        display = ranking_df.copy()
        display["median_headroom_mw"] = display["median_headroom_mw"].apply(
            lambda v: f"{v:+,.0f} MW"
        )
        display["min_headroom_mw"] = display["min_headroom_mw"].apply(
            lambda v: f"{v:+,.0f} MW"
        )
        display["pct_hours_ok"] = display["pct_hours_ok"].apply(lambda v: f"{v:.0f}%")
        display["at_risk"] = display.apply(
            lambda r: f"{r['at_risk_hours']} / {r['total_hours']}", axis=1
        )
        display = display.rename(columns={
            "region": "Region",
            "median_headroom_mw": "Median Headroom",
            "min_headroom_mw": "Min Headroom",
            "pct_hours_ok": "Hours OK",
            "at_risk": "AT RISK",
        })[["Region", "Median Headroom", "Min Headroom", "Hours OK", "AT RISK"]]

        st.dataframe(display, use_container_width=True, hide_index=True)

        with st.expander("Text table"):
            st.code(result["ranking_table"], language=None)

    elif run and not regions:
        st.warning("Select at least one region.")
    else:
        st.info("Select regions and click **Run Ranking**.")
