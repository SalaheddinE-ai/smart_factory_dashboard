from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..utils import FAIL_COLORS, binned_mean, explain, need, rate_bar, rate_table, sample, show, table


def render(df: pd.DataFrame, ctx: dict):
    st.header("Temporal & Operational Analysis")
    if not need(df, "timestamp"):
        return
    d = df.dropna(subset=["timestamp"])
    if len(d) < len(df):
        st.warning(f"{len(df) - len(d):,} record(s) without a valid timestamp are excluded from time-based charts only.")
    if d.empty:
        st.info("No valid timestamps in the current selection.")
        return
    freq_lbl = st.radio("Aggregation period", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True)
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[freq_lbl]
    st.caption("Time series below aggregate across machines per period (fleet view). Individual machines are only plotted one at a time in the explorer.")

    agg = {"records": ("timestamp", "size")}
    if "_fail" in d:
        agg.update(failures=("_fail", "sum"), labelled=("_fail", "count"))
    if "machine_health_score" in d:
        agg["health"] = ("machine_health_score", "mean")
    if "remaining_useful_life_hours" in d:
        agg["rul"] = ("remaining_useful_life_hours", "mean")
    g = d.groupby(pd.Grouper(key="timestamp", freq=freq)).agg(**agg).reset_index()
    g = g[g.records > 0]

    if "_fail" in d:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(g, x="timestamp", y="failures", title=f"Failed records per period ({freq_lbl.lower()})")
            show(fig)
        with c2:
            g["rate"] = g.failures / g.labelled
            fig = px.line(g, x="timestamp", y="rate", markers=True, title="Failure rate per period (record-level)")
            fig.update_yaxes(tickformat=".1%")
            show(fig)
        explain("Failed record counts / rates per period across all filtered machines.",
                "Trends in any individual machine; rate changes can come from changes in the number of machines reporting.")
    c1, c2 = st.columns(2)
    if "health" in g:
        with c1:
            show(px.line(g, x="timestamp", y="health", markers=True, title="Mean machine health score"))
    if "rul" in g:
        with c2:
            show(px.line(g, x="timestamp", y="rul", markers=True, title="Mean remaining useful life (h)"))
    th, vb = ctx["sensors"].get("Thermal", []), ctx["sensors"].get("Vibration", [])
    pick = st.multiselect("Sensor evolution (fleet mean)", th + vb, default=(th[:1] + vb[:1]))
    cols = st.columns(2)
    for i, s in enumerate(pick):
        gs = d.groupby(pd.Grouper(key="timestamp", freq=freq))[s].mean().reset_index()
        with cols[i % 2]:
            show(px.line(gs, x="timestamp", y=s, markers=True, title=f"Mean {s} per period"), 320)

    st.subheader("Operational load vs energy")
    if "operational_load" in d and "energy_consumption" in d:
        x = d[["operational_load", "energy_consumption"]].dropna()
        s = sample(x)
        fig = px.scatter(s, x="operational_load", y="energy_consumption", opacity=0.3, render_mode="webgl", title="Operational load vs energy consumption")
        bm = binned_mean(x, "operational_load", "energy_consumption", 25)
        if len(bm):
            fig.add_trace(go.Scatter(x=bm.x, y=bm.y, mode="lines+markers", name="Binned mean", line=dict(color="black")))
        show(fig)
        explain("Relationship between load and energy use.", "Efficiency or waste — units and the load scale are not documented in the CSV.")
    else:
        st.info("operational_load and/or energy_consumption not found (check Column mapping).")

    st.subheader("Downtime, maintenance and shifts")
    c1, c2 = st.columns(2)
    with c1:
        if "downtime_hours" in d and ("production_line" in d or "machine_type" in d):
            by = [c for c in ["production_line", "machine_type"] if c in d]
            how = st.radio("Downtime statistic", ["mean", "sum"], horizontal=True)
            t = d.groupby(by, observed=True)["downtime_hours"].agg(how).reset_index()
            fig = px.bar(t, x=by[0], y="downtime_hours", color=by[1] if len(by) > 1 else None, barmode="group",
                         title=f"Downtime ({how}) by {' & '.join(by)}")
            show(fig)
            explain("Downtime column aggregated by line / type.", "Cause of downtime; the column's unit is taken as-is.")
        else:
            st.info("downtime column or line/type columns not found.")
    with c2:
        if "maintenance_type" in d:
            m = d.groupby([pd.Grouper(key="timestamp", freq=freq), "maintenance_type"], observed=True).size().reset_index(name="records")
            show(px.bar(m, x="timestamp", y="records", color="maintenance_type", title="Records by maintenance type over time"))
            explain("Distribution of the maintenance_type label over time (records, not distinct interventions).", "Actual number of maintenance events.")
        elif "days_since_maintenance" in d:
            z = d.assign(m0=(d["days_since_maintenance"] == 0)).groupby(pd.Grouper(key="timestamp", freq=freq))["m0"].sum().reset_index()
            show(px.bar(z, x="timestamp", y="m0", title="PROXY: records with days_since_maintenance = 0"))
            st.caption("Proxy only — assumes 0 days since maintenance marks a maintenance day.")
    if "shift" in d and "_fail" in d:
        c1, c2 = st.columns(2)
        with c1:
            show(rate_bar(rate_table(d, "shift"), "shift", "Failure rate by shift", 1))
        with c2:
            if "machine_health_score" in d:
                t = d.groupby("shift", observed=True)["machine_health_score"].mean().reset_index()
                show(px.bar(t, x="shift", y="machine_health_score", title="Mean health score by shift"))
        explain("Failure rate and health per shift.", "Shift effects independent of workload mix.")

    st.subheader("Machine explorer (single machine, chronological)")
    if not need(d, "machine_id"):
        return
    ids = [str(i) for i in d["machine_id"].cat.categories if (d["machine_id"] == i).any()] if hasattr(d["machine_id"], "cat") else sorted(d["machine_id"].astype(str).unique())
    mid = st.selectbox("Machine", ids)
    md = d[d["machine_id"] == mid].sort_values("timestamp")
    metrics = [c for c in ["machine_health_score", "remaining_useful_life_hours", "failure_probability"] if c in md] + ctx["all_sensors"][:0]
    extra = st.multiselect("Additional sensors", ctx["all_sensors"], default=ctx["all_sensors"][:2])
    st.caption(f"{len(md)} chronological records for machine {mid}.")
    if len(md) < 2:
        st.info("Fewer than two records for this machine in the current filters.")
    for m in metrics + extra:
        fig = px.line(md, x="timestamp", y=m, markers=True, title=f"{m} — machine {mid}")
        if "_fail" in md and m == (metrics + extra)[0]:
            f = md[md["_fail"] == 1]
            fig.add_trace(go.Scatter(x=f["timestamp"], y=f[m], mode="markers", name="Observed failure", marker=dict(color="#D62839", size=11, symbol="x")))
        show(fig, 300)
    show_cols = [c for c in ["_row", "timestamp", "_fail", "failure_type", "maintenance_priority"] + metrics + extra if c in md]
    table(md[show_cols])
