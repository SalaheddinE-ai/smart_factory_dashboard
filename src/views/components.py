from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..utils import (binned_mean, explain, investigation_table, latest_per_machine, need, sample, show, table)


def _machine_ids(df):
    return sorted(df["machine_id"].astype(str).unique())


def render(df: pd.DataFrame, ctx: dict):
    st.header("Machine Health & Component Analysis")
    comps = ctx["comps"]
    if not comps:
        st.warning("No component health columns (names ending in '_health') detected.")
        return
    if not need(df, "machine_id"):
        return
    last = latest_per_machine(df)
    st.caption(f"Components detected: {', '.join(comps)}. Latest observation per machine is chosen by timestamp.")

    c1, c2 = st.columns(2)
    with c1:
        m = sample(df[comps], 20000).melt(var_name="component", value_name="health")
        show(px.box(m, x="component", y="health", title="Component health distributions"))
    with c2:
        mm = df[comps].mean().reset_index()
        mm.columns = ["component", "mean_health"]
        show(px.bar(mm.sort_values("mean_health"), x="mean_health", y="component", orientation="h", title="Mean component health"))
    explain("Distribution and mean of each component's health value.", "Whether lower-health components cause machine-level failures.")

    st.subheader("Component heatmap across machines")
    mode = st.radio("Machines shown", ["Lowest overall health (latest)", "Manual selection"], horizontal=True)
    if mode.startswith("Lowest"):
        n = st.slider("Number of machines", 5, 60, 25)
        key = "machine_health_score" if "machine_health_score" in last else comps[0]
        sub = last.nsmallest(n, key)
    else:
        pick = st.multiselect("Machines", _machine_ids(last), default=_machine_ids(last)[:10])
        sub = last[last["machine_id"].astype(str).isin(pick)]
    if len(sub):
        z = sub.set_index(sub["machine_id"].astype(str))[comps]
        show(px.imshow(z, aspect="auto", color_continuous_scale="RdYlGn", title="Latest component health per machine"), max(300, 18 * len(z) + 120))
        explain("Latest component values for the selected machines.", "Trajectory — only the latest record is shown here.")

    st.subheader("Machine profile (radar) and detail view")
    ids = _machine_ids(df)
    mid = st.selectbox("Machine", ids, key="comp_machine")
    md = df[df["machine_id"].astype(str) == mid].sort_values("timestamp") if "timestamp" in df else df[df["machine_id"].astype(str) == mid]
    lm = md.iloc[-1]
    fleet = df[comps].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[lm[c] for c in comps] + [lm[comps[0]]], theta=comps + [comps[0]], fill="toself", name=f"{mid} (latest)"))
    fig.add_trace(go.Scatterpolar(r=list(fleet.values) + [fleet.iloc[0]], theta=comps + [comps[0]], name="Filtered-fleet mean"))
    fig.update_layout(title="Component health profile")
    show(fig, 450)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Latest recorded condition**")
        keep = [c for c in md.columns if not c.startswith("_") and c not in ("age_group", "dsm_group", "health_band")]
        table(lm[keep].astype(str).rename("value").reset_index().rename(columns={"index": "field"}))
    with c2:
        if "timestamp" in md:
            ev = [c for c in ["machine_health_score"] + comps if c in md]
            fig = px.line(md, x="timestamp", y=ev, title=f"Health evolution — machine {mid}", markers=True)
            show(fig, 380)
    mc = [c for c in ["timestamp", "maintenance_type", "days_since_maintenance", "maintenance_count", "maintenance_cost",
                      "last_repair_duration_hours", "replacement_parts_count", "previous_failure_count"] if c in md]
    if len(mc) > 1:
        st.markdown("**Maintenance history (as recorded in this machine's rows)**")
        table(md[mc])

    st.subheader("Components vs machine health and failure probability")
    cc = st.selectbox("Component", comps)
    c1, c2 = st.columns(2)
    for col, tgt, ttl in [(c1, "machine_health_score", "machine health score"), (c2, "failure_probability", "failure probability")]:
        with col:
            if need(df, tgt):
                d = df[[cc, tgt]].dropna()
                fig = px.scatter(sample(d), x=cc, y=tgt, opacity=0.25, render_mode="webgl", title=f"{cc} vs {ttl}")
                bm = binned_mean(d, cc, tgt, 25)
                if len(bm):
                    fig.add_trace(go.Scatter(x=bm.x, y=bm.y, mode="lines+markers", name="Binned mean", line=dict(color="black")))
                show(fig)
                st.caption(f"Spearman ρ = {d[cc].corr(d[tgt], method='spearman'):.2f}")
    explain("Association between one component and the machine-level score / probability.",
            "Whether the machine score is derived from components (likely in synthetic data) — an association would then be built in.")

    st.subheader("Machines requiring further investigation")
    inv = investigation_table(last, ctx)
    if inv["investigation_index"].isna().all():
        st.info("No risk columns available to build the index.")
        return
    topn = st.slider("Rows", 10, 200, 25)
    cols = [c for c in ["machine_id", "timestamp", "investigation_index", "machine_health_score", "failure_probability",
                        "remaining_useful_life_hours", "maintenance_priority"] + comps if c in inv]
    table(inv.sort_values("investigation_index", ascending=False).head(topn)[cols])
    st.caption("Investigation index = mean percentile rank of available risk signals "
               f"({inv['signals_used'].iloc[0]}) on each machine's latest record. A screening heuristic for triage, not a model, "
               "not a diagnosis, and not a repair recommendation.")
