from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..utils import (binned_mean, explain, investigation_table, latest_per_machine, need, priority_rank, priority_sorted,
                     sample, show, table)

LISTED = ["maintenance_priority", "days_since_maintenance", "maintenance_count", "maintenance_cost", "last_repair_duration_hours",
          "replacement_parts_count", "technician_experience_years", "previous_failure_count", "days_since_previous_failure", "failure_frequency"]


def _trend(df, x, y, title, xl=None, yl=None):
    d = df[[x, y]].dropna()
    if d.empty:
        st.info(f"No data for {title}.")
        return
    fig = px.scatter(sample(d), x=x, y=y, opacity=0.25, render_mode="webgl", title=title, labels={x: xl or x, y: yl or y})
    bm = binned_mean(d, x, y, 20)
    if len(bm):
        fig.add_trace(go.Scatter(x=bm.x, y=bm.y, mode="lines+markers", name="Binned mean", line=dict(color="black")))
    show(fig)
    st.caption(f"Spearman ρ = {d[x].corr(d[y], method='spearman'):.2f} (n={len(d):,})")


def render(df: pd.DataFrame, ctx: dict):
    st.header("Maintenance Intelligence")
    st.warning("All findings on this page are **observations**, not causal conclusions. No repair or replacement action is recommended.")
    present = [c for c in LISTED if c in df]
    st.caption(f"Maintenance columns found: {', '.join(present) or 'none'}")
    num = [c for c in present if c != "maintenance_priority"]
    if num:
        table(df[num].describe().T.reset_index().rename(columns={"index": "column"}).round(3))

    c1, c2 = st.columns(2)
    with c1:
        if need(df, "maintenance_cost") and "machine_type" in df:
            if "maintenance_type" in df:
                t = df.groupby(["machine_type", "maintenance_type"], observed=True)["maintenance_cost"].mean().reset_index()
                fig = px.density_heatmap(t, x="maintenance_type", y="machine_type", z="maintenance_cost", histfunc="avg", text_auto=".0f",
                                         title="Mean maintenance cost: machine type × maintenance type", color_continuous_scale="Blues")
            else:
                fig = px.box(sample(df, 30000), x="machine_type", y="maintenance_cost", title="Maintenance cost by machine type")
            show(fig)
            explain("Average recorded cost by machine type (and maintenance type when available).",
                    "Cost-effectiveness; currency/units are not stated in the CSV.")
    with c2:
        if need(df, "maintenance_cost") and "downtime_hours" in df:
            _trend(df, "maintenance_cost", "downtime_hours", "Downtime vs maintenance cost")
            explain("Co-movement of downtime and maintenance cost.", "That cost drives, or is driven by, downtime.")
        elif "last_repair_duration_hours" in df and "maintenance_cost" in df:
            _trend(df, "maintenance_cost", "last_repair_duration_hours", "Repair duration vs maintenance cost")

    c1, c2 = st.columns(2)
    with c1:
        yv = "failure_frequency" if "failure_frequency" in df else ("previous_failure_count" if "previous_failure_count" in df else None)
        if yv and "maintenance_count" in df:
            _trend(df, "maintenance_count", yv, f"{yv} vs maintenance count")
            explain("Whether machines with more maintenance have more/less failure history.",
                    "Direction of cause: troubled machines are usually maintained more, so a positive link is expected.")
        else:
            st.info("maintenance_count and failure_frequency/previous_failure_count needed.")
    with c2:
        if "remaining_useful_life_hours" in df and "days_since_maintenance" in df:
            _trend(df, "days_since_maintenance", "remaining_useful_life_hours", "RUL vs days since maintenance", "Days since maintenance", "RUL (hours)")
            explain("Relationship between maintenance recency and RUL.", "That maintenance extends RUL.")

    if "maintenance_priority" in df and "production_line" in df:
        ct = pd.crosstab(df["production_line"], df["maintenance_priority"], normalize="index")
        ct = ct[priority_sorted(ct.columns)]
        fig = px.bar(ct.reset_index().melt(id_vars="production_line", var_name="priority", value_name="share"), x="production_line",
                     y="share", color="priority", title="Maintenance priority mix by production line (row %)",
                     color_discrete_sequence=["#6BAA75", "#F2C14E", "#F78154", "#B33F40"])
        fig.update_yaxes(tickformat=".0%")
        show(fig)
        explain("Share of each priority class within each production line.", "Why lines differ.")

    st.subheader("Prioritised machine investigation table")
    if not need(df, "machine_id"):
        return
    last = latest_per_machine(df)
    inv = investigation_table(last, ctx)
    if "maintenance_priority" in inv:
        inv["priority_rank"] = priority_rank(inv["maintenance_priority"])
        inv = inv.sort_values(["priority_rank", "investigation_index"], ascending=[False, False], na_position="last")
    else:
        inv = inv.sort_values("investigation_index", ascending=False)
    topn = st.slider("Rows", 10, 200, 30, key="mt_rows")
    cols = [c for c in ["machine_id", "timestamp", "maintenance_priority", "investigation_index", "machine_health_score",
                        "failure_probability", "remaining_useful_life_hours"] + num if c in inv]
    table(inv.head(topn)[list(dict.fromkeys(cols))])
    st.caption("Sorted by labelled priority, then by a transparent screening index (see Components page). Latest record per machine by timestamp. "
               "This is triage support, not a maintenance recommendation.")
