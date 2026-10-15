from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..utils import (FAIL_COLORS, binned_mean, explain, fail_label, insight, is_no_failure, latest_per_machine,
                     need, norm, priority_sorted, sample, show)


def _kpi(col, label, value, help_=None):
    col.metric(label, value, help=help_)


def render(df: pd.DataFrame, ctx: dict):
    st.header("Executive Overview")
    c = st.columns(6)
    n = len(df)
    has_m = "machine_id" in df
    _kpi(c[0], "Monitoring records", f"{n:,}")
    _kpi(c[1], "Unique machines", f"{df['machine_id'].nunique():,}" if has_m else "n/a")
    if "_fail" in df:
        rate = df["_fail"].mean()
        extra = ""
        if has_m:
            mm = df.groupby("machine_id", observed=True)["_fail"].max()
            extra = f"Machine-level: {mm.mean():.1%} of machines have ≥1 failed record."
        _kpi(c[2], "Failure rate (record-level)", f"{rate:.2%}", extra or None)
    else:
        _kpi(c[2], "Failure rate", "n/a")
    _kpi(c[3], "Avg health score", f"{df['machine_health_score'].mean():.1f}" if "machine_health_score" in df else "n/a")
    _kpi(c[4], "Avg RUL (h)", f"{df['remaining_useful_life_hours'].mean():,.0f}" if "remaining_useful_life_hours" in df else "n/a")
    if has_m and "maintenance_priority" in df:
        last = latest_per_machine(df)
        crit = int(last["maintenance_priority"].astype("object").map(lambda v: norm(v) == "critical").sum())
        _kpi(c[5], "Machines at Critical priority", f"{crit:,}", "Machines whose latest record (by timestamp) is labelled Critical.")
    else:
        _kpi(c[5], "Critical machines", "n/a")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if need(df, "_fail"):
            vc = fail_label(df).value_counts(dropna=False).rename_axis("status").reset_index(name="records")
            vc["share"] = vc.records / vc.records.sum()
            fig = px.bar(vc, x="status", y="records", color="status", color_discrete_map=FAIL_COLORS,
                         text=vc.apply(lambda r: f"{r.records:,} ({r.share:.2%})", axis=1),
                         title="Observed machine failure (record-level)")
            fig.update_layout(showlegend=False)
            show(fig)
            explain("Count of records labelled failed vs not failed (observed label).",
                    "Machine-level failure incidence, or whether the imbalance reflects real operations (data is synthetic).")
    with c2:
        if need(df, "maintenance_priority"):
            order = priority_sorted(df["maintenance_priority"])
            vc = df["maintenance_priority"].astype("object").value_counts().reindex(order).rename_axis("priority").reset_index(name="records")
            vc["share"] = vc.records / vc.records.sum()
            fig = px.bar(vc, x="priority", y="records", text=vc.apply(lambda r: f"{r.records:,} ({r.share:.1%})", axis=1),
                         title="Maintenance priority distribution", color="priority",
                         color_discrete_sequence=["#6BAA75", "#F2C14E", "#F78154", "#B33F40"])
            fig.update_layout(showlegend=False)
            show(fig)
            explain("Records per labelled priority class.", "How the priority was derived — see Data Inspection for its consistency with failures.")

    c1, c2 = st.columns(2)
    with c1:
        if need(df, "machine_health_score"):
            fig = px.histogram(df, x="machine_health_score", nbins=60, title="Machine health score distribution",
                               labels={"machine_health_score": "Health score"})
            t = df.attrs.get("health_thresholds") or None
            h = df["machine_health_score"]
            scale = 1.0 if h.max() <= 1.5 else 100.0
            colors = ["#B33F40", "#F78154", "#F2C14E", "#6BAA75"]
            edges = [h.min(), 0.4 * scale, 0.6 * scale, 0.8 * scale, h.max()]
            for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                if b > a:
                    fig.add_vrect(x0=a, x1=b, fillcolor=colors[i], opacity=0.12, line_width=0,
                                  annotation_text=["Critical", "Poor", "Fair", "Good"][i], annotation_position="top left")
            show(fig)
            explain("Distribution of the machine_health_score. Shaded bands (40/60/80% of scale) are display conventions chosen here, not dataset-defined thresholds.",
                    "That a given band implies failure or required action.")
    with c2:
        if need(df, "remaining_useful_life_hours"):
            fig = px.histogram(df, x="remaining_useful_life_hours", nbins=60, title="Remaining useful life distribution",
                               labels={"remaining_useful_life_hours": "RUL (hours)"})
            show(fig)
            explain("Distribution of RUL, a regression target (continuous hours).",
                    "Whether RUL values are consistent with the failure label — checked on the Data Inspection page.")

    c1, c2 = st.columns(2)
    with c1:
        if need(df, "failure_type"):
            ft = df["failure_type"].astype("object")
            excl = st.checkbox("Exclude 'no failure' labels", value=True, key="ov_excl")
            if excl:
                ft = ft[~ft.map(is_no_failure)]
            vc = ft.value_counts().rename_axis("failure_type").reset_index(name="records").sort_values("records")
            if vc.empty:
                st.info("No failure-type labels in the current selection.")
            else:
                fig = px.bar(vc, x="records", y="failure_type", orientation="h", title="Failure types (ranked)", text="records")
                show(fig)
                explain("Frequency of each failure_type label among filtered records.",
                        "Relative risk — frequency depends on how many records each machine group contributes.")
    with c2:
        if need(df, "machine_health_score", "failure_probability"):
            d = df[["machine_health_score", "failure_probability"] + (["_fail"] if "_fail" in df else [])].dropna(subset=["machine_health_score", "failure_probability"])
            s = sample(d)
            fig = px.scatter(s, x="machine_health_score", y="failure_probability", opacity=0.35, render_mode="webgl",
                             color=fail_label(s) if "_fail" in s else None, color_discrete_map=FAIL_COLORS,
                             title="Health score vs failure probability",
                             labels={"machine_health_score": "Health score", "failure_probability": "Failure probability"})
            bm = binned_mean(d, "machine_health_score", "failure_probability", 25)
            if len(bm):
                fig.add_trace(go.Scatter(x=bm.x, y=bm.y, mode="lines+markers", name="Binned mean", line=dict(color="black")))
            show(fig)
            rho = d["machine_health_score"].corr(d["failure_probability"], method="spearman")
            insight(f"Spearman correlation between health score and failure probability: {rho:.2f} (n={len(d):,}).")
            explain("Whether low-health records receive higher failure_probability values.",
                    "Calibration or causality — failure_probability is a dataset column whose meaning is audited on the Data Inspection page.")
