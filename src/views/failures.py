from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ..utils import (FAIL_COLORS, explain, fail_label, is_no_failure, machine_rate_table, need, qbin, rate_bar,
                     rate_table, show, table)


def render(df: pd.DataFrame, ctx: dict):
    st.header("Failure Analysis")
    if not need(df, "_fail"):
        return
    can_machine = "machine_id" in df
    opts = ["Record-level"] + (["Machine-level (≥1 failed record)"] if can_machine else [])
    level = st.radio("Failure-rate definition", opts, horizontal=True)
    rec = level.startswith("Record")
    min_n = st.slider("Hide groups with fewer than N observations", 1, 500, 30)
    unit = "records" if rec else "machines"
    st.caption("Record-level = failed records / records. Machine-level = machines with ≥1 failed record / machines. "
               "Error bars are Wilson 95% confidence intervals. Binned continuous variables (age, days since maintenance) are always record-level.")

    def tbl(by):
        return rate_table(df, by) if rec else machine_rate_table(df, by)

    cats = [("machine_type", "Failure rate by machine type"), ("manufacturer", "Failure rate by manufacturer")]
    cols = st.columns(2)
    for i, (c, title) in enumerate(cats):
        with cols[i]:
            if need(df, c):
                show(rate_bar(tbl(c), c, title, min_n, unit))
                explain(f"Share of {unit} that failed, per {c}.", "Whether the group itself drives failures (confounded by age, load, line).")

    c1, c2 = st.columns(2)
    with c1:
        if need(df, "production_line"):
            show(rate_bar(tbl("production_line"), "production_line", "Failure rate by production line", min_n, unit))
    with c2:
        if "factory_id" in df and "production_line" in df:
            t = tbl(["factory_id", "production_line"])
            t = t.assign(rate=t.rate.where(t.n >= min_n))
            pv = t.pivot(index="factory_id", columns="production_line", values="rate")
            pn = t.pivot(index="factory_id", columns="production_line", values="n")
            fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns.astype(str), y=pv.index.astype(str), colorscale="Reds",
                                       text=[[f"{v:.1%}<br>n={int(n) if pd.notna(n) else 0}" if pd.notna(v) else "" for v, n in zip(rv, rn)]
                                             for rv, rn in zip(pv.values, pn.values)], texttemplate="%{text}",
                                       colorbar=dict(title="rate", tickformat=".0%")))
            fig.update_layout(title="Failure rate: factory × production line", xaxis_title="Production line", yaxis_title="Factory")
            show(fig)
        elif need(df, "factory_id"):
            show(rate_bar(tbl("factory_id"), "factory_id", "Failure rate by factory", min_n, unit))
    explain("Failure rate by line / factory (cells below the minimum n are blanked).", "Differences between factories if lines are named identically across factories.")

    st.subheader("Failure types across machine categories")
    if need(df, "failure_type", "machine_type"):
        d = df[~df["failure_type"].astype("object").map(is_no_failure)]
        if d.empty:
            st.info("No failure-type labels other than 'no failure' in the current selection.")
        else:
            norm_ = st.checkbox("Show row percentages (share within machine type)", value=False)
            ct = pd.crosstab(d["machine_type"], d["failure_type"])
            z = ct.div(ct.sum(axis=1), axis=0) if norm_ else ct
            fig = px.imshow(z, text_auto=".0%" if norm_ else True, aspect="auto", color_continuous_scale="Blues",
                            title="Failure-type frequency by machine type")
            show(fig)
            explain("How labelled failure types distribute across machine types.", "Per-type risk — counts reflect how many records each type contributes.")

    st.subheader("Predicted probability vs observed failures, by health")
    if need(df, "machine_health_score", "failure_probability"):
        d = df[["machine_health_score", "failure_probability", "_fail"]].dropna()
        d = d.assign(hb=qbin(d["machine_health_score"], 10))
        g = d.groupby("hb", observed=True).agg(n=("_fail", "size"), failures=("_fail", "sum"), prob=("failure_probability", "mean")).reset_index()
        g["obs"] = g.failures / g.n
        fig = go.Figure()
        fig.add_bar(x=g.hb.astype(str), y=g.obs, name="Observed failure rate")
        fig.add_scatter(x=g.hb.astype(str), y=g.prob, mode="lines+markers", name="Mean failure_probability")
        fig.update_layout(title="Observed failure rate vs mean failure_probability by health decile", yaxis_tickformat=".1%",
                          xaxis_title="Health score decile range", yaxis_title="Rate / probability")
        show(fig)
        explain("Whether the dataset's failure_probability tracks observed failure frequency across health levels.",
                "That failure_probability is calibrated; see Data Inspection for a calibration table.")

    c1, c2 = st.columns(2)
    with c1:
        if need(df, "machine_age_years"):
            d = df.assign(age_group=qbin(df["machine_age_years"], 5))
            show(rate_bar(rate_table(d, "age_group"), "age_group", "Failure rate by machine age group (years)", min_n))
            explain("Record-level failure rate in age quintiles.", "An age effect: age may correlate with maintenance and load.")
    with c2:
        if need(df, "days_since_maintenance"):
            d = df.assign(dsm=qbin(df["days_since_maintenance"], 8, 0))
            t = rate_table(d, "dsm")
            t = t[t.n >= min_n]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_bar(x=t.dsm.astype(str), y=t.failures, name="Failed records", secondary_y=False)
            fig.add_scatter(x=t.dsm.astype(str), y=t.rate, mode="lines+markers", name="Failure rate", secondary_y=True)
            fig.update_yaxes(title_text="Failed records", secondary_y=False)
            fig.update_yaxes(title_text="Failure rate", tickformat=".1%", secondary_y=True)
            fig.update_layout(title="Failures vs days since maintenance (binned)", xaxis_title="Days since maintenance")
            show(fig)
            explain("Count and rate of failed records per days-since-maintenance bin.", "That maintenance timing causes failures.")

    st.subheader("Anomaly scores vs observed failure labels")
    if ctx["scores"]:
        sc = st.selectbox("Anomaly score", ctx["scores"])
        d = df[[sc, "_fail"]].dropna()
        d = d.assign(status=fail_label(d))
        fig = px.box(d.sample(min(len(d), 30000), random_state=0), x="status", y=sc, color="status", color_discrete_map=FAIL_COLORS,
                     title=f"{sc} by observed failure status")
        show(fig)
        rows = []
        if d["_fail"].nunique() == 2:
            from sklearn.metrics import roc_auc_score
            for c in ctx["scores"]:
                dd = df[[c, "_fail"]].dropna()
                rows.append({"score": c, "mean | no failure": dd.loc[dd._fail == 0, c].mean(),
                             "mean | failure": dd.loc[dd._fail == 1, c].mean(), "AUC vs failure label": roc_auc_score(dd._fail, dd[c])})
            table(pd.DataFrame(rows).round(3))
        explain("Descriptive separation of each anomaly score by failure label (AUC 0.5 = no separation).",
                "That a high score indicates a confirmed failure; scores may be generated independently of the label.")
    else:
        st.info("No anomaly score columns detected.")
