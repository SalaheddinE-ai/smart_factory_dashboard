from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..utils import FAIL_COLORS, binned_mean, explain, fail_label, need, sample, show, table


def render(df: pd.DataFrame, ctx: dict):
    st.header("Sensor Analytics")
    groups = ctx["sensors"]
    if not groups:
        st.warning("No sensor-like numeric columns were detected. Check the column mapping / Data Inspection page.")
        return
    g = st.selectbox("Sensor group", list(groups))
    cols_g = groups[g]
    sel = st.multiselect("Sensors", cols_g, default=cols_g[:6])
    st.caption(f"Sensors are auto-grouped by column-name keywords ({sum(len(v) for v in groups.values())} detected). "
               "Unclassified numeric columns appear under 'Other'.")
    t1, t2, t3, t4, t5 = st.tabs(["Distributions", "Correlations", "Temperature comparison", "Pairwise & segments", "Health & RUL relationships"])

    with t1:
        if not sel:
            st.info("Select at least one sensor.")
        else:
            split = st.checkbox("Split by observed failure", value="_fail" in df, disabled="_fail" not in df)
            cols = sel + (["_fail"] if split else [])
            m = sample(df[cols], 20000).melt(id_vars=["_fail"] if split else None, value_vars=sel)
            if split:
                m["status"] = m["_fail"].map({0.0: "No failure", 1.0: "Failure"})
            fig = px.histogram(m, x="value", facet_col="variable", facet_col_wrap=3, color="status" if split else None,
                               color_discrete_map=FAIL_COLORS, barmode="overlay", opacity=0.65, histnorm="probability density", nbins=40)
            fig.update_xaxes(matches=None)
            fig.update_yaxes(matches=None, showticklabels=False)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            show(fig, 280 * int(np.ceil(len(sel) / 3)) + 80)
            table(df[sel].describe().T.reset_index().rename(columns={"index": "sensor"}).round(3))
            explain("Marginal distribution of each sensor (density-normalised when split).",
                    "Units (not stated in the CSV) or whether differences between classes are meaningful.")
    with t2:
        allv = ctx["all_sensors"]
        extra = [c for c in ["machine_health_score", "remaining_useful_life_hours", "failure_probability"] if c in df]
        default = (sel or allv)[:15]
        vars_ = st.multiselect("Variables for the heatmap", allv + extra, default=default + [c for c in extra if c not in default][:0])
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
        if len(vars_) >= 2:
            corr = df[vars_].corr(method=method)
            fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f" if len(vars_) <= 15 else False,
                            title=f"{method.title()} correlation (n={len(df):,})", aspect="auto")
            show(fig, 560)
            explain("Pairwise monotonic/linear association between selected variables.",
                    "Causation; also, strong correlations in synthetic data can be artefacts of the generator.")
        else:
            st.info("Select at least two variables.")
    with t3:
        th = groups.get("Thermal", [])
        if not th:
            st.info("No temperature columns detected.")
        else:
            tsel = st.multiselect("Temperature sensors", th, default=th, key="temp_sel")
            if tsel:
                s = sample(df[tsel + (["_fail"] if "_fail" in df else [])], 20000)
                m = s.melt(id_vars=["_fail"] if "_fail" in s else None, value_vars=tsel)
                if "_fail" in m:
                    m["status"] = m["_fail"].map({0.0: "No failure", 1.0: "Failure"})
                fig = px.box(m, x="variable", y="value", color="status" if "_fail" in m else None, color_discrete_map=FAIL_COLORS,
                             title="Temperature sensors compared (box plots)", labels={"value": "Temperature (dataset units)", "variable": ""})
                show(fig)
                explain("Spread of each temperature sensor, by failure status.",
                        "Whether differences are physically meaningful: columns may use different scales/units.")
    with t4:
        allv = ctx["all_sensors"]
        if len(allv) >= 2:
            th = groups.get("Thermal", []); vb = groups.get("Vibration", [])
            dx = allv.index(th[0]) if th else 0
            dy = allv.index(vb[0]) if vb else min(1, len(allv) - 1)
            c1, c2 = st.columns(2)
            x = c1.selectbox("X sensor", allv, index=dx)
            y = c2.selectbox("Y sensor", allv, index=dy)
            s = sample(df[[x, y] + (["_fail"] if "_fail" in df else [])].dropna())
            fig = px.scatter(s, x=x, y=y, color=fail_label(s) if "_fail" in s else None, color_discrete_map=FAIL_COLORS,
                             opacity=0.4, render_mode="webgl", title=f"{x} vs {y}")
            show(fig)
            explain("Joint behaviour of two sensors, coloured by observed failure.", "That the overlap/separation seen generalises to real machines.")
        seg = [c for c in ["machine_type", "production_line"] if c in df]
        if seg and allv:
            c1, c2 = st.columns(2)
            by = c1.selectbox("Group by", seg)
            sv = c2.selectbox("Sensor", allv, key="seg_sensor")
            gg = df.groupby(by, observed=True)[sv].agg(mean="mean", sd="std", n="count").reset_index()
            gg["ci"] = 1.96 * gg.sd / np.sqrt(gg.n)
            fig = go.Figure(go.Bar(x=gg[by].astype(str), y=gg["mean"], error_y=dict(type="data", array=gg.ci),
                                   customdata=gg.n, hovertemplate="%{x}<br>mean=%{y:.3f}<br>n=%{customdata:,}<extra></extra>"))
            fig.update_layout(title=f"Mean {sv} by {by} (±95% CI)", yaxis_title=sv, xaxis_title=by)
            show(fig)
            explain("Group means with normal-approximation confidence intervals.", "Whether group differences persist after controlling for load, age or type.")
        elif not seg:
            st.info("machine_type / production_line columns not found.")
    with t5:
        targets = [c for c in ["machine_health_score", "remaining_useful_life_hours"] if c in df]
        allv = ctx["all_sensors"]
        if not targets or not allv:
            st.info("Needs health score / RUL and at least one sensor.")
        else:
            c1, c2 = st.columns(2)
            tg = c1.selectbox("Target", targets)
            sv = c2.selectbox("Sensor", allv, key="rel_sensor")
            d = df[[sv, tg]].dropna()
            s = sample(d)
            fig = px.scatter(s, x=sv, y=tg, opacity=0.3, render_mode="webgl", title=f"{tg} vs {sv}")
            bm = binned_mean(d, sv, tg, 25)
            if len(bm):
                fig.add_trace(go.Scatter(x=bm.x, y=bm.y, mode="lines+markers", name="Binned mean", line=dict(color="black")))
            show(fig)
            rho = df[allv + targets].corr(method="spearman")[targets].loc[allv]
            r = rho[tg].sort_values(key=abs, ascending=False).head(15).reset_index()
            r.columns = ["sensor", "spearman"]
            fig = px.bar(r.sort_values("spearman"), x="spearman", y="sensor", orientation="h", title=f"Top |Spearman| sensors vs {tg}")
            show(fig)
            explain("Rank correlation between each sensor and the health / RUL columns.",
                    "Direction of influence: health and RUL may themselves be computed from sensors in the generator.")
