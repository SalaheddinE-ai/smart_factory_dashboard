"""Shared helpers: statistics, safe plotting, binning, machine-level logic."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .config import NO_FAILURE, PRIORITY_ORDER

FAIL_COLORS = {"No failure": "#2E86AB", "Failure": "#D62839"}


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ----------------------------------------------------------------- UI helpers
def show(fig: go.Figure, height: int = 400):
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=55, b=10), height=height)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def table(df: pd.DataFrame, **kw):
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    fl = df.select_dtypes("float").columns
    df[fl] = df[fl].round(4)
    st.dataframe(df, hide_index=True, **kw)


def need(df: pd.DataFrame, *cols: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.info(f"Visualization unavailable — missing column(s): {', '.join(missing)}. "
                "Use the sidebar *Column mapping* panel if the column exists under another name.")
        return False
    return True


def explain(measures: str, cannot: str):
    st.caption(f"**Measures:** {measures}  \n**Cannot establish:** {cannot}")


def insight(text: str):
    st.info(f"**Observation (descriptive):** {text}")


def sample(df: pd.DataFrame, n: int = 8000) -> pd.DataFrame:
    if len(df) <= n:
        return df
    st.caption(f"Scatter shows a random sample of {n:,} of {len(df):,} filtered records (statistics use all records).")
    return df.sample(n, random_state=0)


def fail_label(df: pd.DataFrame) -> pd.Series:
    return df["_fail"].map({0.0: "No failure", 1.0: "Failure"})


# ----------------------------------------------------------------- statistics
def wilson(k, n, z: float = 1.96):
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)


def _as_list(by):
    return [by] if isinstance(by, (str, pd.Series)) else list(by)


def rate_table(df: pd.DataFrame, by) -> pd.DataFrame:
    """Record-level failure rate with Wilson 95% CI. Denominator = records with a known label."""
    g = df.groupby(_as_list(by), observed=True)["_fail"].agg(n="count", failures="sum").reset_index()
    g["rate"] = g.failures / g.n
    g["lo"], g["hi"] = wilson(g.failures, g.n)
    return g


def machine_rate_table(df: pd.DataFrame, by) -> pd.DataFrame:
    """Machine-level: share of distinct machines with >=1 failed record in the group."""
    by = [b for b in _as_list(by)]
    m = df.groupby(by + ["machine_id"], observed=True)["_fail"].max().reset_index()
    m = m.dropna(subset=["_fail"])
    g = m.groupby(by, observed=True)["_fail"].agg(n="count", failures="sum").reset_index()
    g["rate"] = g.failures / g.n
    g["lo"], g["hi"] = wilson(g.failures, g.n)
    return g


def binned_mean(df: pd.DataFrame, x: str, y: str, bins: int = 20) -> pd.DataFrame:
    d = df[[x, y]].dropna()
    if d[x].nunique() < 3:
        return pd.DataFrame(columns=["x", "y", "n"])
    b = pd.qcut(d[x], bins, duplicates="drop")
    g = d.groupby(b, observed=True).agg(x=(x, "mean"), y=(y, "mean"), n=(y, "size")).reset_index(drop=True)
    return g


def qbin(s: pd.Series, q: int = 5, prec: int = 1):
    try:
        c = pd.qcut(s, q, duplicates="drop")
    except Exception:
        return None
    try:
        return c.cat.rename_categories([f"{iv.left:.{prec}f}–{iv.right:.{prec}f}" for iv in c.cat.categories])
    except Exception:
        return c.astype(str)


def rate_bar(t: pd.DataFrame, x: str, title: str, min_n: int = 30, unit: str = "records") -> go.Figure:
    hidden = int((t.n < min_n).sum())
    t = t[t.n >= min_n]
    fig = go.Figure(go.Bar(
        x=t[x].astype(str), y=t.rate, marker_color="#2E86AB",
        error_y=dict(type="data", symmetric=False, array=(t.hi - t.rate), arrayminus=(t.rate - t.lo)),
        customdata=np.c_[t.n, t.failures, t.lo, t.hi],
        hovertemplate=f"%{{x}}<br>rate=%{{y:.2%}} (95% CI %{{customdata[2]:.2%}}–%{{customdata[3]:.2%}})"
                      f"<br>n={'%{customdata[0]:,.0f}'} {unit}, failures=%{{customdata[1]:,.0f}}<extra></extra>"))
    sub = f" — {hidden} group(s) hidden (n<{min_n})" if hidden else ""
    fig.update_layout(title=title + sub, yaxis_title=f"Failure rate (% of {unit})", yaxis_tickformat=".1%", xaxis_title=x)
    return fig


# ---------------------------------------------------------------- label logic
def is_no_failure(v) -> bool:
    return pd.isna(v) or norm(v) in NO_FAILURE


def priority_sorted(values) -> list:
    vals = [v for v in pd.unique(pd.Series(values).dropna())]
    return sorted(vals, key=lambda v: PRIORITY_ORDER.index(norm(v)) if norm(v) in PRIORITY_ORDER else 99)


def priority_rank(s: pd.Series) -> pd.Series:
    return s.astype("object").map(lambda v: PRIORITY_ORDER.index(norm(v)) if norm(v) in PRIORITY_ORDER else np.nan)


def latest_per_machine(df: pd.DataFrame) -> pd.DataFrame:
    """Latest record per machine chosen by actual timestamp (never by CSV row order)."""
    if "machine_id" not in df.columns:
        return df
    if "timestamp" in df.columns:
        d = df.sort_values(["machine_id", "timestamp"], na_position="first")
    else:
        d = df
    return d.drop_duplicates("machine_id", keep="last")


def investigation_table(last: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """Transparent screening index = mean percentile rank of available risk signals.

    This is a heuristic ranking for triage — not a model and not a diagnosis.
    """
    parts = {}
    if "machine_health_score" in last:
        parts["low_health"] = 1 - last["machine_health_score"].rank(pct=True)
    if "failure_probability" in last:
        parts["high_failure_prob"] = last["failure_probability"].rank(pct=True)
    if "remaining_useful_life_hours" in last:
        parts["low_RUL"] = 1 - last["remaining_useful_life_hours"].rank(pct=True)
    if ctx["comps"]:
        parts["worst_component"] = 1 - last[ctx["comps"]].rank(pct=True).min(axis=1)
    if ctx["scores"]:
        parts["max_anomaly"] = last[ctx["scores"]].rank(pct=True).max(axis=1)
    out = last.copy()
    if not parts:
        out["investigation_index"] = np.nan
        return out
    p = pd.concat(parts, axis=1)
    out["investigation_index"] = p.mean(axis=1)
    out["signals_used"] = ", ".join(parts)
    return out
