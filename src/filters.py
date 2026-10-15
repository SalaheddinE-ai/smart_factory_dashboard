"""Sidebar filters. Empty multiselect = no restriction (sensible default: everything)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

FILTER_COLS = [
    ("factory_id", "Factory"), ("production_line", "Production line"), ("machine_type", "Machine type"),
    ("manufacturer", "Manufacturer"), ("failure_type", "Failure type"),
    ("maintenance_priority", "Maintenance priority"), ("machine_id", "Machine identifier"),
]


def _reset(keys: dict):
    for k, v in keys.items():
        st.session_state[k] = v


def render_filters(df: pd.DataFrame, sig: str) -> pd.DataFrame:
    sb = st.sidebar
    sb.subheader("Filters")
    defaults: dict = {}
    mask = pd.Series(True, index=df.index)
    missing = []

    for col, label in FILTER_COLS:
        if col not in df:
            missing.append(label)
            continue
        key = f"flt_{sig}_{col}"
        defaults[key] = []
        opts = [str(c) for c in df[col].cat.categories] if hasattr(df[col], "cat") else sorted(df[col].dropna().astype(str).unique())
        sel = sb.multiselect(label, opts, key=key, placeholder="All")
        if sel:
            mask &= df[col].isin(sel)

    if "_fail" in df:
        key = f"flt_{sig}_status"
        defaults[key] = "All"
        status = sb.radio("Machine failure status", ["All", "Failed", "Not failed"], key=key, horizontal=True)
        if status == "Failed":
            mask &= df["_fail"] == 1
        elif status == "Not failed":
            mask &= df["_fail"] == 0
    else:
        missing.append("Failure status")

    if "timestamp" in df and df["timestamp"].notna().any():
        dmin, dmax = df["timestamp"].min().date(), df["timestamp"].max().date()
        key = f"flt_{sig}_dates"
        defaults[key] = (dmin, dmax)
        rng = sb.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, key=key)
        if isinstance(rng, (tuple, list)) and len(rng) == 2 and (rng[0], rng[1]) != (dmin, dmax):
            ts = df["timestamp"]
            mask &= (ts >= pd.Timestamp(rng[0])) & (ts < pd.Timestamp(rng[1]) + pd.Timedelta(days=1))
    else:
        missing.append("Date range")

    sb.button("Reset filters", on_click=_reset, args=(defaults,), width="stretch") if _supports_width() \
        else sb.button("Reset filters", on_click=_reset, args=(defaults,))
    out = df[mask]
    nm = out["machine_id"].nunique() if "machine_id" in out else "n/a"
    sb.caption(f"**{len(out):,}** of {len(df):,} records · **{nm}** machines")
    if missing:
        sb.caption("Unavailable filters (column not found): " + ", ".join(missing))
    return out


def _supports_width() -> bool:
    import inspect
    return "width" in inspect.signature(st.button).parameters
