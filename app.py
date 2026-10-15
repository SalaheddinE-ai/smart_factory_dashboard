from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.config import KEY_ALIASES
from src.data import auto_mapping, discover, prepare, read_columns
from src.filters import render_filters
from src.views import anomalies, components, failures, inspection, maintenance, overview, sensors, temporal

st.set_page_config(page_title="Predictive Maintenance Analytics",page_icon="🌕", layout="wide")
st.markdown("<style>div[data-testid='stMetricValue']{font-size:1.5rem}</style>", unsafe_allow_html=True)

PAGES = {
    "0 · Data Inspection": None,
    "1 · Executive Overview": overview,
    "2 · Sensor Analytics": sensors,
    "3 · Failure Analysis": failures,
    "4 · Temporal & Operational": temporal,
    "5 · Health & Components": components,
    "6 · Anomaly Detection": anomalies,
    "7 · Maintenance Intelligence": maintenance,
}


def landing():
    st.title(" Smart Factory — Predictive Maintenance Analytics")
    st.markdown(
        "Upload the **Smart Factory Predictive Maintenance** CSV in the sidebar (or give a local path for very large files).\n\n"
        "The app first inspects the file (dimensions, types, missing values, label consistency, leakage risks) and maps "
        "column names automatically — you can correct the mapping in the sidebar.\n\n"
        "*The dataset is described as synthetic: findings are exploratory and must not be read as validated industrial laws.*")


def main():
    sb = st.sidebar
    sb.title("PdM Analytics")
    up = sb.file_uploader("Upload CSV", type=["csv"])
    path = sb.text_input("…or local CSV path", "", help="Bypasses the browser upload limit.")
    src, mtime, sig = None, 0.0, None
    if up is not None:
        src, sig = up.getvalue(), f"{up.name}-{up.size}"
    elif path:
        if os.path.isfile(path):
            mtime = os.path.getmtime(path)
            src, sig = path, f"{os.path.basename(path)}-{int(mtime)}"
        else:
            sb.error("File not found.")
    if src is None:
        landing()
        return

    try:
        cols = read_columns(src, mtime)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the CSV: {e}")
        return
    auto = auto_mapping(cols)
    with sb.expander(f"Column mapping ({len(auto)}/{len(KEY_ALIASES)} auto-detected)"):
        opts = ["(none)"] + cols
        mapping = {}
        for k in KEY_ALIASES:
            v = st.selectbox(k, opts, index=opts.index(auto[k]) if k in auto else 0, key=f"map_{sig}_{k}")
            if v != "(none)":
                mapping[k] = v
    try:
        df, info = prepare(src, mtime, tuple(sorted(mapping.items())))
    except Exception as e:  # noqa: BLE001
        st.error(f"Preprocessing failed: {e}")
        return
    ctx = discover(df)

    page = sb.radio("Page", list(PAGES))
    filtered = render_filters(df, sig)

    if page.startswith("0"):
        inspection.render(df, ctx, info)
        return
    if filtered.empty:
        st.warning("No records match the current filters. Adjust or reset them in the sidebar.")
        return
    try:
        PAGES[page].render(filtered, ctx)
    except Exception as e:  # noqa: BLE001
        st.error(f"This page could not be rendered: {type(e).__name__}: {e}")
        st.exception(e)


main()
