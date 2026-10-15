"""Loading, column resolution, preprocessing and schema discovery."""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import streamlit as st

from .config import CAT_KEYS, KEY_ALIASES, NON_SENSOR_TOKENS, SENSOR_GROUPS
from .utils import norm


# --------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Reading CSV…")
def read_csv_cached(src, mtime: float = 0.0) -> pd.DataFrame:
    """src = bytes (upload) or str path. Cached without copying (treat result as read-only)."""
    buf = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src
    return pd.read_csv(buf, low_memory=False)


@st.cache_data(show_spinner=False)
def read_columns(src, mtime: float = 0.0) -> list[str]:
    buf = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src
    return list(pd.read_csv(buf, nrows=0).columns)


def auto_mapping(columns) -> dict[str, str]:
    """logical name -> actual column (exact, normalised, then alias match)."""
    nm = {norm(c): c for c in columns}
    out = {}
    for key, aliases in KEY_ALIASES.items():
        for a in [key] + aliases:
            if norm(a) in nm:
                out[key] = nm[norm(a)]
                break
    return out


def qbin_labels(s: pd.Series, q: int, prec: int = 1):
    try:
        c = pd.qcut(s, q, duplicates="drop")
        return c.cat.rename_categories([f"{iv.left:.{prec}f}–{iv.right:.{prec}f}" for iv in c.cat.categories])
    except Exception:
        return None


# --------------------------------------------------------------- preprocessing
@st.cache_resource(show_spinner="Preprocessing…", max_entries=2)
def prepare(src, mtime: float, mapping_items: tuple):
    raw = read_csv_cached(src, mtime)
    mapping = dict(mapping_items)
    df = raw.copy()
    info: dict = {
        "raw_shape": raw.shape,
        "raw_columns": list(raw.columns),
        "raw_mem_mb": float(raw.memory_usage(deep=True).sum() / 2**20),
        "raw_full_dups": int(raw.duplicated().sum()),
        "mapping": mapping,
    }
    ren = {a: k for k, a in mapping.items() if a != k and k not in df.columns and a in df.columns}
    df = df.rename(columns=ren)
    info["renamed"] = ren
    df["_row"] = np.arange(len(df)) + 2  # CSV line number (header = line 1)

    # timestamp
    if "timestamp" in df:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce")
        info["ts_unparsed"] = int(parsed.isna().sum() - df["timestamp"].isna().sum())
        info["ts_missing"] = int(df["timestamp"].isna().sum())
        df["timestamp"] = parsed
        if "machine_id" in df:
            info["raw_sorted"] = bool(df.groupby("machine_id", observed=True)["timestamp"]
                                      .apply(lambda s: s.dropna().is_monotonic_increasing).all())

    # observed failure label -> numeric helper column (original column untouched)
    if "machine_failure" in df:
        s = df["machine_failure"]
        if pd.api.types.is_bool_dtype(s):
            f = s.astype(float)
        elif pd.api.types.is_numeric_dtype(s):
            info["fail_raw_values"] = sorted(map(float, s.dropna().unique()))[:10]
            f = (s > 0).astype(float).where(s.notna())
        else:
            mp = {"1": 1, "true": 1, "yes": 1, "y": 1, "failure": 1, "failed": 1,
                  "0": 0, "false": 0, "no": 0, "n": 0, "none": 0}
            f = s.astype(str).str.strip().str.lower().map(mp).astype(float).where(s.notna())
            info["fail_unmapped"] = int(f.isna().sum() - s.isna().sum())
        df["_fail"] = f

    # categorical columns: dtype only, labels untouched
    for k in CAT_KEYS:
        if k in df:
            s = df[k].astype("object")
            df[k] = pd.Series(np.where(s.isna(), None, s.astype(str)), index=df.index).astype("category")

    # chronological, machine-aware ordering
    if "machine_id" in df and "timestamp" in df:
        df = df.sort_values(["machine_id", "timestamp"], kind="stable").reset_index(drop=True)
    elif "timestamp" in df:
        df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    # derived analysis helpers
    if "machine_health_score" in df:
        h = df["machine_health_score"]
        scale = 1.0 if (h.max() <= 1.5) else 100.0
        t = [0.4 * scale, 0.6 * scale, 0.8 * scale]
        info["health_thresholds"] = t
        df["health_band"] = pd.cut(h, [-np.inf, *t, np.inf],
                                   labels=["Critical", "Poor", "Fair", "Good"])
    if "machine_age_years" in df:
        df["age_group"] = qbin_labels(df["machine_age_years"], 5)
    if "days_since_maintenance" in df:
        df["dsm_group"] = qbin_labels(df["days_since_maintenance"], 8, 0)
    return df, info


# ------------------------------------------------------------------- discovery
def tokens(c: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", c.lower()) if t]


def discover(df: pd.DataFrame) -> dict:
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
           and not pd.api.types.is_bool_dtype(df[c]) and not c.startswith("_")]
    comps = [c for c in num if tokens(c)[-1:] == ["health"] and c != "machine_health_score"]
    anomaly_all = [c for c in num if "anomaly" in tokens(c)]
    flags = [c for c in anomaly_all if df[c].nunique(dropna=True) <= 2]
    scores = [c for c in anomaly_all if c not in flags]
    sensors = {g: [] for g in SENSOR_GROUPS}
    sensors["Other"] = []
    keep_logical = {"operational_load", "energy_consumption"}
    for c in num:
        toks = set(tokens(c))
        if toks & NON_SENSOR_TOKENS or (c in KEY_ALIASES and c not in keep_logical):
            continue
        for g, kws in SENSOR_GROUPS.items():
            if any(tok.startswith(k) for tok in toks for k in kws):
                sensors[g].append(c)
                break
        else:
            sensors["Other"].append(c)
    sensors = {g: v for g, v in sensors.items() if v}
    return {
        "num": num, "comps": comps, "scores": scores, "flag": flags[0] if flags else None,
        "sensors": sensors, "all_sensors": [c for v in sensors.values() for c in v],
    }
