"""Step 1 of the brief: inspect the file before trusting any chart."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..config import KEY_ALIASES
from ..utils import is_no_failure, norm, show, table

import plotly.express as px

LEAK_TOKENS = ("failure", "rul", "remaining", "health", "probab", "downtime", "priority", "anomaly", "repair")


def render(df: pd.DataFrame, ctx: dict, info: dict):
    st.header("Data Inspection & Validation")
    st.caption("Always computed on the full, unfiltered file. Nothing is removed or relabelled by the dashboard.")

    # 1. dimensions
    rows, cols = info["raw_shape"]
    c = st.columns(4)
    c[0].metric("Rows", f"{rows:,}")
    c[1].metric("Columns", cols)
    c[2].metric("Unique machines", f"{df['machine_id'].nunique():,}" if "machine_id" in df else "n/a")
    c[3].metric("Memory (raw)", f"{info['raw_mem_mb']:.1f} MB")
    st.subheader("Dataset description vs actual file")
    checks = [("Rows ≈ 100,000", abs(rows - 100_000) <= 1000, f"actual {rows:,}"),
              ("Columns ≈ 72", abs(cols - 72) <= 2, f"actual {cols}")]
    if "machine_id" in df:
        checks.append(("Machines ≈ 5,000", abs(df["machine_id"].nunique() - 5000) <= 50, f"actual {df['machine_id'].nunique():,}"))
    for label, ok, actual in checks:
        (st.success if ok else st.warning)(f"{label}: {'matches' if ok else 'DISCREPANCY'} ({actual})")
    if info["renamed"]:
        st.caption("Columns mapped to logical names: " + ", ".join(f"{a} → {k}" for k, a in info["mapping"].items() if a != k))
    unmapped = [k for k in KEY_ALIASES if k not in info["mapping"]]
    if unmapped:
        st.warning("Logical columns not found (related charts will be disabled): " + ", ".join(unmapped))

    # 2. roles
    st.subheader("Column roles, types and missing values")
    roles = []
    for col in df.columns:
        if col.startswith("_") or col in ("health_band", "age_group", "dsm_group"):
            continue
        s = df[col]
        if col == "machine_id" or ("id" in norm(col).split("_") or col.lower().endswith("_id")):
            role = "identifier"
        elif pd.api.types.is_datetime64_any_dtype(s):
            role = "datetime"
        elif col in ("machine_failure", "failure_type", "maintenance_priority", "remaining_useful_life_hours"):
            role = "target / label"
        elif col in ("failure_probability", "machine_health_score") or col in ctx["scores"]:
            role = "derived score (check leakage)"
        elif pd.api.types.is_numeric_dtype(s):
            role = "numeric"
        else:
            role = "categorical"
        roles.append({"column": col, "dtype": str(s.dtype), "role": role, "missing %": round(s.isna().mean() * 100, 2),
                      "unique": int(s.nunique(dropna=True))})
    table(pd.DataFrame(roles))

    # 3. quality
    st.subheader("Quality checks")
    miss = df.isna().sum()
    miss = miss[miss > 0]
    st.write(f"Columns with missing values: **{len(miss)}**" + (f" — worst: {miss.idxmax()} ({miss.max():,})" if len(miss) else ""))
    st.write(f"Fully duplicated rows: **{info['raw_full_dups']:,}**")
    if "machine_id" in df and "timestamp" in df:
        dup = int(df.duplicated(["machine_id", "timestamp"]).sum())
        st.write(f"Duplicate (machine_id, timestamp) pairs: **{dup:,}**")
        st.write(f"Unparseable timestamps: **{info.get('ts_unparsed', 0):,}** · missing: **{info.get('ts_missing', 0):,}** · "
                 f"CSV chronologically ordered within machine: **{info.get('raw_sorted', 'n/a')}** (the app sorts by machine and timestamp; the file is not modified)")
    issues = []
    if "machine_health_score" in df:
        h = df["machine_health_score"]
        issues.append(("health score range", f"min {h.min():.3g}, max {h.max():.3g}"))
    if "failure_probability" in df:
        p = df["failure_probability"]
        issues.append(("failure_probability outside [0,1]", int(((p < 0) | (p > 1)).sum())))
    if "remaining_useful_life_hours" in df:
        issues.append(("negative RUL", int((df["remaining_useful_life_hours"] < 0).sum())))
    for k in ["maintenance_cost", "days_since_maintenance", "maintenance_count", "downtime_hours", "machine_age_years"]:
        if k in df:
            issues.append((f"negative {k}", int((df[k] < 0).sum())))
    inf_cols = {c: int(np.isinf(df[c]).sum()) for c in ctx["num"] if np.isinf(df[c]).any()}
    issues.append(("columns containing ±inf", inf_cols or 0))
    if "fail_raw_values" in info:
        issues.append(("raw values of machine_failure", info["fail_raw_values"]))
    table(pd.DataFrame(issues, columns=["check", "result"]).astype(str))
    incons = []
    for k in [c for c in df.columns if hasattr(df[c], "cat")]:
        labels = pd.Series([str(x) for x in df[k].cat.categories])
        grp = labels.groupby(labels.map(norm)).agg(list)
        incons += [{"column": k, "variants": ", ".join(v)} for v in grp if len(v) > 1]
    st.write("Inconsistent categorical spellings (same label, different case/spacing): " + (str(len(incons)) if incons else "none found"))
    if incons:
        table(pd.DataFrame(incons))

    # 4. machine/time structure
    if "machine_id" in df:
        st.subheader("Machine / timestamp structure")
        per = df.groupby("machine_id", observed=True).size()
        st.write(f"Records per machine — min {per.min()}, median {per.median():.0f}, max {per.max()}")
        if "timestamp" in df and df["timestamp"].notna().any():
            st.write(f"Time span: {df['timestamp'].min()} → {df['timestamp'].max()}")
            dt = df.dropna(subset=["timestamp"]).groupby("machine_id", observed=True)["timestamp"].diff().dropna()
            if len(dt):
                st.write(f"Median interval between consecutive records of the same machine: {dt.median()}")

    # 5. targets
    st.subheader("Targets and class balance")
    if "_fail" in df:
        r = df["_fail"].mean()
        st.write(f"Observed failure rate: **{r:.3%}** ({int(df['_fail'].sum()):,} of {int(df['_fail'].count()):,} labelled records) · "
                 f"imbalance ≈ 1 : {(1 - r) / r:.1f}" if r > 0 else "No failed records in the file.")
        if r < 0.1:
            st.warning("Strong class imbalance: accuracy is uninformative; use PR-AUC / recall / precision and machine-aware splits for any model.")
        if "machine_id" in df:
            st.write(f"Machines with ≥1 failed record: **{df.groupby('machine_id', observed=True)['_fail'].max().mean():.2%}**")
    for k in ["failure_type", "maintenance_priority"]:
        if k in df:
            vc = df[k].value_counts(dropna=False).rename_axis(k).reset_index(name="records")
            table(vc)

    # 6. consistency
    st.subheader("Consistency between labels")
    if "_fail" in df and "failure_type" in df:
        nf = df["failure_type"].astype("object").map(is_no_failure)
        a = int(((df["_fail"] == 1) & nf).sum())
        b = int(((df["_fail"] == 0) & ~nf).sum())
        st.write(f"Failed records labelled 'no failure' in failure_type: **{a:,}** · Non-failed records carrying a failure type: **{b:,}**")
        table(pd.crosstab(df["failure_type"], df["_fail"]).reset_index())
    if "_fail" in df and "maintenance_priority" in df:
        ct = pd.crosstab(df["maintenance_priority"], df["_fail"], normalize="index").round(4)
        st.write("Share of failed records within each priority class:")
        table(ct.reset_index())
    if "remaining_useful_life_hours" in df and "_fail" in df:
        g = df.groupby("_fail")["remaining_useful_life_hours"].describe().round(1).reset_index()
        st.write("RUL by observed failure label (RUL is a regression target, not a class):")
        table(g)

    # 7. failure_probability audit
    st.subheader("What is failure_probability? (calibration audit)")
    if "failure_probability" in df and "_fail" in df and df["_fail"].nunique() == 2:
        d = df[["failure_probability", "_fail"]].dropna()
        d = d.assign(decile=pd.qcut(d["failure_probability"].rank(method="first"), 10, labels=False) + 1)
        cal = d.groupby("decile").agg(n=("_fail", "size"), mean_pred=("failure_probability", "mean"), observed=("_fail", "mean")).reset_index()
        table(cal.round(4))
        fig = px.line(cal, x="mean_pred", y="observed", markers=True, title="Reliability curve (deciles of failure_probability)")
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
        show(fig, 360)
        from sklearn.metrics import brier_score_loss, roc_auc_score
        pc = d["failure_probability"].clip(0, 1)
        st.write(f"Brier score: {brier_score_loss(d['_fail'], pc):.4f} · ROC-AUC vs observed label: {roc_auc_score(d['_fail'], pc):.3f}")
        st.caption("If points sit far from the diagonal the column is a score, not a calibrated probability. "
                   "Whether it was generated from the label (leakage) cannot be determined from the CSV alone.")
    else:
        st.info("failure_probability or a two-class failure label is missing.")

    # 8. leakage
    st.subheader("Leakage review (before any supervised model)")
    flagged = [c for c in df.columns if any(t in c.lower() for t in LEAK_TOKENS) and not c.startswith("_")]
    st.write("Columns whose names suggest they may be outcomes or post-failure information — exclude or justify before predicting failure:")
    st.code(", ".join(flagged) or "none")
    st.write("Identifiers (machine_id, timestamp, row number) must not be used as predictive numeric features.")
    if "_fail" in df and df["_fail"].nunique() == 2 and st.checkbox("Run single-feature AUC scan (may take a few seconds)"):
        from sklearn.metrics import roc_auc_score
        s = df.sample(min(len(df), 50000), random_state=0)
        out = []
        for col in ctx["num"]:
            dd = s[[col, "_fail"]].dropna()
            if dd[col].nunique() > 1 and dd["_fail"].nunique() == 2:
                a = roc_auc_score(dd["_fail"], dd[col])
                out.append({"feature": col, "AUC": a, "|AUC-0.5|": abs(a - 0.5)})
        o = pd.DataFrame(out).sort_values("|AUC-0.5|", ascending=False).head(20)
        table(o.round(3))
        st.caption("A single feature with AUC near 0/1 deserves scrutiny as potential leakage (or a generator artefact).")
