from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..utils import FAIL_COLORS, explain, fail_label, need, sample, show, table, wilson


def render(df: pd.DataFrame, ctx: dict):
    st.header("Anomaly Detection")
    scores = ctx["scores"]
    if not scores:
        st.warning("No anomaly score columns detected (names containing 'anomaly').")
        return

    st.subheader("1 · Score definitions and distributions")
    table(df[scores].describe().T.reset_index().rename(columns={"index": "score"}).round(4))
    st.caption("Check the min/max above: the scale and meaning of each score are not documented in the CSV. Nothing here assumes a score of X means failure.")
    m = sample(df[scores], 20000).melt(var_name="score", value_name="value")
    fig = px.histogram(m, x="value", facet_col="score", facet_col_wrap=3, nbins=50)
    fig.update_xaxes(matches=None)
    fig.update_yaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    show(fig, 300 * int(np.ceil(len(scores) / 3)) + 60)

    st.subheader("2 · Anomalous vs non-anomalous observations")
    if ctx["flag"] and st.checkbox(f"Use dataset flag column '{ctx['flag']}'", value=True):
        flag = df[ctx["flag"]].astype(float) > 0
        how = f"dataset flag '{ctx['flag']}'"
    else:
        p = st.slider("Flag a record when ANY score is at/above this percentile of its own distribution", 80, 99, 95)
        thr = {c: df[c].quantile(p / 100) for c in scores}
        flag = pd.concat([df[c] >= thr[c] for c in scores], axis=1).any(axis=1)
        how = f"user-defined rule (any score ≥ P{p})"
    st.caption(f"Definition used: {how}. Flagged: {int(flag.sum()):,} of {len(df):,} records ({flag.mean():.1%}).")
    if flag.nunique() < 2:
        st.info("Only one class under this definition; adjust the percentile.")
    else:
        sens = ctx["all_sensors"]
        if sens:
            a, b = df.loc[flag, sens], df.loc[~flag, sens]
            sd = pd.concat([a, b]).std().replace(0, np.nan)
            smd = ((a.mean() - b.mean()) / sd).dropna().sort_values(key=abs, ascending=False).head(15)
            fig = px.bar(smd.sort_values().reset_index(), x=0, y="index", orientation="h", labels={"0": "Standardised mean difference", "index": "sensor"},
                         title="Sensors differing most between flagged and unflagged records")
            show(fig)
        if "_fail" in df:
            rows = []
            for name, msk in [("Flagged", flag), ("Not flagged", ~flag)]:
                d = df.loc[msk, "_fail"].dropna()
                lo, hi = wilson(d.sum(), len(d))
                rows.append({"group": name, "n": len(d), "failure rate": d.mean(), "CI low": float(lo), "CI high": float(hi)})
            table(pd.DataFrame(rows).round(4))
        explain("Sensor profile and observed failure rate of flagged vs unflagged records.",
                "That flagged records are true faults: a high score is not a confirmed failure.")

    st.subheader("3 · Multivariate view (PCA)")
    sens = ctx["all_sensors"]
    if len(sens) >= 3:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        feats = st.multiselect("Sensors for PCA", sens, default=sens[:12])
        if len(feats) >= 3:
            d = df[feats].dropna()
            idx = d.sample(min(len(d), 10000), random_state=0).index
            X = StandardScaler().fit_transform(d.loc[idx])
            pca = PCA(2, random_state=0).fit(X)
            Z = pca.transform(X)
            col = st.radio("Colour by", ["Anomaly flag", "Observed failure"] if "_fail" in df else ["Anomaly flag"], horizontal=True)
            colour = flag.loc[idx].map({True: "Flagged", False: "Not flagged"}) if col == "Anomaly flag" else fail_label(df.loc[idx])
            fig = px.scatter(x=Z[:, 0], y=Z[:, 1], color=colour.values, opacity=0.4, render_mode="webgl",
                             labels={"x": f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", "y": f"PC2 ({pca.explained_variance_ratio_[1]:.0%})"},
                             title=f"PCA of standardised sensors (sample {len(idx):,})")
            show(fig, 480)
            explain("2-D projection of standardised sensors.", "Structure outside the first two components; PCA is linear.")
    else:
        st.info("Needs at least 3 sensor columns.")

    st.subheader("4 · Anomaly investigation table")
    topn = st.slider("Rows", 10, 300, 30, key="an_rows")
    comp = df[scores].rank(pct=True).max(axis=1)
    keys = [c for c in ["_row", "machine_id", "timestamp"] if c in df]
    show_s = [v[0] for v in ctx["sensors"].values()][:6]
    out = df.assign(composite_rank=comp)[keys + ["composite_rank"] + scores + show_s + (["_fail"] if "_fail" in df else [])]
    table(out.sort_values("composite_rank", ascending=False).head(topn))
    st.caption("Ranked by the highest within-score percentile. `_row` = CSV line number. `_fail` = observed label (shown for context, not used for ranking).")

    st.subheader("5 · Optional Isolation Forest (unsupervised, separate from dataset scores)")
    with st.expander("Configure and run"):
        feats = st.multiselect("Features (raw sensors only — scores, health, RUL, probabilities and labels are excluded)", sens, default=sens[:15], key="if_feats")
        c1, c2, c3 = st.columns(3)
        cont = c1.slider("Contamination", 0.005, 0.2, 0.05)
        nest = c2.slider("Trees", 50, 300, 100)
        fit_n = c3.slider("Max fitting rows", 5000, 100000, 50000, step=5000)
        if st.button("Run Isolation Forest") and len(feats) >= 2:
            from sklearn.ensemble import IsolationForest
            d = df[feats]
            d = d.fillna(d.median())
            mdl = IsolationForest(n_estimators=nest, contamination=cont, random_state=0, n_jobs=-1)
            mdl.fit(d.sample(min(len(d), fit_n), random_state=0))
            st.session_state["if_res"] = (pd.Series(mdl.predict(d) == -1, index=d.index), pd.Series(-mdl.score_samples(d), index=d.index), tuple(feats))
        res = st.session_state.get("if_res")
        if res is not None and res[0].index.isin(df.index).all() and len(res[0]) == len(df):
            iff, isc, used = res
            st.write(f"Model flagged **{int(iff.sum()):,}** records ({iff.mean():.1%}) using {len(used)} features. "
                     "This is a model prediction — not a dataset anomaly score and not a failure label.")
            if "_fail" in df:
                d = df["_fail"]
                ok = d.notna()
                tp = int((iff & (d == 1)).sum())
                prec = tp / max(int(iff.sum()), 1)
                rec = tp / max(int((d == 1).sum()), 1)
                base = d.mean()
                table(pd.DataFrame([{"IF-flag ∩ observed failure": tp, "precision (descriptive)": prec, "recall (descriptive)": rec,
                                     "base failure rate": base, "lift": prec / base if base else np.nan}]).round(4))
            if flag.nunique() == 2:
                st.write(f"Agreement with the flag defined in step 2: {(iff == flag).mean():.1%}; overlap of flagged records: {int((iff & flag).sum()):,}.")
            st.caption("The model never sees failure labels, so there is no label leakage in fitting; interpretation caveat: the "
                       "features may still be influenced by failures that already occurred, and the data is synthetic.")
        elif res is not None:
            st.info("Filters changed since the last run — run the model again.")
