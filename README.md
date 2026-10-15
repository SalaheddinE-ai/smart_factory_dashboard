# Smart Factory — Predictive Maintenance Analytics Platform

Interactive Streamlit dashboard for the Kaggle dataset
[Smart Factory Predictive Maintenance](https://www.kaggle.com/datasets/mobeenfatimah/smart-factory-predictive-maintenance-dataset)
(described as ~100k records, ~5,000 machines, 72 features; **synthetic data**).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then: upload the CSV in the sidebar (or paste a local path — recommended for large files; otherwise run
`streamlit run app.py --server.maxUploadSize 500`) → open **0 · Data Inspection** first → explore pages and filters.

Download the data from Kaggle manually, or: `kaggle datasets download -d mobeenfatimah/smart-factory-predictive-maintenance-dataset --unzip`.

## Design principles

* **Inspect first.** Page 0 reports shape, dtypes, missing values, duplicates, invalid ranges, label consistency
  (failure vs failure_type vs priority), machine/timestamp structure, class imbalance, a `failure_probability`
  calibration audit, and a leakage review (name-based + optional single-feature AUC scan).
* **No assumed schema.** Logical columns (e.g. `machine_failure`, `machine_health_score`) are auto-matched to the real header
  (exact → normalised → alias). Fix any mismatch in the sidebar *Column mapping*. Sensors, component-health and
  anomaly-score columns are **discovered by name patterns**, not hard-coded. Missing columns disable only the charts that need them.
* **No silent changes.** Rows are never dropped; labels are never edited. Derived helpers (`_fail` numeric label, `_row` CSV line
  number, health/age/maintenance bins) are added as new columns.
* **Correct denominators.** Record-level vs machine-level failure rates are separate; Wilson 95% CIs and minimum-n thresholds
  guard small groups.
* **Machine-aware time.** Records sorted by (machine, timestamp); latest observation chosen by timestamp; machines are never
  joined into one series (fleet charts aggregate per period; the explorer plots one machine).
* **Observation ≠ causation.** Every chart carries a *Measures / Cannot establish* caption; Maintenance page makes no repair recommendation.

## Pages

| Page | Content |
|---|---|
| 0 Data Inspection | schema, quality, consistency, calibration, leakage |
| 1 Executive Overview | 6 KPIs, failure/priority/health/RUL/failure-type charts, health vs probability |
| 2 Sensor Analytics | grouped sensors, distributions, correlation heatmap, temperature comparison, pairwise, segments, health/RUL relations |
| 3 Failure Analysis | rates by type/manufacturer/line/factory, failure-type × type, observed vs predicted, age, maintenance recency, anomaly vs label |
| 4 Temporal & Operational | fleet trends, load vs energy, downtime, maintenance activity, shifts, single-machine explorer |
| 5 Health & Components | component comparison, heatmap, radar, vs health/probability, investigation ranking, machine detail |
| 6 Anomaly Detection | score audit, flagged vs unflagged, PCA, investigation table, optional Isolation Forest (kept separate from dataset scores and labels) |
| 7 Maintenance Intelligence | cost, downtime, history vs failures, priority mix, prioritised investigation table |

## Assumptions to review

* Health bands (Critical/Poor/Fair/Good = <40/40–60/60–80/≥80 % of scale) are display conventions.
* The *investigation index* (mean percentile rank of risk signals on each machine's latest record) is a transparent triage heuristic.
* The "anomalous" flag uses a user-set percentile rule unless the CSV has its own binary flag column.
* Maintenance activity over time uses `maintenance_type` counts; without it, a clearly labelled proxy is shown.

## Structure

```
app.py                 routing, upload, column mapping, sidebar
src/config.py          aliases, sensor-group rules
src/data.py            cached loading, preprocessing, schema discovery
src/filters.py         sidebar filters + reset
src/utils.py           stats (Wilson CI), binning, machine-level logic, plot helpers
src/views/*.py         one module per page
```

## Testing status

Developed and smoke-tested (all pages, filters, Isolation Forest, leakage scan) against a **synthetic stand-in CSV with
assumed column names** because the Kaggle file could not be downloaded in the build environment. Run it on the real file and
check page 0 and the *Column mapping* panel first.
