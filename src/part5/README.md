# Part 5 — Model Evaluation

## What this is

The threshold / false-positive / precision-over-recall workstream for the
metamodel. Consumes the existing per-instrument model notebooks
AS-IS and adds the evaluation layer:

- frozen decision threshold per instrument, picked on a leak-free selection
  slice (NOT the deliverable window);
- per-instrument and aggregate precision / recall / F1 / AUC / AP;
- ROC, PR, threshold-sweep, confusion-matrix plots;
- blind-primary-signal baseline comparison;
- critical analysis: volatility-regime overlay (daily realised vol, `vol_20d × 100`,
  sweep cutoffs around 10%, plus combined `metamodel ∧ vol`);
- the required `deliverables/predictions.csv` (`date,instrument,prediction`
  for Jan-Jun 2022, probabilities — never thresholded);
- a tidy CSV + multi-sheet Excel results bundle under `results/part5/`.

## How to reproduce

From the repo root:

```bash
# 1. Populate the prediction cache. SLOW — runs each of the four
#    per-instrument model notebooks headlessly via nbclient and extracts
#    model_probas. Idempotent: skips notebooks whose cache parquet exists.
python -m src.part5.predict_cache

# 2. Run the evaluation notebook (notebooks/part5_evaluation.ipynb) end-to-end.
#    Outputs land in:
#      deliverables/predictions.csv      <-- the submission CSV
#      results/part5/*.csv               <-- per-table tidy exports
#      results/part5/part5_evaluation.xlsx
#      results/part5/figures/*.png
```

## Design notes

- **Threshold selection is leak-free.** The team's model notebooks only
  expose test-set probabilities (no CV OOF), so the threshold is picked on
  the first half of the internal test window (Aug-Dec 2021) and evaluated on
  Jan-Jun 2022 — the graded deliverable period. This is documented in §3 of
  the evaluation notebook.
- **Models are consumed as-is.** No retraining or retuning (Part 3 scope).
  `predict_cache.py` appends a probe cell to each model notebook that pickles
  the relevant locals; the original notebooks are not modified on disk.
- **Choice of evaluation rule:** F0.5 is the primary report (precision-biased
  F-beta). Target-precision (0.60) and cost-sensitive (2:1) are reported
  side-by-side so the chosen rule is defensible.
- **Vol overlay** uses **daily** realised vol in % (`vol_20d × 100`), not annualised.
  The teammate's standup wording was "fixed volatility threshold ≈ 10". Two
  conventions could put a vol number near 10: daily (WTI typical 1–5%, crisis
  spikes to ~20) or annualised `vol_20d × √252 × 100` (WTI typical 30–70%, min
  ~15%). A cutoff at 10 only filters meaningfully under the daily convention —
  under annualised it would filter out almost everything in this window. The
  daily-vol convention is what makes the teammate's "≈ 10" empirically
  actionable, so it is the one used. Two sweeps are reported: the around-10
  reference `{5, 7.5, 10, 12.5, 15, 20}` and a narrow sweep `{3, 4, 5, 6, 7.5,
  10}` that brackets the empirical 1–8% range of the deliverable window. Both
  `daily_vol_pct` and `annualised_vol_pct` are exposed in `vol_overlay.py` so
  the choice is reversible.

## What was deliberately NOT done

- Did not modify any teammate notebook or feature module.
- Did not retrain or retune any model.
- Did not change the team's 70/30 chronological split.
- Did not select the decision threshold on the graded deliverable window.
- Did not push or force-push anything. All work is on local branch
  `part5-evaluation-rr`.
