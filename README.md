# Metamodel for Triple-Barrier Trading Signals

## Project Brief

We are given a primary trading model's daily signals (-1, 0, +1) for 11 instruments across three asset classes (equity index futures, energy, metals). Our job is to build a metamodel that sits on top of those signals and predicts, for each one, the probability in [0, 1] that following the signal would be profitable under a triple-barrier exit rule.

In short: the primary model decides *whether to bet*, our metamodel decides *whether the bet is worth taking*.

We must cover **at least one full asset class**. Covering more (up to all 11 instruments) is optional. Marking is focused on methodology and rigour, not on whether the metamodel beats the primary signal.

This submission covers the **Energy** asset class: WTI crude oil (`cl1s`), heating oil (`ho1s`), RBOB gasoline (`rb1s`) and natural gas (`ng1s`).

## Deadline

**4 June 2026**

## Notebook structure

The submission is split into one feature-engineering notebook, four per-instrument model notebooks, one orchestrator notebook, and one sanity-checks notebook. This layout was confirmed acceptable by the course lecturer:

> [LECTURER QUOTE — paste exact wording from the email/message that approved
> the per-instrument + orchestrator split, with date. Replace this placeholder
> verbatim before submission.]

The notebooks, in execution order:

1. **`notebooks/1_feature_engineering.ipynb`** — loads OHLCV and primary signals, builds the four-tier feature matrix (base, energy-domain, walk-forward latent, primary-signal-aware), and writes `data/processed/features/features.parquet` plus `data/processed/features/feature_dictionary.csv`. Walk-forward latent models (GMM, HMM, K-means) use a 504-day warmup and refit every 63 trading days; the HMM uses an explicit hand-rolled forward algorithm rather than `hmmlearn.predict_proba`'s smoother to avoid look-ahead.
2. **`notebooks/2a_rbob_gasoline_model_training.ipynb`** — RBOB gasoline metamodel. Winner: **MLP** (parsimony pick inside the variance-aware CV competitive tier; full rationale in §11).
3. **`notebooks/2b_heating_oil_model_training.ipynb`** — heating oil metamodel. Winner: **Logistic Regression** (deliberate override; full rationale in §11 — the 27-trade test set is statistically uninformative so the chosen model is the most robust low-capacity fallback).
4. **`notebooks/2c_natural_gas_model_training.ipynb`** — natural gas metamodel. Winner: **Logistic Regression** (only model inside the variance-aware CV tier).
5. **`notebooks/2d_wti_crude_oil_model_training.ipynb`** — WTI crude oil metamodel. Winner: **Logistic Regression** (parsimony tie-break against the LSTM after both cleared the variance-adjusted lower bound).
6. **`notebooks/3_final_pipeline.ipynb`** — orchestrator. Runs notebooks 1 → 2a → 2b → 2c → 2d and assembles the four per-instrument prediction CSVs into the single required `data/deliverables/predictions.csv`. Set `RUN_NOTEBOOKS = False` to skip the slow re-execution and just rebuild the combined CSV from existing per-instrument files.
7. **`notebooks/sanity_checks.ipynb`** — independent verification: reconstructs labels, checks that no trading-day horizon exceeds `max_hold`, verifies feature shift (no same-day return look-ahead), reports class imbalance, runs ADF/KPSS stationarity per feature, and checks for outliers.

## Files

All raw CSV files live in `data/src/`:

- `data/src/ohlcv_data.csv` — daily OHLCV history for all 11 instruments, one row per (instrument, date). Columns: `date`, `instrument`, `open`, `high`, `low`, `close`, `volume`, `open_interest`. History starts in 1990 for most instruments (ES1S 1997, FESX1S 1998, NQ1S 1999).
- `data/src/primary_signals.csv` — daily primary signals from January 2020 onwards, one row per date, one column per instrument. Values are in {-1, 0, +1}.

Additional Bloomberg-sourced files (10Y yield, 3-2-1 crack spread, OVX) are documented in `data_sources.md` and live in `data/src/` alongside the OHLCV.

Important: the released data covers up to **30 June 2022**. July to December 2022 is held out as a hidden test set. Our code will be rerun on that hidden period — confirmed by the course lecturer:

> [LECTURER QUOTE — paste exact wording from the email/message confirming
> that the marker reruns the pipeline on hidden Jul–Dec 2022 data, with
> date. Replace this placeholder verbatim before submission.]

See the "Reproducibility" and "Label edge handling" sections below for what this means for the pipeline.

## Reproducibility

The submission is fully seeded for deterministic top-to-bottom execution on CPU. Every per-instrument model notebook (`2a`–`2d`) sets the following at import time, identically:

```python
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)
keras.utils.set_random_seed(42)              # seeds Python, NumPy and TensorFlow together
tf.config.experimental.enable_op_determinism()
```

Each neural-network builder (`make_mlp`, `make_vsn`, `make_lstm`) re-seeds via `keras.utils.set_random_seed(42)` so weight initialisation is independent of cell run-order. Every sklearn estimator and XGBoost classifier is passed `random_state=42`. The walk-forward GMM/HMM/K-means in the feature pipeline pass `random_state=42` and the GMM/HMM use post-fit permutation to stabilise component labels across refits.

A clean top-to-bottom run (`notebooks/3_final_pipeline.ipynb` with `RUN_NOTEBOOKS = True`) reproduces the exported `data/deliverables/predictions_<ticker>.csv` files byte-for-byte under these seeds. The same seeds apply when the marker reruns the pipeline on the hidden H2 2022 window.

## Label edge handling

The triple-barrier labeller in §8 of each per-instrument notebook uses `max_hold = 20` trading days. Where a primary signal fires inside the last ~20 bars of the available data, the vertical barrier collapses to the data end (`t1 = close.index[min(idx + max_hold, len(close.index) - 1)]`), so the label is computed on a truncated, shorter horizon than the rest. Truncated-horizon labels are biased toward "time barrier first" outcomes (the `0` class) because the path has fewer days to hit the profit-take.

**On the released window (2020-01 to 2022-06-30):** this affects roughly the last 20 trading days of June 2022 in each instrument. Those biased labels enter the training and selection-slice statistics but not the H1 2022 deliverable export (the export is keyed on the primary-signal index of the deliverable window, so the label horizon is internal to the model's training, not the predicted probabilities).

**On the hidden H2 2022 window:** the same truncation occurs in the last ~20 trading days of December 2022 when the marker reruns the pipeline. Predicted probabilities for those days are unaffected (they're computed from features known up to entry time, with no forward look-up); only the *labels* used for any model-evaluation step inside §11.1 are biased on that sub-slice. The §11.1 frozen-threshold evaluation on H2 2022 reports the affected day-count alongside the metric so the bias is visible rather than hidden.

[OPTIONAL: if you also apply the document-and-fix variant rather than document-only, mention here: "Labels whose vertical barrier would exceed the data end are dropped from the training set rather than computed on a truncated horizon. See `triple_barrier_meta_labels` in §8 of each model notebook."]

## Methodology limitations

- **Walk-forward CV is chronological, not purged/embargoed.** All cross-validation uses `sklearn.model_selection.TimeSeriesSplit(n_splits=5)`. Because consecutive triple-barrier labels overlap (up to `max_hold = 20` days), training and validation folds share label paths at the fold boundary — the López de Prado §7 purged/embargoed scheme would shrink the effective training set further but eliminate that overlap. We use chronological splits for sample-size reasons and report this as a methodology trade-off rather than as a leak.
- **Threshold sweep in §10 is on the full test set.** It is informational; the *operational* threshold used in §11.1 is selected on a leak-free Aug-Dec 2021 selection slice and frozen for the Jan-Jun 2022 OOS slice. The deliverable is raw probabilities (not thresholded), so no test-tuned threshold leaks into the shipped CSV.

## Steps and marking weights (per coursework brief)

1. **Feature engineering** (20 marks). Build a rich feature set from OHLCV and anything we can derive: technical indicators, latent variable models (GMM, HMM), unsupervised learning methods. Document what each feature is meant to capture.
2. **Labelling via the triple-barrier method** (20 marks). Apply triple-barrier labelling as taught in the course and justify our choice of barrier widths and time limit.
3. **Model development and comparison** (30 marks). Train and tune at least three models drawn from across the three families: linear (e.g. regularised logistic regression), tree-based (e.g. Random Forest, XGBoost, LightGBM), and neural networks. Present a clear comparison of which model wins, on which metric, and why.
4. **Cluster-level feature importance** (10 marks). Cluster correlated features, then apply MDI, MDA, or SHAP at the cluster level. Discuss which feature groups drive the metamodel.
5. **Model evaluation** (20 marks). Evaluate on a clean out-of-sample period carved from the training data. Report precision, recall, F1, and AUC, a confusion matrix and decision-threshold analysis, a per-instrument breakdown, and a comparison against a baseline that follows the primary signal blindly.
6. **(Optional) Strategy construction** (+10 bonus). Use the metamodel probabilities to build a position-sizing strategy. Full constraints (position limits, gross/net exposure, rebalancing, target volatility) are released on 20 May. Backtest metrics to report: CAGR, annualised volatility, Sharpe, Sortino, maximum drawdown, average holding period, and turnover.

## Deliverables

1. **Code.** The Jupyter notebooks listed above run end-to-end and produce the deliverable CSV files below. Reproducibility, structure and documentation are described in the "Reproducibility" and "Notebook structure" sections.
2. **Required: metamodel predictions.** A single CSV covering January to June 2022, one row per (date, instrument, prediction), at `data/deliverables/predictions.csv`.

   ```
   date,instrument,prediction
   2022-01-03,cl1s,0.08
   2022-01-03,ho1s,0.63
   2022-01-03,ng1s,0.47
   2022-01-03,rb1s,0.61
   ```

   `prediction` is the probability in [0, 1] that the primary signal is worth taking. The per-instrument files `predictions_<ticker>.csv` are intermediate outputs of notebooks `2a`–`2d`; the combined `predictions.csv` is assembled by `3_final_pipeline.ipynb` §2.
3. **Optional: strategy weights** (competition track only). An additional CSV covering January to June 2022, one row per (date, instrument, weight).

   ```
   date,instrument,weight
   2022-01-03,cl1s,0.18
   2022-01-03,es1s,-0.05
   ```

   `weight` is the signed position weight (positive long, negative short).
