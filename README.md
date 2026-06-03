# Metamodel for Triple-Barrier Trading Signals

## What this project does

We are given a primary trading model's daily signals (-1, 0, +1) and we build a metamodel that sits on top of them. For each signal the metamodel outputs a probability in [0, 1] that following the signal would be profitable under a triple-barrier exit rule. The primary model decides whether to bet, the metamodel decides whether the bet is worth taking.

The brief requires covering at least one full asset class. This submission covers the **Energy** asset class:

| Ticker | Instrument |
|---|---|
| cl1s | WTI crude oil |
| ho1s | Heating oil |
| rb1s | RBOB gasoline |
| ng1s | Natural gas |


## Repository structure

```
systematic_trading_2026_project/
  README.md                  this file
  requirements.txt           pinned dependencies (Python 3.10)
  original_coursework.md     the coursework brief
  data_sources.md            provenance of the Bloomberg-sourced inputs
  notebooks/
    1_feature_engineering.ipynb          builds the feature matrix
    2a_rbob_gasoline_model_training.ipynb metamodel for rb1s
    2b_heating_oil_model_training.ipynb   metamodel for ho1s
    2c_natural_gas_model_training.ipynb   metamodel for ng1s
    2d_wti_crude_oil_model_training.ipynb metamodel for cl1s
    3_final_pipeline.ipynb               orchestrator, builds the deliverable
    4_strategy_construction.ipynb        optional competition track
    sanity_checks.ipynb                  independent label and feature checks
  src/
    part5/                   helper modules for the strategy notebook
  data/
    src/                     raw inputs (ohlcv_data.csv, primary_signals.csv, ovx.csv)
    processed/features/      features.parquet, feature_dictionary.csv (built by notebook 1)
    deliverables/            predictions.csv and the optional weights.csv
```

## Setup

The code runs top to bottom on CPU under Python 3.10.

```
pip install -r requirements.txt
```

## How to run and reproduce the deliverable

`notebooks/3_final_pipeline.ipynb` is the single entry point. Open it and run all cells. It

1. runs every notebook in order, feature engineering (`1`) followed by the four per-instrument metamodels (`2a` to `2d`),
2. assembles the one required CSV at `data/deliverables/predictions.csv`,
3. prints the winning-model comparison and the prediction graphs.

The run-order cell exposes one switch:

- `RUN_NOTEBOOKS = True` does a clean end-to-end run, regenerating features and retraining every metamodel before rebuilding the deliverable.
- `RUN_NOTEBOOKS = False` skips the slow re-execution and rebuilds `predictions.csv` from the per-instrument CSVs already on disk.

The pipeline is fully seeded (Python, NumPy, TensorFlow and Keras, and every scikit-learn and XGBoost estimator all use seed 42), so a clean run reproduces the per-instrument prediction CSVs deterministically.

## Testing on the hidden H2 2022 data

The released data covers up to 30 June 2022. July to December 2022 is held out as the hidden test set. The deliverable `predictions.csv` covers H1 2022 (January to June). To test on the hidden window, drop the extended `ohlcv_data.csv` and `primary_signals.csv` into `data/src/`, set `RUN_NOTEBOOKS = True`, and run `3_final_pipeline.ipynb` top to bottom. No code edits are needed. The export window is data-driven (`EXPORT_END` is set from the last available signal date, not hard-coded), so a rerun on through-December data emits predictions for the whole hidden window. The models are not persisted to disk, but the train and test split is pinned to a fixed date (signals before 16 August 2021 train the model, everything from that date on is out of sample), so a rerun retrains the same model on the same window under the same seeds and H2 2022 stays strictly out of sample.

Predicted probabilities are computed only from features known at entry time, with no forward look-up. Triple-barrier labels in the last roughly 20 trading days of the data are dropped rather than computed on a truncated horizon, so the tail of the window is never labelled on a biased path. See Section 8 of each per-instrument notebook and `sanity_checks.ipynb` for the verification.

## Deliverables

**Required, metamodel predictions.** `data/deliverables/predictions.csv`, one row per (date, instrument, prediction), covering H1 2022 for the four energy instruments. `prediction` is the probability in [0, 1] that the primary signal is worth taking.

```
date,instrument,prediction
2022-01-03,cl1s,0.08
2022-01-03,ho1s,0.63
2022-01-03,ng1s,0.47
2022-01-03,rb1s,0.61
```

The per-instrument `predictions_<ticker>.csv` files are intermediate outputs of notebooks `2a` to `2d`, and `3_final_pipeline.ipynb` merges them into the combined file.

**Optional, strategy weights (competition track).** `notebooks/4_strategy_construction.ipynb` turns the metamodel probabilities into a position-sizing strategy and writes `data/deliverables/weights.csv`, one row per (date, instrument, weight), where weight is the signed position (positive long, negative short). This notebook is not part of the core pipeline in notebook 3 and is run separately.

## Methodology notes

- Walk-forward cross-validation uses chronological `TimeSeriesSplit`. Because consecutive triple-barrier labels overlap by up to 20 days, neighbouring folds can share label paths at the boundary. We use chronological splits for sample-size reasons and report this as a trade-off rather than a leak.
- The deliverable is raw probabilities, not thresholded, so no test-tuned decision threshold leaks into the shipped CSV. The operational threshold used inside the evaluation sections is selected on a leak-free Aug to Dec 2021 slice and frozen for Jan to Jun 2022.
- Per the brief, results are reported honestly. The evaluation sections flag where the metamodel does not beat the blind primary baseline on H1 2022 rather than hiding it.
