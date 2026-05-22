# Metamodel for Triple-Barrier Trading Signals

## Project Brief

We are given a primary trading model's daily signals (-1, 0, +1) for 11 instruments across three asset classes (equity index futures, energy, metals). Our job is to build a metamodel that sits on top of those signals and predicts, for each one, the probability in [0, 1] that following the signal would be profitable under a triple-barrier exit rule.

In short: the primary model decides *whether to bet*, our metamodel decides *whether the bet is worth taking*.

We must cover **at least one full asset class**. Covering more (up to all 11 instruments) is optional. Marking is focused on methodology and rigour, not on whether the metamodel beats the primary signal.

## Deadline

**4 June 2026**

## Files

All raw CSV files live in `data/raw`:

- `data/raw/ohlcv_data.csv` - daily OHLCV history for all 11 instruments, one row per (instrument, date). Columns: `date`, `instrument`, `open`, `high`, `low`, `close`, `volume`, `open_interest`. History starts in 1990 for most instruments (ES1S 1997, FESX1S 1998, NQ1S 1999).
- `data/raw/primary_signals.csv` - daily primary signals from January 2020 onwards, one row per date, one column per instrument. Values are in {-1, 0, +1}.

Important: the released data covers up to **30 June 2022**. July to December 2022 is held out as a hidden test set. Our code will be rerun on that hidden period.

## Steps (roughly)

1. **Feature engineering** (20 marks). Build a rich feature set from OHLCV and anything we can derive: technical indicators, latent variable models (GMM, HMM), unsupervised learning methods. Document what each feature is meant to capture.
2. **Labelling via the triple-barrier method** (20 marks). Apply triple-barrier labelling as taught in the course and justify our choice of barrier widths and time limit.
3. **Model development and comparison** (30 marks). Train and tune at least three models drawn from across the three families: linear (e.g. regularised logistic regression), tree-based (e.g. Random Forest, XGBoost, LightGBM), and neural networks. Present a clear comparison of which model wins, on which metric, and why.
4. **Cluster-level feature importance** (10 marks). Cluster correlated features, then apply MDI, MDA, or SHAP at the cluster level. Discuss which feature groups drive the metamodel.
5. **Model evaluation** (20 marks). Evaluate on a clean out-of-sample period carved from the training data. Report precision, recall, F1, and AUC, a confusion matrix and decision-threshold analysis, a per-instrument breakdown, and a comparison against a baseline that follows the primary signal blindly.
6. **(Optional) Strategy construction** (+10 bonus). Use the metamodel probabilities to build a position-sizing strategy. Full constraints (position limits, gross/net exposure, rebalancing, target volatility) are released on 20 May. Backtest metrics to report: CAGR, annualised volatility, Sharpe, Sortino, maximum drawdown, average holding period, and turnover.

## Deliverables

1. **Code**. A Jupyter notebook or set of Python files that runs end-to-end and produces the deliverable CSV files below. Clean, well-documented, reproducible code is part of the mark.
2. **Required: metamodel predictions**. A CSV covering January to June 2022, one row per (date, instrument, prediction).

   ```
   date,instrument,prediction
   2022-01-03,cl1s,0.74
   2022-01-03,es1s,0.51
   ```

   `prediction` is the probability in [0, 1] that the primary signal is worth taking.
3. **Optional: strategy weights** (competition track only). An additional CSV covering January to June 2022, one row per (date, instrument, weight).

   ```
   date,instrument,weight
   2022-01-03,cl1s,0.18
   2022-01-03,es1s,-0.05
   ```

   `weight` is the signed position weight (positive long, negative short).