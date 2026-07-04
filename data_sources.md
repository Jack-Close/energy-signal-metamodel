# Data Sources

This project consumes three raw inputs which are **not included in this repository**:

| File | Description | Why it's excluded |
|---|---|---|
| `ohlcv_data.csv` | Daily OHLCV history per instrument | Vendor-licensed market data |
| `primary_signals.csv` | Daily primary-model signal per instrument, in {-1, 0, +1} | Provided under a course/vendor agreement |
| `ovx.csv` | CBOE Crude Oil ETF Volatility Index | Sourced from a Bloomberg Terminal feed (16 May 2026 pull); Bloomberg Terminal data is licensed and redistribution is restricted |

The 3-2-1 crack spread proxy (`crack_321_proxy`) is computed directly from OHLCV log-returns and is not an external feed.

## Schema, to reproduce with your own data

Drop files matching the schemas below into `data/src/`, then run `notebooks/1_feature_engineering.ipynb`.

**`ohlcv_data.csv`** &mdash; one row per (instrument, date):

| Column | Description |
|---|---|
| date | Trading date (YYYY-MM-DD) |
| instrument | Lowercase ticker (e.g. `cl1s`, `ho1s`, `rb1s`, `ng1s`) |
| open, high, low, close | Continuous-contract prices |
| volume | Daily volume |
| open_interest | Daily open interest |

**`primary_signals.csv`** &mdash; one row per date, one column per instrument, values in {-1, 0, +1} (long / flat / short).

**`ovx.csv`** &mdash; daily volatility index level, any public proxy for crude oil implied volatility works (e.g. CBOE's published OVX series).

Processed features (`data/processed/features/`) and model outputs (`data/deliverables/`) are included in this repo, since they don't redistribute the licensed raw feed.
