"""Volatility-regime overlay — naive critical-analysis comparator.

Rule
----
Trade the primary signal only when annualised realised volatility is BELOW a
cutoff. The intuition: avoid taking primary-signal trades during turbulent
regimes; the primary model is presumed more reliable in calm markets.

Units
-----
Two conventions are exposed:

  - `daily_vol_pct`        : vol_20d * 100. For WTI this typically sits in
                             [1, 5] with crisis spikes to ~20. The natural
                             cutoff "around 10" corresponds to this convention
                             (matches the teammate's standup wording).
  - `annualised_vol_pct`   : vol_20d * sqrt(252) * 100. The textbook VIX-style
                             unit, typically 15-100 for crude. A cutoff of 10
                             here would filter out everything; not useful.

The overlay defaults to daily_vol_pct because that's what makes "≈ 10"
empirically meaningful. The choice is documented in the notebook so the
marker can see we chose between options rather than asserted one.

This is intentionally a hard cutoff with no learning — it sweeps so that the
choice of 10 is defended rather than asserted, and so that the metamodel
comparison is honest.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from . import metrics as M

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def daily_vol_pct(vol_20d: pd.Series) -> pd.Series:
    """vol_20d (daily logret std, fraction) → daily realised vol in pct points."""
    return vol_20d * 100.0


def annualised_vol_pct(vol_20d: pd.Series) -> pd.Series:
    """vol_20d (daily logret std, fraction) → annualised vol in pct points."""
    return vol_20d * np.sqrt(TRADING_DAYS) * 100.0


def vol_filter_mask(annual_vol: pd.Series, cutoff: float, direction: str = "below") -> pd.Series:
    """Boolean mask: True where the regime is favourable (trade allowed).

    direction='below': trade only when vol < cutoff (calm regime).
    direction='above': trade only when vol > cutoff (rarely useful; kept for completeness).
    """
    if direction == "below":
        return annual_vol < cutoff
    if direction == "above":
        return annual_vol > cutoff
    raise ValueError(f"direction must be 'below' or 'above', got {direction!r}")


def overlay_sweep(
    y_true: pd.Series,
    proba: pd.Series,
    annual_vol: pd.Series,
    base_threshold: float,
    cutoffs: Iterable[float] = (5.0, 7.5, 10.0, 12.5, 15.0, 20.0),
    direction: str = "below",
) -> pd.DataFrame:
    """Sweep vol cutoffs and compare against the metamodel filter.

    For each cutoff, three rows are produced:
      - vol_only        : take signal iff vol filter passes
      - metamodel_only  : take signal iff metamodel proba >= base_threshold
      - combined        : take signal iff BOTH pass

    metamodel_only is constant across cutoffs but repeated for ease of plotting.
    """
    y_true = y_true.astype(int)
    common_idx = y_true.index.intersection(proba.index).intersection(annual_vol.index)
    if len(common_idx) == 0:
        raise ValueError("y_true, proba, annual_vol have no common index")

    y = y_true.loc[common_idx].to_numpy()
    p = proba.loc[common_idx].to_numpy()
    v = annual_vol.loc[common_idx]

    meta_pred = (p >= base_threshold).astype(int)
    meta_pm = M.compute_point_metrics(y, p, threshold=base_threshold)

    rows = []
    for cutoff in cutoffs:
        vol_pass = vol_filter_mask(v, cutoff=cutoff, direction=direction).to_numpy().astype(int)

        # vol-only: take every signal where vol passes.
        # Equivalent to "proba" = vol_pass, threshold = 0.5.
        vol_pm = M.compute_point_metrics(y, vol_pass.astype(float), threshold=0.5)

        # combined: AND of metamodel and vol filter
        combined_pred = (meta_pred & vol_pass).astype(float)
        combined_pm = M.compute_point_metrics(y, combined_pred, threshold=0.5)

        rows.extend([
            {"vol_cutoff": cutoff, "strategy": "vol_only",
             **_pm_dict(vol_pm)},
            {"vol_cutoff": cutoff, "strategy": "metamodel_only",
             **_pm_dict(meta_pm)},
            {"vol_cutoff": cutoff, "strategy": "metamodel_AND_vol",
             **_pm_dict(combined_pm)},
        ])
    return pd.DataFrame(rows)


def _pm_dict(pm: M.PointMetrics) -> dict:
    return {
        "n_trades": pm.n_trades,
        "precision": pm.precision,
        "recall": pm.recall,
        "f1": pm.f1,
        "tp": pm.tp, "fp": pm.fp, "fn": pm.fn, "tn": pm.tn,
    }
