"""Blind-primary-signal baseline.

The naive trader takes every non-zero primary signal. The metamodel filter
should — if it adds value — deliver higher precision at the cost of recall.
This module quantifies the delta.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import metrics as M

logger = logging.getLogger(__name__)


def blind_primary_baseline(meta_labels: pd.DataFrame) -> M.PointMetrics:
    """Treat the labeled set as the "blind" strategy.

    `meta_labels` is the output of triple_barrier_meta_labels: every row
    corresponds to a non-zero primary signal that was actually traded.
    Predicting "take every trade" is equivalent to proba=1.0 for every row.

    Returns PointMetrics computed at threshold=0.5 (which is irrelevant since
    every prediction is 1, but kept for schema consistency).
    """
    if "meta_label" not in meta_labels.columns:
        raise ValueError("expected a 'meta_label' column in meta_labels")
    y = meta_labels["meta_label"].to_numpy().astype(int)
    proba = np.ones_like(y, dtype=float)
    pm = M.compute_point_metrics(y, proba, threshold=0.5)
    logger.info("blind baseline: n=%d, precision=%.3f (hit rate)", pm.n_trades, pm.precision)
    return pm


def baseline_vs_filter_table(
    baseline_pm: M.PointMetrics,
    filtered_pm: M.PointMetrics,
    label: str,
) -> pd.DataFrame:
    """Side-by-side baseline vs metamodel-filtered trades."""
    return pd.DataFrame([
        {
            "strategy": "blind_primary",
            "n_trades": baseline_pm.n_trades,
            "precision": baseline_pm.precision,
            "recall": baseline_pm.recall,
            "f1": baseline_pm.f1,
        },
        {
            "strategy": label,
            "n_trades": filtered_pm.n_trades,
            "precision": filtered_pm.precision,
            "recall": filtered_pm.recall,
            "f1": filtered_pm.f1,
        },
    ])
