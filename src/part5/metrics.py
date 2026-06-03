"""Classification metrics for the meta-labeling evaluation.

Positive class = "primary signal's trade is profitable" (meta_label == 1).
False positive = bad trade we took. False negative = good trade we skipped.
A trade filter favours precision over recall; metrics here are framed accordingly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PointMetrics:
    """Metrics evaluated at a single decision threshold."""

    threshold: float
    n_trades: int
    pct_taken: float
    precision: float
    recall: float
    f1: float
    auc: float
    average_precision: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_row(self, **extra: object) -> dict[str, object]:
        return {**self.__dict__, **extra}


def compute_point_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> PointMetrics:
    """Compute the full Part-5 metric tuple at one threshold.

    Notes
    -----
    `auc` and `average_precision` are threshold-free ranking metrics, so they
    are constant across thresholds for a given (y_true, proba) pair. They are
    repeated on each row for convenience in the exported sweep tables.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)

    n_trades = int(pred.sum())
    pct_taken = float(n_trades / len(pred)) if len(pred) else float("nan")

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return PointMetrics(
        threshold=float(threshold),
        n_trades=n_trades,
        pct_taken=pct_taken,
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        auc=float(_safe_auc(y_true, proba)),
        average_precision=float(average_precision_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def threshold_sweep(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Sweep the decision threshold from 0 to 1 and tabulate metrics.

    Parameters
    ----------
    thresholds : iterable of float, optional
        Defaults to np.arange(0.05, 1.0, 0.05).

    Returns
    -------
    pd.DataFrame
        One row per threshold, columns from PointMetrics.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    rows = [compute_point_metrics(y_true, proba, t).to_row() for t in thresholds]
    return pd.DataFrame(rows)


def roc_curve_points(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Return ROC curve as a DataFrame with columns: fpr, tpr, threshold."""
    fpr, tpr, thr = roc_curve(y_true, proba)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})


def pr_curve_points(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Return precision-recall curve as a DataFrame: precision, recall, threshold."""
    precision, recall, thr = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns precision/recall of length T+1, thr of length T
    thr = np.append(thr, np.nan)
    return pd.DataFrame({"precision": precision, "recall": recall, "threshold": thr})


def fbeta(y_true: np.ndarray, proba: np.ndarray, threshold: float, beta: float = 0.5) -> float:
    """F-beta at a threshold. beta<1 weights precision above recall."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    return float(fbeta_score(y_true, pred, beta=beta, zero_division=0))


def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """ROC AUC, returning NaN when only one class is present (undefined)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, proba)
