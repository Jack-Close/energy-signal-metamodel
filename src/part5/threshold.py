"""Decision-threshold selection rules.

LEAKAGE DISCIPLINE
------------------
All selectors here are intended to be called on TRAINING (out-of-fold) data.
The chosen threshold is then frozen and applied to the test set in a separate
step. Selecting on the test set inflates apparent test precision and is a
serious methodology error in meta-labeling; see e.g. López de Prado (2018) §3.

Three rules are provided so the choice can be compared rather than asserted:

  - fbeta            : argmax F-beta with beta < 1 (default 0.5) — biases
                       toward precision while still rewarding some recall.
  - target_precision : lowest threshold meeting a stated precision floor (e.g.
                       0.60). Operational reading: "I will only take trades
                       the model is at least 60% confident about, given
                       training-time evidence."
  - cost_sensitive   : minimise expected cost given asserted costs of
                       false-positive (bad trade) and false-negative
                       (missed trade). FP > FN cost is the standard meta-label
                       assumption: bad trades bleed money, skipped trades have
                       opportunity cost only.

Each selector also returns metadata describing the rule, so the notebook can
state explicitly *which* rule produced the frozen threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from . import metrics as M

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThresholdChoice:
    """A selected threshold plus the rule that produced it."""

    threshold: float
    rule: str
    rule_detail: str
    train_precision: float
    train_recall: float
    train_f1: float
    train_n_trades: int


def _candidate_thresholds(n: int = 99) -> np.ndarray:
    """Dense grid over (0, 1) for selection. n=99 → step 0.01."""
    return np.linspace(0.01, 0.99, n)


def select_fbeta(
    y_true: np.ndarray,
    proba: np.ndarray,
    beta: float = 0.5,
    grid: Iterable[float] | None = None,
) -> ThresholdChoice:
    """Pick threshold that maximises F-beta on the input data.

    The input is expected to be train/OOF — never the test set.
    """
    grid = np.asarray(list(grid)) if grid is not None else _candidate_thresholds()
    best_score, best_t = -np.inf, float("nan")
    for t in grid:
        s = M.fbeta(y_true, proba, threshold=t, beta=beta)
        if s > best_score:
            best_score, best_t = s, float(t)
    pm = M.compute_point_metrics(y_true, proba, threshold=best_t)
    logger.info("fbeta(beta=%.2f) → threshold=%.3f (F%g=%.3f)", beta, best_t, beta, best_score)
    return ThresholdChoice(
        threshold=best_t,
        rule=f"fbeta",
        rule_detail=f"argmax F_beta with beta={beta}",
        train_precision=pm.precision,
        train_recall=pm.recall,
        train_f1=pm.f1,
        train_n_trades=pm.n_trades,
    )


def select_target_precision(
    y_true: np.ndarray,
    proba: np.ndarray,
    target: float = 0.60,
    min_trades: int = 10,
    grid: Iterable[float] | None = None,
) -> ThresholdChoice:
    """Lowest threshold achieving precision >= target on input data.

    `min_trades` guards against degenerate solutions where precision = 1
    because only one trade is taken. If no threshold qualifies, falls back
    to the threshold that maximises precision subject to the min_trades guard
    and logs a warning so the notebook can flag it.
    """
    grid = np.asarray(list(grid)) if grid is not None else _candidate_thresholds()
    qualifying: list[tuple[float, M.PointMetrics]] = []
    for t in grid:
        pm = M.compute_point_metrics(y_true, proba, threshold=float(t))
        if pm.precision >= target and pm.n_trades >= min_trades:
            qualifying.append((float(t), pm))

    if qualifying:
        # Lowest threshold = take the most trades that still meet the bar
        t, pm = min(qualifying, key=lambda x: x[0])
        detail = f"min threshold with precision>={target} and n_trades>={min_trades}"
    else:
        # Fallback: best precision under min_trades
        best_t, best_pm = float("nan"), None
        best_p = -np.inf
        for t in grid:
            pm = M.compute_point_metrics(y_true, proba, threshold=float(t))
            if pm.n_trades >= min_trades and pm.precision > best_p:
                best_p, best_t, best_pm = pm.precision, float(t), pm
        t, pm = best_t, best_pm
        logger.warning("target_precision=%.2f infeasible (min_trades=%d); fallback t=%.3f, p=%.3f",
                       target, min_trades, t, pm.precision if pm else float("nan"))
        detail = f"target {target} infeasible; fallback to max precision with n_trades>={min_trades}"

    logger.info("target_precision(%.2f) → threshold=%.3f (precision=%.3f, n=%d)",
                target, t, pm.precision, pm.n_trades)
    return ThresholdChoice(
        threshold=float(t),
        rule="target_precision",
        rule_detail=detail,
        train_precision=pm.precision,
        train_recall=pm.recall,
        train_f1=pm.f1,
        train_n_trades=pm.n_trades,
    )


def select_cost_sensitive(
    y_true: np.ndarray,
    proba: np.ndarray,
    cost_fp: float = 2.0,
    cost_fn: float = 1.0,
    grid: Iterable[float] | None = None,
) -> ThresholdChoice:
    """Minimise expected cost = cost_fp * FP + cost_fn * FN on input data.

    Default cost_fp:cost_fn = 2:1 reflects the meta-label intuition that bad
    trades hit P&L (taking a loss after fees) whereas missed trades only forgo
    expected profit. Adjust to taste; the rule is reported with the choice.
    """
    grid = np.asarray(list(grid)) if grid is not None else _candidate_thresholds()
    best_cost, best_t, best_pm = np.inf, float("nan"), None
    for t in grid:
        pm = M.compute_point_metrics(y_true, proba, threshold=float(t))
        cost = cost_fp * pm.fp + cost_fn * pm.fn
        if cost < best_cost:
            best_cost, best_t, best_pm = cost, float(t), pm
    assert best_pm is not None
    logger.info("cost_sensitive(fp=%.1f, fn=%.1f) → threshold=%.3f (cost=%.1f)",
                cost_fp, cost_fn, best_t, best_cost)
    return ThresholdChoice(
        threshold=best_t,
        rule="cost_sensitive",
        rule_detail=f"argmin {cost_fp}*FP + {cost_fn}*FN",
        train_precision=best_pm.precision,
        train_recall=best_pm.recall,
        train_f1=best_pm.f1,
        train_n_trades=best_pm.n_trades,
    )


def naive_half() -> ThresholdChoice:
    """The textbook 0.5 baseline — included so every report has a reference."""
    return ThresholdChoice(
        threshold=0.5,
        rule="naive_0.5",
        rule_detail="fixed at 0.5 (no selection)",
        train_precision=float("nan"),
        train_recall=float("nan"),
        train_f1=float("nan"),
        train_n_trades=-1,
    )


def compare_choices(
    y_train: np.ndarray,
    proba_train: np.ndarray,
    y_test: np.ndarray,
    proba_test: np.ndarray,
    fbeta_value: float = 0.5,
    target_precision: float = 0.60,
    cost_fp: float = 2.0,
    cost_fn: float = 1.0,
) -> pd.DataFrame:
    """Run all four selectors on train data, score each on test, side-by-side.

    The notebook uses this to defend whichever rule it ultimately picks.
    """
    choices = [
        select_fbeta(y_train, proba_train, beta=fbeta_value),
        select_target_precision(y_train, proba_train, target=target_precision),
        select_cost_sensitive(y_train, proba_train, cost_fp=cost_fp, cost_fn=cost_fn),
        naive_half(),
    ]
    rows = []
    for c in choices:
        test_pm = M.compute_point_metrics(y_test, proba_test, threshold=c.threshold)
        rows.append({
            "rule": c.rule,
            "rule_detail": c.rule_detail,
            "threshold": c.threshold,
            "train_precision": c.train_precision,
            "train_recall": c.train_recall,
            "train_f1": c.train_f1,
            "train_n_trades": c.train_n_trades,
            "test_precision": test_pm.precision,
            "test_recall": test_pm.recall,
            "test_f1": test_pm.f1,
            "test_n_trades": test_pm.n_trades,
            "test_pct_taken": test_pm.pct_taken,
        })
    return pd.DataFrame(rows)


def is_inverted_auc(auc: float, margin: float = 0.05) -> bool:
    """Flag when a probabilistic ranking is class-inverted.

    AUC below 0.5 means the model's probabilities are anti-correlated with the
    realized labels — high-proba rows are *less* likely to be positive than
    low-proba rows. The standalone Part 5 audit's "within margin of 0.5" check
    catches near-coin-flip rankings but is symmetric: it doesn't distinguish
    "uninformative" from "actively backwards", which is the case that quietly
    survives orientation gates run on a full test set that mixes a correctly
    oriented sub-period with an inverted one. Returns True for AUCs strictly
    below ``0.5 - margin``.
    """
    return float(auc) < (0.5 - float(margin))
