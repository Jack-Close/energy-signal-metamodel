"""Part 5 — Model evaluation utilities.

Threshold / false-positive / precision-over-recall workstream.

All public APIs are deterministic (random_state=42 where stochastic),
type-hinted, and use logging rather than print.

Module layout
-------------
- metrics        : precision/recall/F1/AUC/AP/confusion matrix helpers
- threshold      : F-beta, target-precision, cost-sensitive selectors
                   (operate on train/OOF predictions; never on test)
- baseline       : blind-primary-signal baseline
- vol_overlay    : volatility-regime cutoff sweep
- export         : tidy CSV + Excel writers for results bundles
- predict_cache  : runs teammate model notebooks headlessly and caches predictions
"""

__all__ = [
    "metrics",
    "threshold",
    "baseline",
    "vol_overlay",
    "export",
    "predict_cache",
]
