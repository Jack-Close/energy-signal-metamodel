"""
WHY THIS EXISTS
---------------
The four per-instrument model notebooks (WTI / Heating-Oil / Natural-gas /
Gasoline) produce probabilities in an in-notebook dict `model_probas` and a
ground-truth series `y_te` (or `y_seq_te` for the LSTM path). They are never
persisted to disk. Part 5 needs those numbers without retraining anything.

WHAT THIS DOES
--------------
For each instrument:
  1. Execute the model notebook via nbclient (kernel: python3).
  2. After execution, append a probe cell that pickles the relevant variables.
  3. Load the pickle and re-emit as a tidy parquet at
       results/part5/cache/<inst>.parquet
     keyed on (date, instrument, model, proba, y_true).

Idempotency: if the cache file exists, the notebook execution is skipped.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Iterable

import nbformat
import pandas as pd
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "results" / "part5" / "cache"

# Per-instrument: (notebook filename, instrument ticker used in the parquet)
INSTRUMENT_NOTEBOOKS: dict[str, str] = {
    "cl1s": "WTI-model.ipynb",
    "ho1s": "Heating-Oil-model.ipynb",
    "ng1s": "Natural-gas-model.ipynb",
    "rb1s": "Gasoline-Full-model.ipynb",
}

# Variable names produced by each model notebook (consistent across all four)
PROBE_VARS = (
    "model_probas",      # dict[str, np.ndarray]
    "model_labels",      # dict[str, np.ndarray]  (y_te or y_seq_te depending on model)
    "X_test_wti",        # for WTI; other notebooks use X_test_<inst>. Probe handles via locals().
    "y_te",
)


def _probe_cell_source(pickle_path: Path) -> str:
    """Source for a probe cell that grabs the relevant locals and pickles them.

    We use locals().get(...) defensively because each notebook may name its
    test-feature DataFrame differently (X_test_wti, X_test_ho, ...).
    """
    return f"""
# === Part 5 probe cell — appended by predict_cache.py ===
import pickle as _pkl
_locals = dict(locals())

def _grab(*names):
    for n in names:
        if n in _locals:
            return _locals[n]
    return None

_payload = {{
    'model_probas': _grab('model_probas'),
    'model_labels': _grab('model_labels'),
    'y_te':         _grab('y_te'),
    'y_seq_te':     _grab('y_seq_te'),
    'X_test':       _grab('X_test_wti', 'X_test_ho', 'X_test_ng', 'X_test_rb',
                          'X_test_heating_oil', 'X_test_natural_gas', 'X_test_gasoline'),
}}

with open(r'{pickle_path.as_posix()}', 'wb') as _f:
    _pkl.dump(_payload, _f)

print('probe wrote', _payload.keys())
"""


def _execute_notebook(nb_path: Path, pickle_path: Path, timeout: int = 3600) -> None:
    """Run nb_path with an appended probe cell. Writes pickle_path on success."""
    nb = nbformat.read(nb_path, as_version=4)
    nb.cells.append(nbformat.v4.new_code_cell(_probe_cell_source(pickle_path)))

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},  # so relative paths in nb work
    )
    try:
        client.execute()
    except CellExecutionError:
        logger.exception("notebook %s failed during execution", nb_path.name)
        raise


def _load_probe(pickle_path: Path) -> dict:
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def _to_tidy(payload: dict, instrument: str) -> pd.DataFrame:
    """Convert the {'model_probas', 'model_labels', ...} payload to long format."""
    rows = []
    probas = payload.get("model_probas") or {}
    labels = payload.get("model_labels") or {}
    fallback_y = payload.get("y_te")
    fallback_X = payload.get("X_test")

    for model_name, proba in probas.items():
        y = labels.get(model_name, fallback_y)
        if y is None or proba is None:
            logger.warning("[%s] missing proba or labels for %s; skipping", instrument, model_name)
            continue

        # Try to recover a date index. For point-in-time models the LSTM aside,
        # X_test rows align with proba rows. LSTM uses a sequence window so its
        # length will be shorter; align from the tail.
        if fallback_X is not None and len(fallback_X) == len(proba):
            dates = pd.Index(fallback_X.index)
        elif fallback_X is not None and len(fallback_X) >= len(proba):
            dates = pd.Index(fallback_X.index[-len(proba):])
        else:
            dates = pd.RangeIndex(len(proba))
            logger.warning("[%s] no usable date index for %s; using positional", instrument, model_name)

        df = pd.DataFrame({
            "date": dates,
            "instrument": instrument,
            "model": model_name,
            "proba": pd.Series(proba).to_numpy(dtype=float),
            "y_true": pd.Series(y).to_numpy(dtype=int),
        })
        rows.append(df)

    if not rows:
        raise RuntimeError(f"no model predictions extracted for {instrument}")
    return pd.concat(rows, ignore_index=True)


def cache_predictions(
    instruments: Iterable[str] | None = None,
    force: bool = False,
    timeout: int = 3600,
) -> dict[str, Path]:
    """Run notebooks (if needed) and write per-instrument tidy parquets.

    Returns a mapping instrument -> parquet path.
    """
    instruments = list(instruments) if instruments else list(INSTRUMENT_NOTEBOOKS.keys())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    for inst in instruments:
        nb_name = INSTRUMENT_NOTEBOOKS[inst]
        nb_path = REPO_ROOT / "notebooks" / nb_name
        parquet_path = CACHE_DIR / f"{inst}.parquet"
        pickle_path = CACHE_DIR / f"{inst}.pkl"

        if parquet_path.exists() and not force:
            logger.info("[%s] cache hit at %s", inst, parquet_path)
            out[inst] = parquet_path
            continue

        logger.info("[%s] executing %s (this is slow — XGB grid + neural nets)", inst, nb_name)
        _execute_notebook(nb_path, pickle_path, timeout=timeout)

        payload = _load_probe(pickle_path)
        tidy = _to_tidy(payload, instrument=inst)
        tidy.to_parquet(parquet_path, index=False)
        logger.info("[%s] wrote %d rows × %d models → %s",
                    inst, len(tidy), tidy["model"].nunique(), parquet_path)
        out[inst] = parquet_path

    return out


def load_cached_predictions(instruments: Iterable[str] | None = None) -> pd.DataFrame:
    """Concatenate cached per-instrument tidy parquets into one long DataFrame."""
    instruments = list(instruments) if instruments else list(INSTRUMENT_NOTEBOOKS.keys())
    parts = []
    for inst in instruments:
        p = CACHE_DIR / f"{inst}.parquet"
        if not p.exists():
            logger.warning("[%s] no cache at %s — run cache_predictions() first", inst, p)
            continue
        parts.append(pd.read_parquet(p))
    if not parts:
        raise FileNotFoundError(f"no cached predictions found in {CACHE_DIR}")
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cache_predictions()
