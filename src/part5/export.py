"""Tidy CSV / Excel writers for the Part-5 results bundle.

Conventions
-----------
- Every DataFrame is in long/tidy form so it round-trips cleanly through
  Excel and is filterable per instrument or per strategy.
- The bundle is written to results/part5/. Files are versioned only if a
  collision is detected to avoid clobbering a teammate's outputs.
- Excel sheet names are kept <= 31 chars (Excel limit) and sanitised.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_sheet(name: str) -> str:
    """Sanitise to a valid Excel sheet name (<=31 chars, no /\\[]:*?)."""
    s = re.sub(r"[\\/\[\]:*?]", "_", name)
    return s[:31] if len(s) > 31 else s


def write_csv(df: pd.DataFrame, path: Path, overwrite: bool = False) -> Path:
    """Write a DataFrame to CSV.

    `overwrite=False` (default) protects against clobbering teammate outputs by
    suffixing _v2, _v3 etc. on collision. Pass `overwrite=True` for deliberate
    re-runs of files this module itself produced.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        df.to_csv(path, index=False)
        logger.info("wrote (overwrite) %s (%d rows x %d cols)", path, len(df), len(df.columns))
        return path
    final = path
    i = 1
    while final.exists():
        i += 1
        final = path.with_name(f"{path.stem}_v{i}{path.suffix}")
    df.to_csv(final, index=False)
    logger.info("wrote %s (%d rows x %d cols)", final, len(df), len(df.columns))
    return final


def write_excel_bundle(tables: dict[str, pd.DataFrame], path: Path,
                        overwrite: bool = False) -> Path:
    """Write multiple DataFrames into one .xlsx workbook (one sheet per key).

    `overwrite=False` (default) suffixes _v2 etc. to avoid clobbering. Pass
    `overwrite=True` for deliberate re-runs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        final = path
    else:
        final = path
        i = 1
        while final.exists():
            i += 1
            final = path.with_name(f"{path.stem}_v{i}{path.suffix}")

    with pd.ExcelWriter(final, engine="openpyxl") as writer:
        for sheet, df in tables.items():
            df.to_excel(writer, sheet_name=_safe_sheet(sheet), index=False)
    logger.info("wrote workbook %s with %d sheets", final, len(tables))
    return final
