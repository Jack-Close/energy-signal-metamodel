"""Wire src.part5.threshold.is_inverted_auc into each notebook's Section 11.1 audit.

Adds two new flag lines (one for selection AUC, one for OOS AUC) immediately
after the existing coin-flip checks. Idempotent: if T.is_inverted_auc is
already present in the audit cell, leave it alone.
"""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOKS = [
    "notebooks/2a_rbob_gasoline_model_training.ipynb",
    "notebooks/2b_heating_oil_model_training.ipynb",
    "notebooks/2c_natural_gas_model_training.ipynb",
    "notebooks/2d_wti_crude_oil_model_training.ipynb",
]


OLD_SEL = (
    "    sel_auc = roc_auc_score(sel_y, sel_proba)\n"
    "    if abs(sel_auc - 0.5) < 0.02:\n"
    "        flags.append(f'selection AUC = {sel_auc:.3f} (within 0.02 of coin flip)')\n"
)
NEW_SEL = (
    "    sel_auc = roc_auc_score(sel_y, sel_proba)\n"
    "    if abs(sel_auc - 0.5) < 0.02:\n"
    "        flags.append(f'selection AUC = {sel_auc:.3f} (within 0.02 of coin flip)')\n"
    "    if T.is_inverted_auc(sel_auc):\n"
    "        flags.append(f'selection AUC = {sel_auc:.3f} < 0.45 (class-inverted on selection slice)')\n"
)

OLD_OOS = (
    "    oos_auc = roc_auc_score(oos_y, oos_proba)\n"
    "    if abs(oos_auc - 0.5) < 0.02:\n"
    "        flags.append(f'OOS AUC = {oos_auc:.3f} (within 0.02 of coin flip)')\n"
)
NEW_OOS = (
    "    oos_auc = roc_auc_score(oos_y, oos_proba)\n"
    "    if abs(oos_auc - 0.5) < 0.02:\n"
    "        flags.append(f'OOS AUC = {oos_auc:.3f} (within 0.02 of coin flip)')\n"
    "    if T.is_inverted_auc(oos_auc):\n"
    "        flags.append(f'OOS AUC = {oos_auc:.3f} < 0.45 (class-inverted on OOS - proba anti-correlated with y)')\n"
)


def process(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if "T.is_inverted_auc" in src:
            return f"  {path.name}: already wired, skipping"
        if OLD_SEL not in src or OLD_OOS not in src:
            continue
        src2 = src.replace(OLD_SEL, NEW_SEL).replace(OLD_OOS, NEW_OOS)
        c["source"] = src2.splitlines(keepends=True)
        changed = True

    if not changed:
        return f"  {path.name}: no audit cell matched"

    with path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return f"  {path.name}: inversion checks wired"


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    for nb in NOTEBOOKS:
        path = repo / nb
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        print(process(path))


if __name__ == "__main__":
    main()
