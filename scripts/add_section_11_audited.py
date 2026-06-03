"""Append the audited Part-5 evaluation block to Section 11 of notebooks 2a-2d.

Operates idempotently — if the marker comment is already present, do nothing.
Edits ONLY Section 11. Also fixes the 2c X_test_wti copy-paste bug as a side
effect (necessary because Section 11 imports the resulting model_probas).
"""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOKS = {
    "rb1s": {
        "path": "notebooks/2a_rbob_gasoline_model_training.ipynb",
        "winner": "LSTM",
        "instrument_label": "RBOB Gasoline (rb1s)",
        "verdict_md": (
            "The LSTM was selected by CV. Its full-test AUC sits just above coin flip "
            "(~0.55), so we should not expect a transformative filter on gasoline. The "
            "carve above isolates a selection slice (Aug-Dec 2021) where the F0.5 "
            "threshold is chosen, then freezes that threshold for the Jan-Jun 2022 "
            "out-of-sample window. Read the precision/recall numbers above against the "
            "blind baseline that takes every signalled trade. If the metamodel lifts "
            "precision a few points while cutting trades, it is doing useful work even "
            "with a weak AUC. If precision matches blind, the metamodel is not earning "
            "its complexity for rb1s and we should say so in writing."
        ),
    },
    "ho1s": {
        "path": "notebooks/2b_heating_oil_model_training.ipynb",
        "winner": "MLP",
        "instrument_label": "Heating Oil (ho1s)",
        "verdict_md": (
            "The MLP was selected by CV. **The structural caveat for ho1s is sample size**: "
            "the full test set carries on the order of 26 signalled trades, so the "
            "Jan-Jun 2022 out-of-sample slice contains only a handful. The printed "
            "metrics above are reported for completeness, but with that few trades "
            "precision/recall are statistically meaningless and we should not present "
            "them as a verdict on the model. The honest reading is that the heating-oil "
            "metamodel task is **unevaluable at this resolution**; the upstream cause is "
            "the triple-barrier labeller plus the primary signal's firing pattern on "
            "ho1s, not the MLP. If the degeneracy audit raises a flag here, treat it as "
            "informational rather than as a failure of the model."
        ),
    },
    "ng1s": {
        "path": "notebooks/2c_natural_gas_model_training.ipynb",
        "winner": "LSTM",
        "instrument_label": "Natural Gas (ng1s)",
        "verdict_md": (
            "The LSTM was selected by CV with the strongest full-test AUC among the "
            "four instruments (~0.66). Because the blind-baseline precision on "
            "natural-gas signals is historically low, any precision lift the metamodel "
            "gives us in the printed OOS numbers above is meaningful, not cosmetic. "
            "Read the degeneracy audit: a tight `proba_std`, an OOS AUC stuck near 0.5, "
            "or `pct_taken` near 100% would indicate the LSTM is collapsing to a single "
            "class rather than discriminating — that turns the apparent precision into "
            "an artefact of the base rate."
        ),
    },
    "cl1s": {
        "path": "notebooks/2d_wti_crude_oil_model_training.ipynb",
        "winner": "MLP",
        "instrument_label": "WTI Crude Oil (cl1s)",
        "verdict_md": (
            "The MLP was selected by CV. Full-test AUC is around 0.67 and the blind "
            "baseline already takes most signalled trades at ~0.83 precision, so the "
            "metamodel's room to improve is bounded by definition: the trades the "
            "primary signal flags on cl1s are already mostly profitable. The "
            "methodologically interesting question is whether the filter can shave off "
            "the worst tail of those trades at the cost of recall. The printed "
            "precision/recall pair against the baseline answers that directly. A small "
            "absolute precision lift is the expected upper bound, not a disappointment."
        ),
    },
}


CROSS_CUTTING_MD = (
    "**Upstream caveats that apply across instruments** (not artefacts of this "
    "metamodel — flagging for honesty, not to fix here):\n"
    "\n"
    "- The triple-barrier labeller in Section 8 uses `max_hold=20`, so the last "
    "~20 trading days of any window can carry no label. On the hidden H2 2022 "
    "evaluation window this blanks late December.\n"
    "- The threshold selected above is frozen from the Aug-Dec 2021 selection "
    "slice. It is **never** re-tuned on the OOS window, and the OOS metrics "
    "above are reported at exactly that frozen value — the standard meta-label "
    "leakage discipline.\n"
    "- These results are for the chosen winning model only; the threshold sweep "
    "across all six models earlier in this section remains the comparator view."
)


MARKER = "# === Part-5 audited evaluation block (do not duplicate) ==="


SETUP_AND_SELECTION_CODE = '''\
# === Part-5 audited evaluation block (do not duplicate) ===
# Carves the test set into a selection slice (Aug-Dec 2021) used to choose the
# decision threshold, and an out-of-sample slice (Jan-Jun 2022) where that
# threshold is frozen and applied. Choosing the threshold on OOS would inflate
# precision; the slice/freeze pattern enforces meta-label leakage discipline.

from pathlib import Path as _Path
import sys as _sys

_repo = _Path('.').resolve()
while _repo != _repo.parent and not (_repo / 'src' / 'part5').exists():
    _repo = _repo.parent
if (_repo / 'src' / 'part5').exists() and str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))

from src.part5 import metrics as M
from src.part5 import threshold as T
from src.part5 import baseline as B

WINNING_MODEL = '__WINNER__'
SELECTION_END = pd.Timestamp('2021-12-31')
OOS_START     = pd.Timestamp('2022-01-01')

winner_proba = np.asarray(model_probas[WINNING_MODEL])
winner_y     = np.asarray(model_labels[WINNING_MODEL]).astype(int)

# LSTM has fewer prediction rows than y_test because of the sequence window;
# re-running build_sequences gives the matching dates aligned 1:1 with y_seq_te.
if WINNING_MODEL == 'LSTM':
    _, _, _winner_dates = build_sequences(full_X_sc, y_test, seq_len)
    winner_dates = pd.DatetimeIndex(_winner_dates)
else:
    winner_dates = pd.DatetimeIndex(y_test.index)

assert len(winner_dates) == len(winner_proba) == len(winner_y), \\
    f'date/proba/y misalignment: {len(winner_dates)}/{len(winner_proba)}/{len(winner_y)}'

sel_mask = winner_dates <= SELECTION_END
oos_mask = winner_dates >= OOS_START
sel_proba, sel_y = winner_proba[sel_mask], winner_y[sel_mask]
oos_proba, oos_y = winner_proba[oos_mask], winner_y[oos_mask]

print(f'winning model: {WINNING_MODEL}')
print(f'  selection (<= {SELECTION_END.date()}):  '
      f'n={len(sel_y)}, positives={int(sel_y.sum())}, '
      f'base_rate={sel_y.mean() if len(sel_y) else float("nan"):.3f}')
print(f'  out-of-sample (>= {OOS_START.date()}): '
      f'n={len(oos_y)}, positives={int(oos_y.sum())}, '
      f'base_rate={oos_y.mean() if len(oos_y) else float("nan"):.3f}')

# F-beta with beta < 1 weights precision over recall, matching the meta-label
# cost asymmetry: a bad trade costs money, a skipped trade costs only
# opportunity. Threshold picked on the selection slice and frozen below.
if len(sel_y) >= 5 and len(np.unique(sel_y)) >= 2:
    choice = T.select_fbeta(sel_y, sel_proba, beta=0.5)
    frozen_threshold = choice.threshold
    print(f'\\nF0.5 on selection slice -> threshold = {frozen_threshold:.3f}')
    print(f'  selection precision={choice.train_precision:.3f}  '
          f'recall={choice.train_recall:.3f}  F1={choice.train_f1:.3f}  '
          f'n_trades_selected={choice.train_n_trades}')
else:
    choice = None
    frozen_threshold = 0.5
    print(f'\\nselection slice too small or single-class (n={len(sel_y)}, '
          f'unique_y={len(np.unique(sel_y))}) -> threshold defaults to {frozen_threshold}')
'''


OOS_AND_AUDIT_CODE = '''\
# OOS metrics at the frozen threshold + blind-primary baseline + degeneracy audit.
# The blind baseline takes every signalled trade in OOS; the filter should buy
# precision at the cost of recall. Audit flags mirror the standalone Part-5
# evaluation: collapsed proba spread, AUC near coin flip, or pct_taken at the
# ceiling indicate the model is not really discriminating.

if len(oos_y) >= 5 and len(np.unique(oos_y)) >= 2:
    oos_pm = M.compute_point_metrics(oos_y, oos_proba, threshold=frozen_threshold)
    print(f'OOS @ frozen threshold {frozen_threshold:.3f}:')
    print(f'  n_trades      = {oos_pm.n_trades} '
          f'({oos_pm.pct_taken*100:.1f}% of OOS signals)')
    print(f'  precision     = {oos_pm.precision:.3f}')
    print(f'  recall        = {oos_pm.recall:.3f}')
    print(f'  F1            = {oos_pm.f1:.3f}')
    print(f'  AUC           = {oos_pm.auc:.3f}')
    print(f'  avg precision = {oos_pm.average_precision:.3f}')
    print(f'  confusion [[TN={oos_pm.tn}, FP={oos_pm.fp}], '
          f'[FN={oos_pm.fn}, TP={oos_pm.tp}]]')

    blind_pm = B.blind_primary_baseline(pd.DataFrame({'meta_label': oos_y}))
    cmp_df = B.baseline_vs_filter_table(blind_pm, oos_pm,
                                        label=f'{WINNING_MODEL}-filtered')
    print('\\nFilter vs blind baseline (OOS):')
    print(cmp_df.round(4).to_string(index=False))
else:
    oos_pm = None
    print(f'OOS too small to evaluate (n={len(oos_y)}, '
          f'unique_y={len(np.unique(oos_y))}) -> reported as unevaluable, '
          'not as a failed filter.')

flags = []
if len(sel_proba) and np.std(sel_proba) < 0.01:
    flags.append(f'selection_proba_std={np.std(sel_proba):.4f} < 0.01 (collapsed spread)')
if len(oos_proba) and np.std(oos_proba) < 0.01:
    flags.append(f'oos_proba_std={np.std(oos_proba):.4f} < 0.01 (collapsed spread)')
if len(sel_y) >= 5 and len(np.unique(sel_y)) >= 2:
    sel_auc = roc_auc_score(sel_y, sel_proba)
    if abs(sel_auc - 0.5) < 0.02:
        flags.append(f'selection AUC = {sel_auc:.3f} (within 0.02 of coin flip)')
if len(oos_y) >= 5 and len(np.unique(oos_y)) >= 2:
    oos_auc = roc_auc_score(oos_y, oos_proba)
    if abs(oos_auc - 0.5) < 0.02:
        flags.append(f'OOS AUC = {oos_auc:.3f} (within 0.02 of coin flip)')
if oos_pm is not None and oos_pm.pct_taken > 0.98:
    flags.append(f'OOS pct_taken = {oos_pm.pct_taken*100:.1f}% > 98 '
                 '(filter takes essentially every trade)')

if flags:
    print('\\nDEGENERACY AUDIT - flags raised:')
    for f in flags:
        print(f'  ! {f}')
else:
    print('\\nDegeneracy audit: no flags.')
'''


def make_md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def make_code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def find_insertion_index(cells: list[dict]) -> int:
    """Return the index just before Section 12 markdown header (which closes Section 11)."""
    for i, c in enumerate(cells):
        if c["cell_type"] != "markdown":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for line in src.splitlines():
            s = line.strip()
            if s.startswith("## 12") and "Feature Importance" in s:
                return i
    raise RuntimeError("could not locate '## 12. Feature Importance Analysis' header")


def fix_2c_xtest_wti(cells: list[dict]) -> bool:
    """Replace `X_test_wti` with `X_test_natural_gas` in the broken Section 11 cell."""
    changed = False
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if "X_test_wti" in src and "model_probas" in src:
            new = src.replace("X_test_wti", "X_test_natural_gas")
            c["source"] = new.splitlines(keepends=True)
            changed = True
    return changed


def block_already_present(cells: list[dict]) -> bool:
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if MARKER in src:
            return True
    return False


def build_block(cfg: dict) -> list[dict]:
    intro_md = (
        f"### 11.1 Frozen-threshold evaluation of the winning model ({cfg['winner']})\n"
        f"\n"
        f"The sweep above compares all six models on the full test set. The block below "
        f"narrows the analysis to the winning {cfg['winner']} for {cfg['instrument_label']} "
        f"under the Part-5 methodology: a held-out selection slice (Aug–Dec 2021) picks "
        f"the decision threshold via F-beta with beta=0.5 (precision-weighted), and that "
        f"threshold is then frozen for the deliverable out-of-sample window (Jan–Jun 2022). "
        f"Choosing the threshold on the OOS window would inflate apparent precision, "
        f"which is a standard meta-labelling leakage error.\n"
    )
    setup_code = SETUP_AND_SELECTION_CODE.replace("__WINNER__", cfg["winner"])
    verdict_md = (
        f"#### Verdict for {cfg['instrument_label']}\n"
        f"\n"
        f"{cfg['verdict_md']}\n"
        f"\n"
        f"{CROSS_CUTTING_MD}\n"
    )
    return [
        make_md_cell(intro_md),
        make_code_cell(setup_code),
        make_code_cell(OOS_AND_AUDIT_CODE),
        make_md_cell(verdict_md),
    ]


def process(path: Path, cfg: dict) -> str:
    with path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    fixed_2c = False
    if cfg["winner"] == "LSTM" and "natural_gas" in str(path):
        fixed_2c = fix_2c_xtest_wti(nb["cells"])

    if block_already_present(nb["cells"]):
        msg = f"  already has marker; skipping append for {path.name}"
        if fixed_2c:
            msg += " (2c rename applied)"
            with path.open("w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
                f.write("\n")
        return msg

    insert_at = find_insertion_index(nb["cells"])
    block = build_block(cfg)
    nb["cells"] = nb["cells"][:insert_at] + block + nb["cells"][insert_at:]

    with path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    tag = f"appended {len(block)} cells at index {insert_at}"
    if fixed_2c:
        tag += "; fixed X_test_wti -> X_test_natural_gas"
    return f"  {path.name}: {tag}"


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    for key, cfg in NOTEBOOKS.items():
        path = repo / cfg["path"]
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        print(process(path, cfg))


if __name__ == "__main__":
    main()
