"""Backfill student-id (`groups`) into concept-level deep predictions NPZ files.

Why concept-level needs its own backfill. The Q-level backfill
(`dump_deep_groups_from_ckpts.py`) recovers `orirow → uid` via pyKT's
`evaluate_question(save_path=...)`, which writes per-prediction rows. That path
works for question-granularity datasets only. ASSISTments-2015 is concept-only
(no `test_question_sequences.csv`), so we cannot reuse that mapping.

Instead we recover groups **without rerunning the model**: pyKT writes
`test_sequences.csv` with a per-row `uid` and a per-position `selectmasks`
string. Each row's number of valid concept predictions is
``(sum(selectmasks==1) - 1)`` — the −1 because pyKT shifts each sequence by
one position (`rshft`, predictions are made from position 1 onwards). The total
sum over all rows equals the length of `concept_y_true` saved during training,
giving an exact alignment proof.

Output: existing NPZ files updated in-place with a `groups` field whose length
matches `concept_y_true`.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ktx import paths

CONCEPT_DATASETS = ["assist2015"]  # extend here if more concept-only datasets join
# multi-KC datasets: need `concept_groups` (concept-level uid) separate from
# `groups` (q-level uid from pyKT evaluate_question). concept_y_true is the
# canonical length target.
MULTI_KC_DATASETS = ["assist2009", "algebra2005", "bridge2algebra2006", "ednet"]


def _per_row_n_pred(selectmasks: pd.Series) -> np.ndarray:
    """Number of concept predictions contributed by each test-sequence row."""
    n = np.empty(len(selectmasks), dtype=np.int64)
    for i, sm in enumerate(selectmasks):
        # selectmask: comma-separated ints; 1 = valid position, -1 = padding,
        # 0 should not appear in current pyKT preprocess (verified empirically).
        bits = [int(x) for x in str(sm).split(",")]
        v = sum(1 for b in bits if b == 1)
        n[i] = max(0, v - 1)   # pyKT shifts by 1: predictions start at position 1
    return n


def _backfill_one(dataset: str, model: str, fold: int, force: bool,
                  field: str = "groups", length_key: str = "y_true") -> str:
    """Backfill a `field` array (student uid per prediction row) into a deep-model
    NPZ. `length_key` picks which array in the NPZ determines the target length:
    - `y_true` for concept-only datasets (assist2015, y_true == concept_y_true).
    - `concept_y_true` for multi-KC datasets, when the NPZ stores q-level
      `y_true` from pyKT evaluate_question and needs a separate concept-level
      groups vector under a different key (default `field=concept_groups`).
    """
    npz = paths.ARTIFACTS_DIR / "predictions" / dataset / f"{model}_fold{fold}.npz"
    if not npz.exists():
        return "NPZ missing"
    d = dict(np.load(npz))
    existing = d.get(field, np.array([]))
    if existing.size > 0 and not force:
        return f"skip ({field} already present, size={existing.size})"

    if length_key not in d:
        return f"NPZ missing {length_key}"
    target_n = int(d[length_key].size)

    test_seq_path = paths.PYKT_ROOT / "data" / dataset / "test_sequences.csv"
    if not test_seq_path.exists():
        return f"test_sequences.csv missing at {test_seq_path}"
    tseq = pd.read_csv(test_seq_path)
    if "uid" not in tseq.columns or "selectmasks" not in tseq.columns:
        return "test_sequences.csv missing uid/selectmasks"

    n_pred_per_row = _per_row_n_pred(tseq["selectmasks"])
    total = int(n_pred_per_row.sum())
    if total != target_n:
        return (f"FAIL: derived total {total} != {length_key} len {target_n}; the "
                f"shift assumption may be wrong for this preprocessing")

    groups = np.repeat(tseq["uid"].to_numpy(dtype=np.int64), n_pred_per_row)
    if groups.size != target_n:
        return f"FAIL: repeat size {groups.size} != target {target_n}"

    d[field] = groups
    # Atomic write: the prediction NPZ files are gitignored and cost GPU hours to
    # regenerate, so an interrupted `savez_compressed` must never leave a
    # truncated file behind. Write beside the target, then replace in one step.
    tmp = npz.with_suffix(".tmp.npz")  # must end in .npz: savez appends it otherwise
    np.savez_compressed(tmp, **d)
    os.replace(tmp, npz)
    return f"OK {field}: n_pred={target_n} n_students={int(np.unique(groups).size)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=CONCEPT_DATASETS)
    ap.add_argument("--models", nargs="+", default=["dkt", "sakt", "akt"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--field", default="groups",
                    help="NPZ field to write. `groups` for concept-only datasets "
                         "(default). `concept_groups` for multi-KC datasets that "
                         "already have q-level `groups` from pyKT.")
    ap.add_argument("--length-key", default="y_true",
                    help="NPZ array whose length determines target size. `y_true` "
                         "for concept-only (default). `concept_y_true` for "
                         "multi-KC + --field concept_groups.")
    args = ap.parse_args()

    for ds in args.datasets:
        for m in args.models:
            for f in args.folds:
                msg = _backfill_one(ds, m, f, args.force,
                                    field=args.field, length_key=args.length_key)
                print(f"[{ds:22s} {m:5s} f{f}] {msg}")


if __name__ == "__main__":
    main()
