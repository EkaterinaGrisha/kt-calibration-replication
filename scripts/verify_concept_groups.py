"""Integrity gate for the concept-level student-id field in prediction NPZs.

Why this exists
---------------
`concept_groups` is what makes the cluster bootstrap possible at
concept-of-knowledge granularity: it says which student each unrolled row
belongs to. The loaders in `cluster_bootstrap_calibration_*` fall back to
`groups` when it is absent and, when that does not match in length, return
`None` — the cell then disappears from the output file without an error. That
silent drop already produced two falsely green summaries in this project, so
the field is verified explicitly rather than assumed.

Four checks per (dataset, model, fold):
  1. `concept_groups` exists and has exactly the length of `concept_y_true`;
  2. every student occupies one contiguous block (pyKT writes sequences
     student by student; interleaving would mean the row order is not what the
     backfill assumed);
  3. the order in which distinct students first appear is identical to the
     order in the question-level `groups` — an independent check of the row
     ordering that the length check alone cannot give;
  4. on datasets with one concept per question the two vectors must be equal
     element by element, since the two granularities coincide there.

Exit code 0 iff every cell passes. Usage:

    python -m scripts.verify_concept_groups
"""
from __future__ import annotations

import sys

import numpy as np

from ktx import paths

DATASETS = ["algebra2005", "assist2009", "assist2012", "assist2015",
            "assist2017", "bridge2algebra2006", "ednet"]
MODELS = ["dkt", "sakt", "akt", "simplekt"]
FOLDS = [0, 1, 2, 3, 4]


def _first_appearance_order(a: np.ndarray) -> np.ndarray:
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]


def _n_blocks(a: np.ndarray) -> int:
    if a.size == 0:
        return 0
    return int(np.count_nonzero(np.diff(a)) + 1)


def check_cell(ds: str, model: str, fold: int) -> tuple[bool, str]:
    p = paths.ARTIFACTS_DIR / "predictions" / ds / f"{model}_fold{fold}.npz"
    if not p.exists():
        return False, "NPZ missing"
    d = np.load(p)
    if "concept_groups" not in d.files:
        return False, "concept_groups absent"
    cg = d["concept_groups"]
    cy = d["concept_y_true"]
    if cg.size != cy.size:
        return False, f"length {cg.size} != concept_y_true {cy.size}"
    if _n_blocks(cg) != np.unique(cg).size:
        return False, (f"students not contiguous: {_n_blocks(cg)} blocks for "
                       f"{np.unique(cg).size} students")
    if "groups" in d.files:
        g = d["groups"]
        if not np.array_equal(_first_appearance_order(cg), _first_appearance_order(g)):
            return False, "distinct-student order differs from question level"
        if cy.size == d["y_true"].size and not np.array_equal(cg, g):
            return False, "single-concept dataset but concept_groups != groups"
    return True, (f"n={cg.size} students={np.unique(cg).size} "
                  f"ratio={cg.size / max(1, d['y_true'].size):.4f}")


def main() -> int:
    bad = []
    for ds in DATASETS:
        for m in MODELS:
            for f in FOLDS:
                ok, msg = check_cell(ds, m, f)
                if not ok:
                    bad.append((ds, m, f, msg))
                    print(f"[FAIL {ds:20s} {m:9s} f{f}] {msg}")
    n = len(DATASETS) * len(MODELS) * len(FOLDS)
    print(f"\n{n - len(bad)}/{n} checks pass, {len(bad)} fail.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
