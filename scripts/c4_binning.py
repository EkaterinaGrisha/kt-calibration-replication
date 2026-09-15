"""Does the verdict depend on how the probability axis is cut into bins?

The calibration error is an average over bins, and there are two usual ways to
draw them: equal width on the probability axis, and equal mass, so that every
bin holds the same number of predictions. Predictions in knowledge tracing
cluster in a narrow band, which is exactly the case where the two disagree, so
the paper reports whether its verdicts survive the change.

Writes `c4_binning.csv` with both errors per cell and the two verdicts:
which correction wins under each binning.
"""
from __future__ import annotations

import csv

import numpy as np

from ktx import paths
from ktx.metrics import ece_equal_mass
from scripts.c4_significance import (
    CLASSICAL,
    DATASETS,
    DEEP,
    FOLDS,
    METHODS,
    _ece,
    calibrated_probabilities,
    load_cell,
)

OUT = paths.ARTIFACTS_DIR / "cross_dataset"


def main() -> None:
    rows = []
    agree = disagree = 0
    for dataset in DATASETS:
        for granularity in ("concept", "question"):
            fams = (("deep", DEEP),) if granularity == "concept" else \
                   (("deep", DEEP), ("classical", CLASSICAL))
            for family, models in fams:
                for model in models:
                    for fold in FOLDS:
                        loaded, _ = load_cell(dataset, family, model, fold, granularity)
                        if loaded is None:
                            continue
                        ty, tp, vy, vp, groups = loaded
                        probs = calibrated_probabilities(tp, vy, vp)
                        ew = {m: _ece(ty, probs[m], groups) for m in METHODS}
                        em = {m: float(ece_equal_mass(ty, probs[m])) for m in METHODS}
                        bw = min(ew, key=ew.get)
                        bm = min(em, key=em.get)
                        agree += bw == bm
                        disagree += bw != bm
                        for m in METHODS:
                            rows.append({
                                "dataset": dataset, "granularity": granularity,
                                "family": family, "model": model, "fold": fold,
                                "method": m, "ece_equal_width": ew[m],
                                "ece_equal_mass": em[m],
                                "best_equal_width": bw, "best_equal_mass": bm,
                            })
    with open(OUT / "c4_binning.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT / 'c4_binning.csv'}")
    print(f"best correction agrees under both binnings in {agree} of {agree + disagree} cells")


if __name__ == "__main__":
    main()
