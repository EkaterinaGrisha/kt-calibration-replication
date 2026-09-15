"""What the fit side costs: two paths to the same question-level probability.

A post-hoc correction is learned on one set of probabilities and applied to
another. Both are called "the probability that the student answers this
question correctly", and the two are produced by different code: the library's
own question-level inference, and the project's averaging of the model's
concept-level output over the concepts of the question. The rows even come out
in slightly different numbers.

This script measures what happens when the correction is learned on one and
applied to the other, against learning and applying on the same one. The test
side is identical in both arms — the library's question-level output — so the
comparison is paired row by row and isolates the fit side alone.

  arm v4 : fit on `valid_y_*_q_pykt`  (library, both sides in agreement)
  arm v2 : fit on `valid_y_*_q`       (project averaging; the mismatch)

Output `c4_path_contrasts.csv`: per (dataset, model, fold, method) the two
calibration errors, their difference with a cluster-bootstrap interval over
students, and Holm inside (method, fold).
"""
from __future__ import annotations

import argparse
import csv
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"sklearn\.linear_model\._linear_loss")

from ktx import paths
from ktx.calibration import CALIBRATORS
from ktx.stats import holm_bonferroni
from ktx.stats_fast import paired_cluster_bootstrap_ece
from scripts.c4_significance import DATASETS, DEEP, FOLDS, _ece

OUT = paths.ARTIFACTS_DIR / "cross_dataset"


def load(dataset: str, model: str, fold: int):
    p = paths.ARTIFACTS_DIR / "predictions" / dataset / f"{model}_fold{fold}.npz"
    if not p.exists():
        return None, "no npz"
    d = np.load(p)
    need = {"y_true", "y_prob", "groups",
            "valid_y_true_q_pykt", "valid_y_prob_q_pykt",
            "valid_y_true_q", "valid_y_prob_q"}
    missing = sorted(need - set(d.files))
    if missing:
        return None, f"missing {missing}"
    return (d["y_true"].astype(int), d["y_prob"].astype(float),
            np.asarray(d["groups"]),
            d["valid_y_true_q_pykt"].astype(int), d["valid_y_prob_q_pykt"].astype(float),
            d["valid_y_true_q"].astype(int), d["valid_y_prob_q"].astype(float)), None


def main() -> None:
    ap = argparse.ArgumentParser()
    # Десять тысяч, а не две: при двух тысячах наименьшее достижимое
    # p-значение равно 2/2001, и после поправки Холма на семью из
    # пятидесяти двух сравнений оно становится 0.052 — вся семья
    # оказывается непроходимой независимо от величины различий.
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    args = ap.parse_args()

    rows, skipped = [], []
    for dataset in DATASETS:
        for model in DEEP:
            for fold in args.folds:
                loaded, why = load(dataset, model, fold)
                if loaded is None:
                    skipped.append((dataset, model, fold, why))
                    continue
                ty, tp, groups, vy4, vp4, vy2, vp2 = loaded
                for name, Cal in CALIBRATORS.items():
                    p4 = Cal().fit(vp4, vy4).transform(tp)
                    p2 = Cal().fit(vp2, vy2).transform(tp)
                    r = paired_cluster_bootstrap_ece(ty, p2, p4, groups,
                                                     n_boot=args.n_boot, seed=fold)
                    rows.append({
                        "dataset": dataset, "model": model, "fold": fold,
                        "method": name,
                        "ece_none": _ece(ty, tp, groups),
                        "ece_mismatched_fit": r.ece_a,
                        "ece_coherent_fit": r.ece_b,
                        "delta_ece": r.diff,
                        "ratio": (r.ece_a / r.ece_b) if r.ece_b > 0 else float("nan"),
                        "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p_value,
                        "n_valid_coherent": int(vy4.size),
                        "n_valid_mismatched": int(vy2.size),
                        "n_test": int(ty.size), "n_students": r.n_students,
                        "family_key": f"{name}|f{fold}",
                    })
                    print(f"[{dataset:20s} {model:9s} f{fold} {name:11s}] "
                          f"mismatched={r.ece_a:.5f} coherent={r.ece_b:.5f} "
                          f"ratio={rows[-1]['ratio']:.2f} p={r.p_value:.4f}")

    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["family_key"]].append(i)
    for key, idxs in by.items():
        res = holm_bonferroni([rows[i]["p"] for i in idxs])
        for j, i in enumerate(idxs):
            rows[i]["p_holm"] = res["adjusted"][j]
            rows[i]["reject_holm"] = bool(res["reject"][j])
            rows[i]["family_size"] = len(idxs)

    if rows:
        with open(OUT / "c4_path_contrasts.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {OUT / 'c4_path_contrasts.csv'}")
    if skipped:
        agg = defaultdict(int)
        for ds, m, f, why in skipped:
            agg[(ds, why)] += 1
        print("\nskipped:")
        for (ds, why), n in sorted(agg.items()):
            print(f"  {ds:20s} x{n:2d} {why}")


if __name__ == "__main__":
    main()
