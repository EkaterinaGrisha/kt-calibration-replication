"""Significance layer for the calibration paper.

What it produces
----------------
Three CSVs under `artifacts/cross_dataset/`:

* `c4_cell_ece.csv` — the raw grid: calibration error, area under the curve and
  test size for every (dataset, granularity, family, model, fold, method). Full
  precision, no rounding to four places, because the differences that decide
  which model is called best-calibrated are of order 1e-4.
* `c4_method_contrasts.csv` — per fold, two contrasts per cell:
  `iso_vs_none`, fixed before looking at the data, and `best_vs_runnerup`,
  which selects the winner on the same test set it is then tested on. Both are
  reported; only the first supports a claim.
* `c4_model_contrasts.csv` — per fold, the best-calibrated model against the
  runner-up: inside the deep family at both granularities (same rows, paired)
  and across families at question level (different rows, unpaired on the
  students the two sides share).

Why five folds
--------------
The earlier artifacts bootstrap fold 0 only. The test students are the same in
all five folds — the folds differ in what the model trained on — so a contrast
computed on one fold answers "is this difference larger than the noise of
resampling students", and a contrast repeated on five folds also answers "does
it survive retraining". The two questions have different answers often enough
that reporting only the first was producing a summary table and a bootstrap
table that disagreed about which model was best calibrated on four datasets of
seven.

Multiplicity
------------
Holm's step-down correction is applied inside a family, and a family is one
(granularity, fold, contrast kind) block: 28 deep cells at concept level, 52
cells at question level, seven datasets for model contrasts. The family is
written into every row so the correction can be recomputed from the file.

Run:
    python -m scripts.c4_significance                 # everything
    python -m scripts.c4_significance --n-boot 200    # quick pass
"""
from __future__ import annotations

import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"sklearn\.linear_model\._linear_loss")

from ktx import paths
from ktx.calibration import CALIBRATORS
from ktx.stats import holm_bonferroni
from ktx.stats_fast import (
    ece_from_totals,
    paired_cluster_bootstrap_ece,
    student_bin_stats,
    unpaired_cluster_bootstrap_ece,
)

DATASETS = ["algebra2005", "assist2009", "assist2012", "assist2015",
            "assist2017", "bridge2algebra2006", "ednet"]
DEEP = ["dkt", "sakt", "akt", "simplekt"]
CLASSICAL = ["bkt", "pfa", "pfa_recency", "elorasch"]
METHODS = ["none", "platt", "isotonic", "temperature"]
FOLDS = [0, 1, 2, 3, 4]
OUT = paths.ARTIFACTS_DIR / "cross_dataset"
CACHE = paths.ARTIFACTS_DIR / "predictions_valid_cache"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_cell(dataset: str, family: str, model: str, fold: int, granularity: str):
    """Return (test_y, test_p, valid_y, valid_p, groups) or None with a reason.

    The three combinations are kept apart deliberately; mixing the fit side of
    one with the apply side of another is the defect this paper measures.
    """
    if family == "classical":
        if granularity != "question":
            return None, "classical models have no concept-level output"
        p = CACHE / f"{dataset}__{model}__fold{fold}.npz"
        if not p.exists():
            return None, "no cached classical predictions"
        d = np.load(p)
        return (d["y_true"].astype(int), d["y_prob"].astype(float),
                d["valid_y_true"].astype(int), d["valid_y_prob"].astype(float),
                np.asarray(d["groups"])), None

    p = paths.ARTIFACTS_DIR / "predictions" / dataset / f"{model}_fold{fold}.npz"
    if not p.exists():
        return None, "no prediction npz"
    d = np.load(p)
    f = set(d.files)
    if granularity == "concept":
        need = {"concept_y_true", "concept_y_prob", "valid_y_true",
                "valid_y_prob", "concept_groups"}
        if not need <= f:
            return None, f"missing {sorted(need - f)}"
        return (d["concept_y_true"].astype(int), d["concept_y_prob"].astype(float),
                d["valid_y_true"].astype(int), d["valid_y_prob"].astype(float),
                np.asarray(d["concept_groups"])), None
    need = {"y_true", "y_prob", "valid_y_true_q_pykt", "valid_y_prob_q_pykt", "groups"}
    if not need <= f:
        return None, f"missing {sorted(need - f)}"
    return (d["y_true"].astype(int), d["y_prob"].astype(float),
            d["valid_y_true_q_pykt"].astype(int),
            d["valid_y_prob_q_pykt"].astype(float),
            np.asarray(d["groups"])), None


def calibrated_probabilities(test_p, valid_y, valid_p) -> dict:
    """The four probability vectors compared throughout the paper."""
    out = {"none": np.asarray(test_p, dtype=float)}
    for name, Cal in CALIBRATORS.items():
        out[name] = Cal().fit(valid_p, valid_y).transform(test_p)
    return out


def _ece(y, p, groups):
    cnt, sy, sp, _ = student_bin_stats(y, p, groups)
    return float(ece_from_totals(cnt.sum(0), sy.sum(0), sp.sum(0)))


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    # Десять тысяч, а не две: при двух тысячах наименьшее достижимое
    # p-значение равно 2/2001, и после поправки Холма на семью из
    # пятидесяти двух сравнений оно становится 0.052 — вся семья
    # оказывается непроходимой независимо от величины различий.
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    cell_rows, method_rows, model_rows, skipped = [], [], [], []
    # probabilities are kept per (dataset, granularity, fold) so that model
    # contrasts can be formed without recalibrating anything.
    store: dict[tuple, dict] = defaultdict(dict)

    for dataset in DATASETS:
        for granularity in ("concept", "question"):
            fams = (("deep", DEEP),) if granularity == "concept" else \
                   (("deep", DEEP), ("classical", CLASSICAL))
            for family, models in fams:
                for model in models:
                    for fold in args.folds:
                        loaded, why = load_cell(dataset, family, model, fold,
                                                granularity)
                        if loaded is None:
                            skipped.append((dataset, granularity, family, model,
                                            fold, why))
                            continue
                        ty, tp, vy, vp, groups = loaded
                        probs = calibrated_probabilities(tp, vy, vp)
                        eces = {m: _ece(ty, probs[m], groups) for m in METHODS}
                        for m in METHODS:
                            cell_rows.append({
                                "dataset": dataset, "granularity": granularity,
                                "family": family, "model": model, "fold": fold,
                                "method": m,
                                "ece": eces[m],
                                "auc": _auc(ty, probs[m]),
                                "n_test": int(ty.size),
                                "n_students": int(np.unique(groups).size),
                            })
                        store[(dataset, granularity, fold)][(family, model)] = \
                            (ty, probs, groups, eces)

                        # (1) pre-specified contrast
                        r = paired_cluster_bootstrap_ece(
                            ty, probs["isotonic"], probs["none"], groups,
                            n_boot=args.n_boot, seed=fold)
                        method_rows.append({
                            "dataset": dataset, "granularity": granularity,
                            "family": family, "model": model, "fold": fold,
                            "contrast": "iso_vs_none",
                            "method_a": "isotonic", "method_b": "none",
                            "ece_a": r.ece_a, "ece_b": r.ece_b,
                            "delta_ece": r.diff,
                            "ci_low": r.ci_low, "ci_high": r.ci_high,
                            "p": r.p_value, "n_students": r.n_students,
                            "n_test": r.n_rows,
                            "family_key": f"{granularity}|f{fold}|iso_vs_none",
                        })

                        # (2) post-selection contrast, reported as such
                        order = sorted(METHODS, key=lambda m: eces[m])
                        best, runner = order[0], order[1]
                        r2 = paired_cluster_bootstrap_ece(
                            ty, probs[best], probs[runner], groups,
                            n_boot=args.n_boot, seed=fold)
                        method_rows.append({
                            "dataset": dataset, "granularity": granularity,
                            "family": family, "model": model, "fold": fold,
                            "contrast": "best_vs_runnerup",
                            "method_a": best, "method_b": runner,
                            "ece_a": r2.ece_a, "ece_b": r2.ece_b,
                            "delta_ece": r2.diff,
                            "ci_low": r2.ci_low, "ci_high": r2.ci_high,
                            "p": r2.p_value, "n_students": r2.n_students,
                            "n_test": r2.n_rows,
                            "family_key": f"{granularity}|f{fold}|best_vs_runnerup",
                        })
                        print(f"[{dataset:20s} {granularity:8s} {family:9s} "
                              f"{model:9s} f{fold}] "
                              f"none={eces['none']:.5f} iso={eces['isotonic']:.5f} "
                              f"Δ={r.diff:+.5f} p={r.p_value:.4f} | best={best}")

    # ------------------------------------------------------------------ #
    # model contrasts
    # ------------------------------------------------------------------ #
    for (dataset, granularity, fold), cells in sorted(store.items()):
        deep = {mdl: v for (fam, mdl), v in cells.items() if fam == "deep"}
        clas = {mdl: v for (fam, mdl), v in cells.items() if fam == "classical"}

        def best_of(group):
            """(model, method, ece) with the smallest calibration error."""
            out = []
            for mdl, (_, _, _, eces) in group.items():
                m = min(METHODS, key=lambda k: eces[k])
                out.append((mdl, m, eces[m]))
            return sorted(out, key=lambda t: t[2])

        if len(deep) >= 2:
            ranked = best_of(deep)
            (ma, meth_a, ea), (mb, meth_b, eb) = ranked[0], ranked[1]
            ty_a, probs_a, groups_a, _ = deep[ma]
            _, probs_b, _, _ = deep[mb]
            r = paired_cluster_bootstrap_ece(
                ty_a, probs_a[meth_a], probs_b[meth_b], groups_a,
                n_boot=args.n_boot, seed=fold)
            model_rows.append({
                "dataset": dataset, "granularity": granularity, "fold": fold,
                "contrast_type": "within_deep",
                "model_a": ma, "method_a": meth_a, "family_a": "deep",
                "model_b": mb, "method_b": meth_b, "family_b": "deep",
                "ece_a": r.ece_a, "ece_b": r.ece_b, "delta_ece": r.diff,
                "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p_value,
                "n_students": r.n_students, "pairing": "paired",
                "family_key": f"{granularity}|f{fold}|within_deep",
            })

        if deep and clas:
            rd = best_of(deep)[0]
            rc = best_of(clas)[0]
            (ma, meth_a, ea), (mb, meth_b, eb) = (rd, rc) if rd[2] <= rc[2] else (rc, rd)
            fam_a = "deep" if (ma in deep) else "classical"
            fam_b = "deep" if (mb in deep) else "classical"
            ty_a, probs_a, groups_a, _ = (deep if fam_a == "deep" else clas)[ma]
            ty_b, probs_b, groups_b, _ = (deep if fam_b == "deep" else clas)[mb]
            r = unpaired_cluster_bootstrap_ece(
                ty_a, probs_a[meth_a], groups_a,
                ty_b, probs_b[meth_b], groups_b,
                n_boot=args.n_boot, seed=fold)
            model_rows.append({
                "dataset": dataset, "granularity": granularity, "fold": fold,
                "contrast_type": "cross_family",
                "model_a": ma, "method_a": meth_a, "family_a": fam_a,
                "model_b": mb, "method_b": meth_b, "family_b": fam_b,
                "ece_a": r.ece_a, "ece_b": r.ece_b, "delta_ece": r.diff,
                "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p_value,
                "n_students": r.n_students, "pairing": "unpaired_shared_students",
                "family_key": f"{granularity}|f{fold}|cross_family",
            })

    # ------------------------------------------------------------------ #
    # Holm inside each family, then write
    # ------------------------------------------------------------------ #
    def add_holm(rows):
        by = defaultdict(list)
        for i, r in enumerate(rows):
            by[r["family_key"]].append(i)
        for key, idxs in by.items():
            res = holm_bonferroni([rows[i]["p"] for i in idxs], alpha=args.alpha)
            for j, i in enumerate(idxs):
                rows[i]["p_holm"] = res["adjusted"][j]
                rows[i]["reject_holm"] = bool(res["reject"][j])
                rows[i]["family_size"] = len(idxs)
        return rows

    method_rows = add_holm(method_rows)
    model_rows = add_holm(model_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("c4_cell_ece.csv", cell_rows),
                       ("c4_method_contrasts.csv", method_rows),
                       ("c4_model_contrasts.csv", model_rows)):
        if not rows:
            continue
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {OUT / name}")

    if skipped:
        print(f"\nskipped {len(skipped)} cells:")
        agg = defaultdict(int)
        for ds, g, fam, m, f, why in skipped:
            agg[(ds, g, fam, why)] += 1
        for (ds, g, fam, why), n in sorted(agg.items()):
            print(f"  {ds:20s} {g:8s} {fam:9s} x{n:2d}  {why}")


if __name__ == "__main__":
    main()
