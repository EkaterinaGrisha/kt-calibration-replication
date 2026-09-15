"""5-fold calibration of the classical models.

Supersedes the fold-0-only run_calibration.py (deleted 2026-08-31) by covering all
5 folds, per the same held-out
validation protocol (D008). For each (dataset, model, fold) we:
  - refit the classical model on train, hold out valid,
  - fit each calibrator on valid predictions, transform test predictions,
  - record ECE / AUC / Brier (none + 3 methods) per fold.

Then aggregate per (dataset, model, method) to mean±std across folds, parallel to
deep_calibration_5fold.csv so the two families can be compared apples-to-apples in
the unified ranking.

Outputs:
  - classical_calibration_5fold_perfold.csv (per fold)
  - classical_calibration_5fold.csv         (mean±std across folds)
"""
from __future__ import annotations

import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from ktx import paths
from ktx.calibration import CALIBRATORS
from ktx.classical.runner import run_classical_experiment
from ktx.metrics import compute_metrics

# Predictions cached by the cluster-bootstrap scripts: one npz per
# (dataset, model, fold) holding test and held-out-valid probabilities plus the
# student id of every test row. Reading them makes this script reproducible
# without refitting 140 classical models, and guarantees that the calibration
# table and the bootstrap contrasts are computed on the very same predictions.
_CACHE_DIR = paths.ARTIFACTS_DIR / "predictions_valid_cache"

# benign sklearn LR overflow on saturated logits — same as deep_calibration_5fold
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"sklearn\.linear_model\._linear_loss")

CLASSICAL = ["pfa", "elorasch", "pfa_recency", "bkt"]
DATASETS = ["assist2009", "assist2015", "assist2017", "algebra2005",
            "bridge2algebra2006", "assist2012", "ednet"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=CLASSICAL)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--perfold-out",
                    default=str(paths.ARTIFACTS_DIR / "cross_dataset"
                                / "classical_calibration_5fold_perfold.csv"))
    ap.add_argument("--agg-out",
                    default=str(paths.ARTIFACTS_DIR / "cross_dataset"
                                / "classical_calibration_5fold.csv"))
    ap.add_argument("--from-cache", action="store_true",
                    help="read predictions from artifacts/predictions_valid_cache "
                         "instead of refitting. Falls back to a refit for any "
                         "cell whose cache file is missing.")
    ap.add_argument("--decimals", type=int, default=4,
                    help="rounding applied when writing CSV. 4 keeps the "
                         "historical format; 6 is needed when the identity of "
                         "the minimum is read off these numbers.")
    args = ap.parse_args()
    nd = args.decimals

    perfold = []
    for dataset in args.datasets:
        for model in args.models:
            for fold in args.folds:
                tag = f"{model}/{dataset}/fold{fold}"
                cache = _CACHE_DIR / f"{dataset}__{model}__fold{fold}.npz"
                if args.from_cache and cache.exists():
                    cd = np.load(cache)
                    vy, vp = cd["valid_y_true"], cd["valid_y_prob"]
                    ty, tp = cd["y_true"], cd["y_prob"]
                else:
                    try:
                        r = run_classical_experiment(model, dataset, fold=fold)
                    except Exception as e:
                        print(f"[FAIL] {tag}: {type(e).__name__}: {e}")
                        continue
                    if r.valid_y_true is None or r.y_true is None:
                        print(f"[skip] {tag}: no valid/test preds")
                        continue
                    vy, vp = np.asarray(r.valid_y_true), np.asarray(r.valid_y_prob)
                    ty, tp = np.asarray(r.y_true), np.asarray(r.y_prob)
                m_b = compute_metrics(ty, tp)
                base = {"dataset": dataset, "model": model, "fold": fold,
                        "n_test": int(m_b.n)}
                perfold.append({**base, "method": "none",
                                "ece": float(m_b.ece),
                                "auc": float(m_b.auc),
                                "brier": float(m_b.brier)})
                for name, Cal in CALIBRATORS.items():
                    tp_cal = Cal().fit(vp, vy).transform(tp)
                    m_a = compute_metrics(ty, tp_cal)
                    perfold.append({**base, "method": name,
                                    "ece": float(m_a.ece),
                                    "auc": float(m_a.auc),
                                    "brier": float(m_a.brier)})
                    print(f"[ok] {tag:30s} {name:11s} "
                          f"ECE {m_b.ece:.4f}->{m_a.ece:.4f} AUC {m_a.auc:.4f}")

    if not perfold:
        print("no rows produced; check inputs/preds.")
        return

    Path(args.perfold_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.perfold_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(perfold[0].keys()))
        w.writeheader()
        for r in perfold:
            w.writerow({**r, "ece": round(r["ece"], nd),
                        "auc": round(r["auc"], nd), "brier": round(r["brier"], nd)})
    print(f"\nwrote {len(perfold)} per-fold rows -> {args.perfold_out}")

    # aggregate
    by = defaultdict(list)
    for r in perfold:
        by[(r["dataset"], r["model"], r["method"])].append(r)
    agg_rows = []
    for (ds, m, meth), rs in sorted(by.items()):
        ece = np.array([r["ece"] for r in rs], dtype=float)
        auc = np.array([r["auc"] for r in rs], dtype=float)
        brier = np.array([r["brier"] for r in rs], dtype=float)
        folds = sorted({r["fold"] for r in rs})
        agg_rows.append({
            "dataset": ds, "model": m, "method": meth,
            "n_folds": len(folds),
            "folds": ",".join(str(f) for f in folds),
            "ece_mean": round(float(ece.mean()), nd),
            "ece_std": round(float(ece.std(ddof=1)) if len(ece) > 1 else 0.0, nd),
            "auc_mean": round(float(auc.mean()), nd),
            "auc_std": round(float(auc.std(ddof=1)) if len(auc) > 1 else 0.0, nd),
            "brier_mean": round(float(brier.mean()), nd),
            "brier_std": round(float(brier.std(ddof=1)) if len(brier) > 1 else 0.0, nd),
            "n_test": int(rs[-1]["n_test"]),
        })
    with open(args.agg_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg_rows[0].keys()))
        w.writeheader(); w.writerows(agg_rows)
    print(f"wrote {len(agg_rows)} aggregate rows -> {args.agg_out}")

    # headline: best method per (dataset, model) by mean ECE
    print("\n=== Best calibration method per (dataset, model) by mean ECE ===")
    by_dm = defaultdict(list)
    for r in agg_rows:
        by_dm[(r["dataset"], r["model"])].append(r)
    print(f"{'dataset':22s} {'model':>12s} {'best':>11s}  "
          f"{'ECE_none':>9s} -> {'ECE_best':>9s}  ΔECE   folds")
    for (ds, m), rs in sorted(by_dm.items()):
        none_r = next((x for x in rs if x["method"] == "none"), None)
        if none_r is None:
            continue
        best = min(rs, key=lambda x: x["ece_mean"])
        d = best["ece_mean"] - none_r["ece_mean"]
        print(f"{ds:22s} {m:>12s} {best['method']:>11s}  "
              f"{none_r['ece_mean']:>9.4f} -> {best['ece_mean']:>9.4f}  "
              f"{d:+.4f}  ({best['n_folds']}/5)")


if __name__ == "__main__":
    main()
