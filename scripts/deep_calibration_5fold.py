"""5-fold deep calibration aggregator at concept granularity.

Supersedes `calibrate_deep_from_npz.py` (deleted 2026-08-31): instead of fold 0 only, scans
`artifacts/predictions/<dataset>/<model>_fold*.npz` for every fold that carries
held-out valid + concept-test predictions, fits each calibrator on valid and applies
it to concept-test (held-out protocol, D008), then aggregates per (dataset, model,
method) to mean±std ECE / AUC / Brier across folds.

Outputs:
  - deep_calibration_5fold_perfold.csv  (per-fold rows, full trail)
  - deep_calibration_5fold.csv          (mean±std, one row per dataset×model×method)

Only entries with >= 1 usable fold are written. (dataset, model) cells with partial
coverage are kept so we can still report mean±std on whatever folds we have, with the
fold list logged.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

# Platt's sklearn LogisticRegression issues benign divide/overflow on large logits
# (sigmoid saturates exactly to 0/1); doesn't affect the fit.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"sklearn\.linear_model\._linear_loss")

from ktx import paths
from ktx.calibration import CALIBRATORS
from ktx.metrics import compute_metrics

DEEP = {"dkt", "sakt", "akt", "dkvmn", "saint", "simplekt"}
FOLD_RE = re.compile(r"(?P<model>[a-z+]+)_fold(?P<fold>\d+)\.npz$")


def _iter_npz(pred_root: Path):
    """Yield (dataset, model, fold, path) for every deep-model npz file under predictions/."""
    for ds_dir in sorted(p for p in pred_root.iterdir() if p.is_dir()):
        for npz in sorted(ds_dir.glob("*_fold*.npz")):
            m = FOLD_RE.search(npz.name)
            if not m:
                continue
            model = m.group("model")
            if model not in DEEP:
                continue
            yield ds_dir.name, model, int(m.group("fold")), npz


def _calibrate_one(npz_path: Path):
    """Return list of dicts: one per method (incl. 'none'). None if npz lacks valid/concept."""
    d = np.load(npz_path)
    if not {"valid_y_true", "valid_y_prob", "concept_y_true", "concept_y_prob"} <= set(d.files):
        return None
    vy, vp = d["valid_y_true"], d["valid_y_prob"]
    ty, tp = d["concept_y_true"], d["concept_y_prob"]
    out = []
    mb = compute_metrics(ty, tp)
    out.append({"method": "none", "ece": mb.ece, "auc": mb.auc, "brier": mb.brier, "n_test": mb.n})
    for name, Cal in CALIBRATORS.items():
        tc = Cal().fit(vp, vy).transform(tp)
        ma = compute_metrics(ty, tc)
        out.append({"method": name, "ece": ma.ece, "auc": ma.auc, "brier": ma.brier, "n_test": ma.n})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perfold-out",
                    default=str(paths.ARTIFACTS_DIR / "cross_dataset" / "deep_calibration_5fold_perfold.csv"))
    ap.add_argument("--agg-out",
                    default=str(paths.ARTIFACTS_DIR / "cross_dataset" / "deep_calibration_5fold.csv"))
    ap.add_argument("--models", nargs="+", default=None,
                    help="restrict to these models. Default: every deep model "
                         "with predictions on disk, which includes leftovers "
                         "from earlier phases (saint) that are not in the "
                         "paper roster.")
    ap.add_argument("--decimals", type=int, default=4,
                    help="rounding applied when writing CSV. Default 4 keeps the "
                         "historical format; use 6 for new artifacts, where "
                         "differences of order 1e-4 are compared.")
    args = ap.parse_args()
    nd = args.decimals

    pred_root = paths.ARTIFACTS_DIR / "predictions"
    perfold_rows = []
    skipped = []
    wanted = set(args.models) if args.models else None
    for dataset, model, fold, npz in _iter_npz(pred_root):
        if wanted is not None and model not in wanted:
            continue
        rows = _calibrate_one(npz)
        if rows is None:
            skipped.append((dataset, model, fold, "no valid/concept fields"))
            continue
        for r in rows:
            # Raw floats are kept here and rounded only at write time: the fold
            # mean must be taken over the unrounded values, otherwise the
            # aggregate inherits the rounding error of five per-fold numbers.
            perfold_rows.append({
                "dataset": dataset, "model": model, "fold": fold,
                "method": r["method"],
                "ece": float(r["ece"]),
                "auc": float(r["auc"]),
                "brier": float(r["brier"]),
                "n_test": int(r["n_test"]),
            })

    if not perfold_rows:
        print("no usable deep npz with valid+concept fields found.")
        if skipped:
            print(f"skipped {len(skipped)} npz without valid/concept fields:")
            for ds, m, f, why in skipped[:10]:
                print(f"  - {ds}/{m}_fold{f}: {why}")
        return

    Path(args.perfold_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.perfold_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(perfold_rows[0].keys()))
        w.writeheader()
        for r in perfold_rows:
            w.writerow({**r, "ece": round(r["ece"], nd),
                        "auc": round(r["auc"], nd), "brier": round(r["brier"], nd)})
    print(f"wrote {len(perfold_rows)} per-fold rows -> {args.perfold_out}")

    # aggregate per (dataset, model, method)
    agg: dict[tuple, list[dict]] = defaultdict(list)
    for r in perfold_rows:
        agg[(r["dataset"], r["model"], r["method"])].append(r)

    agg_rows = []
    for (ds, m, meth), rs in sorted(agg.items()):
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
    by_dm: dict[tuple, list[dict]] = defaultdict(list)
    for r in agg_rows:
        by_dm[(r["dataset"], r["model"])].append(r)
    print(f"{'dataset':22s} {'model':5s} {'best':>11s}  "
          f"{'ECE_none':>9s} -> {'ECE_best':>9s}  ΔECE   folds")
    for (ds, m), rs in sorted(by_dm.items()):
        none_r = next((x for x in rs if x["method"] == "none"), None)
        if none_r is None:
            continue
        best = min(rs, key=lambda x: x["ece_mean"])
        delta = best["ece_mean"] - none_r["ece_mean"]
        print(f"{ds:22s} {m:5s} {best['method']:>11s}  "
              f"{none_r['ece_mean']:>9.4f} -> {best['ece_mean']:>9.4f}  "
              f"{delta:+.4f}  ({best['n_folds']}/5)")

    if skipped:
        print(f"\n[note] {len(skipped)} npz skipped (no valid/concept fields, likely older SAKT/AKT fold0 — "
              f"expect Kaggle GPU output to bring them in).")


if __name__ == "__main__":
    main()
