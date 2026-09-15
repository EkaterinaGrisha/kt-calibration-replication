"""Deep post-hoc calibration on a coherent, leak-free question-level pipeline.

Contrast with earlier versions
------------------------------
* v2 (`deep_calibration_5fold_qlvl.py`): fit on `valid_y_*_q` (our
  `late_mean` aggregation over concept_y_prob — teacher-forcing-leaked on
  `is_repeat=1` positions), apply on
  `y_true`/`y_prob` (pyKT `evaluate_question` test, leak-free). The
  calibrator sees a leaked distribution during fit and a non-leaked one
  during apply — this is the mismatch the paper measures.
* v3 (retracted): fit AND
  apply on our `late_mean` aggregation on both sides. Closes the fit/apply
  distribution gap but does so at the price of using leaked probabilities
  everywhere — the resulting ECE numbers are calibrated leaked scores,
  which have no interpretable meaning.
* v4 (this script): fit on `valid_y_*_q_pykt` (pyKT `evaluate_question` on
  the freshly-generated valid-question loader, leak-free — see
  `run_valid_question_inference_local.py`), apply on `y_true`/`y_prob`
  (pyKT `evaluate_question` test, also leak-free). Both sides on the same
  cq/cshft inference path with no teacher-forcing leakage — first
  honest-to-god q-level calibration measurement in the project.

Outputs (mirror v2 field names but with `_v4` suffix — original files
kept intact so before/after tables are reproducible):
  - deep_calibration_5fold_qlvl_v4_perfold.csv
  - deep_calibration_5fold_qlvl_v4.csv
  - deep_calibration_5fold_qlvl_v4_diff.csv  (per-cell v2 vs v4 ECE gap;
    acceptance criterion: |Δ| < 0.005 → the earlier result was measuring
    the double-pipeline artifact)

Run:
    python -m scripts.deep_calibration_5fold_qlvl_v4
"""
from __future__ import annotations

import argparse
import csv
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"sklearn\.linear_model\._linear_loss")

from ktx import paths
from ktx.calibration import CALIBRATORS
from ktx.metrics import compute_metrics

DEEP = {"dkt", "sakt", "akt", "simplekt"}
FOLD_RE = re.compile(r"(?P<model>[a-z+]+)_fold(?P<fold>\d+)\.npz$")


def _iter_npz(pred_root: Path):
    for ds_dir in sorted(p for p in pred_root.iterdir() if p.is_dir()):
        for npz in sorted(ds_dir.glob("*_fold*.npz")):
            m = FOLD_RE.search(npz.name)
            if not m:
                continue
            model = m.group("model")
            if model not in DEEP:
                continue
            yield ds_dir.name, model, int(m.group("fold")), npz


def _calibrate_leakfree(npz_path: Path):
    d = np.load(npz_path)
    need = {"valid_y_true_q_pykt", "valid_y_prob_q_pykt", "y_true", "y_prob"}
    if not need <= set(d.files):
        return None, "missing valid_y_*_q_pykt or y_true/y_prob"
    vy, vp = d["valid_y_true_q_pykt"], d["valid_y_prob_q_pykt"]
    ty, tp = d["y_true"], d["y_prob"]

    out = []
    mb = compute_metrics(ty, tp)
    out.append({"method": "none", "ece": mb.ece, "auc": mb.auc,
                "brier": mb.brier, "n_test": mb.n, "n_valid": int(vy.size)})
    for name, Cal in CALIBRATORS.items():
        tc = Cal().fit(vp, vy).transform(tp)
        ma = compute_metrics(ty, tc)
        out.append({"method": name, "ece": ma.ece, "auc": ma.auc,
                    "brier": ma.brier, "n_test": ma.n, "n_valid": int(vy.size)})
    return out, None


def _load_v2_map(v2_csv: Path):
    if not v2_csv.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(v2_csv)):
        try:
            out[(r["dataset"], r["model"], r["method"])] = float(r["ece_mean"])
        except (KeyError, ValueError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perfold-out",
                    default=str(paths.ARTIFACTS_DIR / "cross_dataset"
                                / "deep_calibration_5fold_qlvl_v4_perfold.csv"))
    ap.add_argument("--agg-out",
                    default=str(paths.ARTIFACTS_DIR / "cross_dataset"
                                / "deep_calibration_5fold_qlvl_v4.csv"))
    ap.add_argument("--diff-out",
                    default=str(paths.ARTIFACTS_DIR / "cross_dataset"
                                / "deep_calibration_5fold_qlvl_v4_diff.csv"))
    ap.add_argument("--v2-csv",
                    default=str(paths.ARTIFACTS_DIR / "cross_dataset"
                                / "deep_calibration_5fold_qlvl.csv"))
    ap.add_argument("--decimals", type=int, default=4,
                    help="rounding applied when writing CSV. 4 keeps the "
                         "historical format; 6 is needed whenever the identity "
                         "of the minimum is read off these numbers, since the "
                         "spread between neighbouring cells is ~1e-4.")
    args = ap.parse_args()
    nd = args.decimals

    pred_root = paths.ARTIFACTS_DIR / "predictions"
    perfold_rows = []
    skipped = []
    for dataset, model, fold, npz in _iter_npz(pred_root):
        rows, why = _calibrate_leakfree(npz)
        if rows is None:
            skipped.append((dataset, model, fold, why))
            continue
        for r in rows:
            perfold_rows.append({
                "dataset": dataset, "model": model, "fold": fold,
                "method": r["method"],
                "ece": float(r["ece"]),
                "auc": float(r["auc"]),
                "brier": float(r["brier"]),
                "n_test": int(r["n_test"]),
                "n_valid": int(r["n_valid"]),
                "pipeline": "leakfree_pykt_v4",
            })

    if not perfold_rows:
        print("no usable deep npz found for v4 calibration "
              "(no valid_y_*_q_pykt fields — did the T1b sweep run?)")
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

    agg = defaultdict(list)
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

    v2_map = _load_v2_map(Path(args.v2_csv))
    if v2_map:
        diff_rows = []
        for r in agg_rows:
            key = (r["dataset"], r["model"], r["method"])
            ec_v2 = v2_map.get(key)
            if ec_v2 is None:
                continue
            delta = r["ece_mean"] - ec_v2
            diff_rows.append({
                "dataset": r["dataset"], "model": r["model"],
                "method": r["method"],
                "ece_v2_leakfit": round(ec_v2, nd),
                "ece_v4_leakfree": round(r["ece_mean"], nd),
                "delta_v4_minus_v2": round(delta, nd),
                "ratio_v2_over_v4": (round(ec_v2 / r["ece_mean"], 2)
                                     if r["ece_mean"] > 0 else float("nan")),
            })
        with open(args.diff_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(diff_rows[0].keys()))
            w.writeheader(); w.writerows(diff_rows)
        print(f"wrote {len(diff_rows)} diff rows -> {args.diff_out}")

        deltas = np.array([r["delta_v4_minus_v2"] for r in diff_rows
                           if r["method"] == "isotonic"], dtype=float)
        if deltas.size:
            print(f"\nT1b acceptance (isotonic only, {deltas.size} cells):")
            print(f"  |delta| median = {float(np.median(np.abs(deltas))):.4f}")
            print(f"  |delta| max    = {float(np.max(np.abs(deltas))):.4f}")
            print(f"  cells with |delta| >= 0.005: "
                  f"{int(np.sum(np.abs(deltas) >= 0.005))}/{deltas.size}")

    if skipped:
        print(f"\nskipped {len(skipped)} npz (no valid_y_*_q_pykt or no y_true):")
        for ds, m, f, why in skipped[:12]:
            print(f"  - {ds}/{m}_fold{f}: {why}")


if __name__ == "__main__":
    main()
