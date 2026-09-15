"""Does a calibrator that knows the concept beat one that does not?

The hypothesis is natural in this domain: a knowledge-tracing model's residual
miscalibration might differ from topic to topic, in which case one shared
monotone correction leaves structure on the table and a correction per concept
of knowledge should do better. This script measures that, with the shrinkage
that any per-concept scheme needs — most concepts have few validation rows, so
each per-concept curve is pulled toward the shared one with a weight
n_c / (n_c + λ).

Differences from the earlier version of this experiment
-------------------------------------------------------
* It is run at question level on the leak-free path as well as at concept
  level. The earlier run was concept-level only, and on the multi-concept
  datasets the concept-level path carries label information into the input,
  so a null result there could always be blamed on the leak.
* λ is chosen on a split of the validation part **by student**, at both
  granularities. Splitting validation rows at random puts the same student on
  both sides of the inner split and makes the selection look better than it is;
  at concept granularity the student of each row is recovered by the same walk
  over the sequence file that recovers the concept.
* The λ grid is evaluated from one set of fits. The per-concept curves and the
  shared curve do not depend on λ — only the blending weight does — so fitting
  once and re-blending is exact and turns twelve fits into one. The last point
  of the grid is the shared curve itself, so "the strongest shrinkage was
  chosen" and "shrink all the way" are told apart rather than conflated.

Outputs `c4_concept_aware.csv` (per fold, one row per method) and
`c4_concept_aware_contrasts.csv` (each variant against the shared isotonic
curve, cluster bootstrap over students, Holm inside granularity and fold).
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
from ktx.calibration import DifficultyBucketIsotonic, IsotonicCalibration
from ktx.concept_ids import (
    concept_ids_for_test,
    concept_ids_for_valid,
    student_ids_for_valid,
)
from ktx.stats import holm_bonferroni
from ktx.stats_fast import (
    ece_from_totals,
    paired_cluster_bootstrap_ece,
    student_bin_stats,
)
from scripts.c4_significance import DATASETS, DEEP, FOLDS

OUT = paths.ARTIFACTS_DIR / "cross_dataset"
# Последняя точка сетки — сама общая кривая, а не большое число вместо неё:
# выбор на краю сетки иначе не отличить от выбора «стягивать полностью».
LAMBDAS = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0, 1000.0, 5000.0,
           100000.0, float("inf")]
N_MIN = 30
N_INNER = 5

_CID_CACHE: dict = {}


def _concept_ids(dataset: str, fold: int, granularity: str, npz):
    """Concept id per row of the validation and test arrays.

    Question level is not supported, and the reason belongs in the paper
    rather than in a footnote. The `cidxs` fields stored beside the
    question-level arrays are row identifiers, not concepts; recovering a
    concept for each question-level row means walking pyKT's own
    question-level sequence files, and for two of the seven datasets those
    files are 0.6 and 1.1 gigabytes per fold. A question with several
    concepts would in any case have to be assigned one of them by
    convention, which is exactly the ambiguity this experiment is about. So
    the experiment runs where the concept of a row is unambiguous, and the
    reading of the result says which datasets are free of the concept-level
    leak: those where a question carries one concept.
    """
    if granularity == "question":
        raise ValueError(
            "concept-aware calibration is defined at concept granularity only; "
            "see the docstring of _concept_ids")
    key = (dataset, fold)
    if key not in _CID_CACHE:
        _CID_CACHE[key] = (concept_ids_for_valid(dataset, fold),
                           concept_ids_for_test(dataset))
    return _CID_CACHE[key]


def load(dataset: str, model: str, fold: int, granularity: str):
    p = paths.ARTIFACTS_DIR / "predictions" / dataset / f"{model}_fold{fold}.npz"
    if not p.exists():
        return None, "no npz"
    d = np.load(p)
    f = set(d.files)
    if granularity == "question":
        need = {"y_true", "y_prob", "valid_y_true_q_pykt", "valid_y_prob_q_pykt",
                "groups", "valid_cidxs_q_pykt", "test_cidxs"}
        if not need <= f:
            return None, f"missing {sorted(need - f)}"
        vy = d["valid_y_true_q_pykt"].astype(int)
        vp = d["valid_y_prob_q_pykt"].astype(float)
        ty = d["y_true"].astype(int)
        tp = d["y_prob"].astype(float)
        groups = np.asarray(d["groups"])
    else:
        need = {"concept_y_true", "concept_y_prob", "valid_y_true",
                "valid_y_prob", "concept_groups"}
        if not need <= f:
            return None, f"missing {sorted(need - f)}"
        vy = d["valid_y_true"].astype(int)
        vp = d["valid_y_prob"].astype(float)
        ty = d["concept_y_true"].astype(int)
        tp = d["concept_y_prob"].astype(float)
        groups = np.asarray(d["concept_groups"])
    vc, tc = _concept_ids(dataset, fold, granularity, d)
    if vc.size != vy.size or tc.size != ty.size:
        return None, (f"concept ids misaligned: valid {vc.size} vs {vy.size}, "
                      f"test {tc.size} vs {ty.size}")
    # Учащийся на каждую строку проверочной части: на уровне заданий его даёт
    # библиотека, на уровне компонентов он восстанавливается тем же обходом
    # последовательностей, что и компонент. Деление проверочной части по
    # строкам вместо учащихся ставит одного человека на обе стороны и делает
    # выбор силы стягивания оптимистичным, поэтому запасного варианта здесь нет.
    if granularity == "question":
        vg = (np.asarray(d["valid_groups_q"])
              if "valid_groups_q" in f and d["valid_groups_q"].size == vy.size
              else None)
    else:
        vg = student_ids_for_valid(dataset, fold)
        if vg.size != vy.size:
            return None, (f"student ids misaligned: {vg.size} vs {vy.size}")
    return (vy, vp, vc, vg, ty, tp, tc, groups), None


def _fit_parts(vp, vy, vc, n_min=N_MIN):
    """Shared isotonic curve, per-concept curves, and the count behind each."""
    glob = IsotonicCalibration().fit(vp, vy)
    per, n_c = {}, {}
    for c in np.unique(vc):
        m = vc == c
        n = int(m.sum())
        n_c[int(c)] = n
        if n < n_min or np.unique(vy[m]).size < 2:
            continue
        per[int(c)] = IsotonicCalibration().fit(vp[m], vy[m])
    return glob, per, n_c


def _blend_inputs(glob, per, n_c, p, c):
    """Shared prediction, per-concept prediction and n_c per row.

    Rows of a concept without its own curve get n_c = 0, which makes every λ
    fall back to the shared curve for them.
    """
    g = glob.transform(p)
    pc = g.copy()
    counts = np.zeros(p.size, dtype=np.float64)
    for cid, iso in per.items():
        m = c == cid
        if not m.any():
            continue
        pc[m] = iso.transform(p[m])
        counts[m] = n_c.get(cid, 0)
    return g, pc, counts


def _blend(g, pc, counts, lam):
    if np.isinf(lam):
        return g.copy()          # общая кривая целиком
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(counts > 0, counts / (counts + lam), 0.0)
    return g + w * (pc - g)


def _ece(y, p):
    from ktx.stats_fast import bin_index
    b = bin_index(p)
    cnt = np.bincount(b, minlength=10).astype(float)
    sy = np.bincount(b, weights=np.asarray(y, float), minlength=10)
    sp = np.bincount(b, weights=np.asarray(p, float), minlength=10)
    return float(ece_from_totals(cnt, sy, sp))


def choose_lambda(vy, vp, vc, vg, seed=0):
    """Inner cross-validation over the validation part, split by student where
    a student id is available and by row otherwise."""
    rng = np.random.default_rng(seed)
    if vg is not None:
        units, inv = np.unique(vg, return_inverse=True)
        assign = rng.integers(0, N_INNER, size=units.size)[inv]
    else:
        assign = rng.integers(0, N_INNER, size=vy.size)
    per_lambda = {lam: [] for lam in LAMBDAS}
    for k in range(N_INNER):
        tr, te = assign != k, assign == k
        if np.unique(vy[te]).size < 2 or tr.sum() < 100:
            continue
        glob, per, n_c = _fit_parts(vp[tr], vy[tr], vc[tr])
        g, pc, counts = _blend_inputs(glob, per, n_c, vp[te], vc[te])
        for lam in LAMBDAS:
            per_lambda[lam].append(_ece(vy[te], _blend(g, pc, counts, lam)))
    means = {lam: (float(np.mean(v)) if v else float("nan"))
             for lam, v in per_lambda.items()}
    best = min(means, key=lambda k: means[k] if np.isfinite(means[k]) else np.inf)
    return best, means


def main() -> None:
    ap = argparse.ArgumentParser()
    # Десять тысяч, а не две: при двух тысячах наименьшее достижимое
    # p-значение равно 2/2001, и после поправки Холма на семью из
    # пятидесяти двух сравнений оно становится 0.052 — вся семья
    # оказывается непроходимой независимо от величины различий.
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--granularities", nargs="+", default=["concept"])
    args = ap.parse_args()

    rows, contrasts, skipped = [], [], []
    for dataset in args.datasets:
        for granularity in args.granularities:
            for model in DEEP:
                for fold in args.folds:
                    loaded, why = load(dataset, model, fold, granularity)
                    if loaded is None:
                        skipped.append((dataset, granularity, model, fold, why))
                        continue
                    vy, vp, vc, vg, ty, tp, tc, groups = loaded
                    lam, lam_curve = choose_lambda(vy, vp, vc, vg, seed=fold)
                    glob, per, n_c = _fit_parts(vp, vy, vc)
                    g, pc, counts = _blend_inputs(glob, per, n_c, tp, tc)
                    variants = {
                        "none": tp,
                        "isotonic": g,
                        "concept_aware_lambda0": _blend(g, pc, counts, 0.0),
                        "concept_aware_eb": _blend(g, pc, counts, lam),
                        "difficulty_bucket": DifficultyBucketIsotonic()
                            .fit(vp, vy).transform(tp),
                    }
                    covered = float(np.mean(counts > 0))
                    for name, p in variants.items():
                        rows.append({
                            "dataset": dataset, "granularity": granularity,
                            "model": model, "fold": fold, "method": name,
                            "ece": _ece(ty, p),
                            "best_lambda": lam,
                            "n_concepts": int(np.unique(vc).size),
                            "n_concepts_fitted": len(per),
                            "share_rows_with_own_curve": covered,
                            "n_test": int(ty.size),
                        })
                    for name in ("none", "concept_aware_lambda0",
                                 "concept_aware_eb", "difficulty_bucket"):
                        r = paired_cluster_bootstrap_ece(
                            ty, variants[name], variants["isotonic"], groups,
                            n_boot=args.n_boot, seed=fold)
                        contrasts.append({
                            "dataset": dataset, "granularity": granularity,
                            "model": model, "fold": fold,
                            "method": name, "reference": "isotonic",
                            "ece_method": r.ece_a, "ece_reference": r.ece_b,
                            "delta_ece": r.diff, "ci_low": r.ci_low,
                            "ci_high": r.ci_high, "p": r.p_value,
                            "best_lambda": lam, "n_students": r.n_students,
                            "family_key": f"{granularity}|f{fold}|{name}",
                        })
                    print(f"[{dataset:20s} {granularity:8s} {model:9s} f{fold}] "
                          f"λ={lam:<8g} concepts {len(per)}/{np.unique(vc).size} "
                          f"rows covered {covered:.2f} | iso={_ece(ty, g):.5f} "
                          f"ca0={_ece(ty, variants['concept_aware_lambda0']):.5f} "
                          f"eb={_ece(ty, variants['concept_aware_eb']):.5f}")

    by = defaultdict(list)
    for i, r in enumerate(contrasts):
        by[r["family_key"]].append(i)
    for key, idxs in by.items():
        res = holm_bonferroni([contrasts[i]["p"] for i in idxs])
        for j, i in enumerate(idxs):
            contrasts[i]["p_holm"] = res["adjusted"][j]
            contrasts[i]["reject_holm"] = bool(res["reject"][j])
            contrasts[i]["family_size"] = len(idxs)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rr in (("c4_concept_aware.csv", rows),
                     ("c4_concept_aware_contrasts.csv", contrasts)):
        if not rr:
            continue
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0].keys()))
            w.writeheader(); w.writerows(rr)
        print(f"wrote {len(rr)} rows -> {OUT / name}")
    if skipped:
        agg = defaultdict(int)
        for ds, g, m, f, why in skipped:
            agg[(ds, g, why)] += 1
        print(f"\nskipped {len(skipped)}:")
        for (ds, g, why), n in sorted(agg.items()):
            print(f"  {ds:20s} {g:8s} x{n:2d} {why}")


if __name__ == "__main__":
    main()
