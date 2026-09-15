"""Доверительные интервалы для величин, которые печатают таблицы 3 и 9.

Третье правило протокола сообщения требует, чтобы точечное значение шло с
интервалом и с числом учащихся. Ячейковые интервалы уже считают `c4_path.py` и
`c4_subgroups.py`, но таблицы 3 и 9 показывают не ячейку, а среднее по моделям и
разбиениям, и интервал такого среднего из ячейковых не складывается.

Здесь он считается прямо. Тестовая часть в протоколе одна и та же на всех пяти
разбиениях и на всех моделях, поэтому повторную выборку учащихся достаточно
провести один раз на набор данных и внутри каждого повторения пересчитать всю
строку таблицы целиком — среднее по ячейкам, отношение средних, ошибку внутри
группы. Это кластерный бутстрап для той величины, которая напечатана, а не для
соседней.

Счёт идёт по достаточным статистикам: на каждого учащегося и каждую корзину
вероятности хранятся число строк, сумма меток и сумма вероятностей, так что одно
повторение — умножение вектора кратностей на матрицу, а не пересчёт по строкам.

Выход `c4_aggregate_ci.csv`: строка на величину таблицы.

Использование:
    python -m scripts.c4_aggregate_ci
    python -m scripts.c4_aggregate_ci --n-boot 2000 --only ednet
"""
from __future__ import annotations

import argparse
import csv
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"sklearn\.linear_model\._linear_loss")

from ktx import paths
from ktx.calibration import CALIBRATORS
from ktx.stats_fast import ece_from_totals, student_bin_stats
from scripts.c4_significance import DATASETS, DEEP, FOLDS, load_cell
from scripts.c4_subgroups import (
    MIN_EVAL_ROWS,
    MIN_ROWS,
    MIN_STUDENTS,
    _blocks,
    _quartiles,
)

OUT = paths.ARTIFACTS_DIR / "cross_dataset"
N_BINS = 10
PREFIX = 10                     # окно признака способности, как в таблице 9
PATH_METHODS = ("isotonic", "platt", "temperature")


# --------------------------------------------------------------------------- #
def _stats(y, p, groups, codes):
    """(cnt, sy, sp) по учащимся; порядок учащихся сверяется с набором данных."""
    cnt, sy, sp, got = student_bin_stats(y, p, groups, N_BINS)
    if got.size != codes.size or not np.array_equal(got, codes):
        raise SystemExit("состав учащихся в ячейке не совпал с набором данных")
    return cnt, sy, sp


def _stack(blocks):
    """Ячейки в одну матрицу (S, n_cells*3*n_bins) для одного умножения."""
    return np.concatenate([np.concatenate(b, axis=1) for b in blocks], axis=1)


def _draws(rng, n_students, n_boot):
    """Кратности учащихся: (n_boot, n_students), строка на повторение."""
    pick = rng.integers(0, n_students, size=(n_boot, n_students))
    flat = pick + (np.arange(n_boot) * n_students)[:, None]
    return np.bincount(flat.ravel(),
                       minlength=n_boot * n_students).astype(np.float64
                                                             ).reshape(n_boot, n_students)


def _cell_ece(m_chunk, big, n_cells):
    """ECE каждой ячейки в каждом повторении: (n_boot_chunk, n_cells)."""
    got = (m_chunk @ big).reshape(m_chunk.shape[0], n_cells, 3, N_BINS)
    with np.errstate(divide="ignore", invalid="ignore"):
        return ece_from_totals(got[:, :, 0], got[:, :, 1], got[:, :, 2])


def _observed(blocks):
    """Среднее по ячейкам от наблюдённой ошибки."""
    return float(np.mean([float(ece_from_totals(c.sum(0), y.sum(0), p.sum(0)))
                          for c, y, p in blocks]))


def _spread(values, alpha):
    """Границы процентильного интервала и среднее по повторениям.

    Среднее записывается рядом с ними не для украшения. Ошибка калибровки —
    среднее модулей, поэтому шум повторной выборки её завышает: у повторения
    зазор в корзине по модулю больше, чем у исходной выборки. На разности и на
    отношении двух ошибок этот сдвиг сокращается, на одной ошибке — нет, и
    сравнение точечного значения со средним по повторениям показывает, насколько
    он велик в каждой строке.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2)),
            float(np.mean(v)))


def _boot(rng, n_students, n_boot, chunk, fn):
    """Пройти повторения кусками и склеить то, что вернул `fn` на каждом куске."""
    out, left = [], n_boot
    while left > 0:
        b = min(chunk, left)
        out.append(fn(_draws(rng, n_students, b)))
        left -= b
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
def path_intervals(dataset, folds, rng, args, rows):
    """Таблица 3: средние по ячейкам на двух путях, их разность и отношение."""
    cells = {m: [] for m in PATH_METHODS}
    codes, n_rows = None, 0
    for model in DEEP:
        for fold in folds:
            p = paths.ARTIFACTS_DIR / "predictions" / dataset / f"{model}_fold{fold}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            need = {"y_true", "y_prob", "groups", "valid_y_true_q_pykt",
                    "valid_y_prob_q_pykt", "valid_y_true_q", "valid_y_prob_q"}
            if not need <= set(d.files):
                continue
            ty = d["y_true"].astype(int)
            tp = d["y_prob"].astype(float)
            groups = np.asarray(d["groups"])
            if codes is None:
                codes, n_rows = np.unique(groups), int(ty.size)
            for name in PATH_METHODS:
                Cal = CALIBRATORS[name]
                p_mm = Cal().fit(d["valid_y_prob_q"].astype(float),
                                 d["valid_y_true_q"].astype(int)).transform(tp)
                p_co = Cal().fit(d["valid_y_prob_q_pykt"].astype(float),
                                 d["valid_y_true_q_pykt"].astype(int)).transform(tp)
                cells[name].append((_stats(ty, p_mm, groups, codes),
                                    _stats(ty, p_co, groups, codes)))
    if codes is None:
        return

    for name in PATH_METHODS:
        got = cells[name]
        if not got:
            continue
        blocks = [arm for cell in got for arm in cell]     # mm, co, mm, co, ...
        big = _stack(blocks)
        n = len(got)

        def fn(m_chunk, big=big, n=n):
            e = _cell_ece(m_chunk, big, 2 * n).reshape(m_chunk.shape[0], n, 2)
            mm = np.nanmean(e[:, :, 0], axis=1)
            co = np.nanmean(e[:, :, 1], axis=1)
            return np.stack([mm, co, mm - co, mm / co], axis=1)

        rep = _boot(rng, codes.size, args.n_boot, args.chunk, fn)
        obs_mm = _observed([a for a, _ in got])
        obs_co = _observed([b for _, b in got])
        points = {"mean_mismatched": obs_mm, "mean_coherent": obs_co,
                  "delta": obs_mm - obs_co, "ratio": obs_mm / obs_co}
        for j, (q, point) in enumerate(points.items()):
            lo, hi, mean = _spread(rep[:, j], args.alpha)
            rows.append({"kind": "path", "dataset": dataset, "key": name,
                         "quantity": q, "point": point, "ci_low": lo, "ci_high": hi,
                         "boot_mean": mean, "n_students": int(codes.size),
                         "n_cells": n, "n_rows": n_rows})
        r_lo, r_hi, _ = _spread(rep[:, 3], args.alpha)
        print(f"[путь   {dataset:20s} {name:11s}] ячеек {n:2d}  "
              f"отношение {obs_mm / obs_co:.2f} [{r_lo:.2f}; {r_hi:.2f}]")


# --------------------------------------------------------------------------- #
def ability_intervals(dataset, folds, rng, args, rows):
    """Таблица 9: ошибка в целом и по группам способности, уровень заданий."""
    per_q, overall, codes_e, q_of, n_eval = {}, [], None, None, 0
    for model in DEEP:
        for fold in folds:
            loaded, _ = load_cell(dataset, "deep", model, fold, "question")
            if loaded is None:
                continue
            ty, tp, vy, vp, groups = loaded
            p_iso = CALIBRATORS["isotonic"]().fit(vp, vy).transform(tp)
            codes, starts, ends = _blocks(np.asarray(groups))
            keep = (ends - starts) >= PREFIX + MIN_EVAL_ROWS
            if keep.sum() < 2 * MIN_STUDENTS:
                continue
            eval_mask = np.zeros(ty.size, dtype=bool)
            acc = np.full(codes.size, np.nan)
            for i in range(codes.size):
                if not keep[i]:
                    continue
                s0, s1 = starts[i], ends[i]
                acc[i] = ty[s0:s0 + PREFIX].mean()
                eval_mask[s0 + PREFIX:s1] = True
            q = _quartiles(acc)
            q[~keep] = -1
            idx = np.flatnonzero(eval_mask)
            if idx.size == 0:
                continue
            ty_e, groups_e = ty[idx], np.asarray(groups)[idx]
            here = np.unique(groups_e)
            pos = {int(c): i for i, c in enumerate(here)}
            qe = np.full(here.size, -1, dtype=int)
            for i, c in enumerate(codes):
                j = pos.get(int(c))
                if j is not None:
                    qe[j] = q[i]
            if codes_e is None:
                codes_e, q_of, n_eval = here, qe, int(idx.size)
            elif not (np.array_equal(here, codes_e) and np.array_equal(qe, q_of)):
                raise SystemExit(f"{dataset}: состав групп разошёлся между ячейками")
            st = _stats(ty_e, p_iso[idx], groups_e, codes_e)
            overall.append(st)
            for qi in sorted(set(qe.tolist()) - {-1}):
                mask = qe == qi
                if float(st[0][mask].sum()) < MIN_ROWS or int(mask.sum()) < MIN_STUDENTS:
                    continue
                # учащиеся вне группы обнуляются, а не выбрасываются: повторная
                # выборка идёт по всем учащимся набора, и ширина матрицы должна
                # совпадать с длиной вектора кратностей
                per_q.setdefault(qi, []).append(
                    tuple(np.where(mask[:, None], a, 0.0) for a in st))
    if codes_e is None or not overall:
        return

    state = rng.bit_generator.state          # один и тот же набор повторений всем

    def series(blocks):
        rng.bit_generator.state = state
        big, n = _stack(blocks), len(blocks)
        return _boot(rng, codes_e.size, args.n_boot, args.chunk,
                     lambda m, big=big, n=n: np.nanmean(_cell_ece(m, big, n), axis=1))

    def emit(key, blocks, students, rep=None):
        rep = series(blocks) if rep is None else rep
        point = _observed(blocks)
        lo, hi, mean = _spread(rep, args.alpha)
        rows.append({"kind": "ability_group", "dataset": dataset, "key": key,
                     "quantity": "ece", "point": point, "ci_low": lo, "ci_high": hi,
                     "boot_mean": mean, "n_students": students,
                     "n_cells": len(blocks), "n_rows": n_eval})
        return rep, point

    rep_all, point_all = emit("overall", overall, int(codes_e.size))
    reps = {}
    for qi in sorted(per_q):
        reps[qi] = emit(f"group{qi}", per_q[qi], int((q_of == qi).sum()))
    if 0 in reps:
        rep_low, point_low = reps[0]
        lo, hi, mean = _spread(rep_low / rep_all, args.alpha)
        rows.append({"kind": "ability_group", "dataset": dataset,
                     "key": "ratio_low_over_overall", "quantity": "ratio",
                     "point": point_low / point_all, "ci_low": lo, "ci_high": hi,
                     "boot_mean": mean, "n_students": int((q_of == 0).sum()),
                     "n_cells": len(per_q[0]), "n_rows": n_eval})
    print(f"[группы {dataset:20s}] ячеек {len(overall):2d}  групп {len(per_q)}  "
          f"учащихся {codes_e.size}  строк оценки {n_eval}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=500,
                    help="сколько повторений считать за одно умножение матриц")
    ap.add_argument("--only", nargs="+", default=None, help="только эти наборы")
    args = ap.parse_args()

    rows = []
    for dataset in DATASETS:
        if args.only and dataset not in args.only:
            continue
        path_intervals(dataset, args.folds, np.random.default_rng(args.seed), args, rows)
        ability_intervals(dataset, args.folds, np.random.default_rng(args.seed + 1),
                          args, rows)

    if not rows:
        print("нечего записывать: файлы предсказаний не найдены")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "c4_aggregate_ci.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"записано {len(rows)} строк -> {OUT / 'c4_aggregate_ci.csv'}")


if __name__ == "__main__":
    main()
