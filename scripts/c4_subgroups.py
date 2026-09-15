"""Калибровка внутри подгрупп учащихся.

Средняя ошибка калибровки говорит, что происходит с типичным предсказанием, и
ничего не говорит о том, ровно ли ошибка распределена по учащимся. Здесь она
пересчитывается внутри групп, заданных тремя свойствами учащегося, и измеряется
разрыв между крайними группами.

Как определяется способность — и почему это главный вопрос раздела
------------------------------------------------------------------
Тестовые учащиеся в протоколе библиотеки — отдельная часть выборки: они не
встречаются ни в обучающей, ни в проверочной части, поэтому «истории до теста» у
них просто нет. Значит признак способности можно взять только из их же тестовых
ответов, а это отбор по исходу: учащийся, у которого доля верных ответов вышла
случайно низкой, на тех же строках даст и положительный зазор между
предсказанием и частотой. Регрессия к среднему тогда сама по себе создаёт
разрыв между крайними группами, и приписывать его модели нельзя.

Поэтому признак и оценка разведены по строкам. Последовательность каждого
учащегося делится на две части: по первым K ответам считается доля верных
(признак), по остальным — ошибка калибровки (оценка). Ни одна строка не участвует
в обоих. Это и корректно, и совпадает с тем, что видит работающая система: она
относит учащегося к группе по тому, что уже наблюдала, и принимает решения по
тому, что будет дальше.

Для сравнения тот же расчёт делается и со старым определением — доля верных по
всем тестовым ответам, включая те, на которых меряется ошибка. Разность двух
величин показывает, сколько разрыва создаёт сам отбор.

Оси, не производные от исхода — длина последовательности и среднее время ответа —
берутся из таблиц признаков. Они описывают не правильность ответов, а
объём и темп работы, поэтому отбором по исходу не являются; ошибка при этом всё равно считается на оценочной части, чтобы все оси измерялись на одних строках.

Выходные файлы:
  * `c4_subgroup_ece.csv`       — строка на ячейку подгруппы;
  * `c4_subgroup_disparity.csv` — разрыв между крайними группами с интервалом,
    мерой неравномерности и пометкой, отобрана ли группа по исходу.
"""
from __future__ import annotations

import argparse
import csv
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"sklearn\.linear_model\._linear_loss")

from ktx import paths
from ktx.stats import holm_bonferroni
from scripts.c4_significance import (
    CLASSICAL,
    DATASETS,
    DEEP,
    FOLDS,
    METHODS,
    calibrated_probabilities,
    load_cell,
)
from ktx.stats_fast import ece_from_totals, student_bin_stats

OUT = paths.ARTIFACTS_DIR / "cross_dataset"
DUMP_AXES = {"history": "history_q", "rt": "rt_q"}
PREFIXES = (5, 10, 20)          # длины окна признака
MIN_EVAL_ROWS = 5               # меньше — учащийся не оценивается
MIN_ROWS = 100                  # меньше — ячейка подгруппы не отчитывается
MIN_STUDENTS = 10


def _subgroup_table(dataset: str, fold: int):
    p = OUT / f"subgroups_{dataset}_fold{fold}.csv"
    return pd.read_csv(p) if p.exists() else None


def _blocks(groups: np.ndarray):
    """Границы блоков подряд идущих строк одного учащегося.

    Порядок строк — тот, которым идёт оценка в библиотеке, то есть внутри
    учащегося он хронологический. Непрерывность блоков проверяется отдельным
    гейтом, поэтому здесь она принимается как данность.
    """
    codes, first = np.unique(groups, return_index=True)
    order = np.argsort(first)
    codes = codes[order]
    starts = np.sort(first)
    ends = np.append(starts[1:], groups.size)
    return codes, starts, ends


def _quartiles(values: np.ndarray, n_bins: int = 4, min_per_bin: int = 30):
    """Квантильные группы; при малом числе учащихся — половины."""
    ok = np.isfinite(values)
    n = int(ok.sum())
    if n < 2 * min_per_bin:
        return np.full(values.size, -1, dtype=int)
    bins = n_bins if n >= n_bins * min_per_bin else 2
    q = np.full(values.size, -1, dtype=int)
    edges = np.quantile(values[ok], np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    q[ok] = np.clip(np.searchsorted(edges, values[ok], side="right") - 1, 0, bins - 1)
    return q


def _ece(cnt, sy, sp, mask, mult=None):
    if mult is None:
        c, y, p = cnt[mask].sum(0), sy[mask].sum(0), sp[mask].sum(0)
    else:
        w = mult[mask]
        c, y, p = w @ cnt[mask], w @ sy[mask], w @ sp[mask]
    if c.sum() <= 0:
        return float("nan"), 0.0
    return float(ece_from_totals(c, y, p)), float(c.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    # Десять тысяч, а не две: при двух тысячах наименьшее достижимое
    # p-значение равно 2/2001, и после поправки Холма на семью из
    # пятидесяти двух сравнений оно становится 0.052 — вся семья
    # оказывается непроходимой независимо от величины различий.
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--prefixes", nargs="+", type=int, default=list(PREFIXES))
    args = ap.parse_args()

    ece_rows, disp_rows, skipped = [], [], []

    for dataset in DATASETS:
        for fold in args.folds:
            dump = _subgroup_table(dataset, fold)
            for granularity in ("concept", "question"):
                fams = (("deep", DEEP),) if granularity == "concept" else \
                       (("deep", DEEP), ("classical", CLASSICAL))
                for family, models in fams:
                    for model in models:
                        loaded, why = load_cell(dataset, family, model, fold,
                                                granularity)
                        if loaded is None:
                            skipped.append((dataset, granularity, family, model, why))
                            continue
                        ty, tp, vy, vp, groups = loaded
                        probs = calibrated_probabilities(tp, vy, vp)
                        codes, starts, ends = _blocks(np.asarray(groups))
                        lengths = ends - starts

                        for K in args.prefixes:
                            # окно признака и окно оценки не пересекаются
                            keep = lengths >= K + MIN_EVAL_ROWS
                            if keep.sum() < 2 * MIN_STUDENTS:
                                continue
                            eval_mask = np.zeros(ty.size, dtype=bool)
                            prefix_acc = np.full(codes.size, np.nan)
                            for i in range(codes.size):
                                if not keep[i]:
                                    continue
                                s0, s1 = starts[i], ends[i]
                                prefix_acc[i] = ty[s0:s0 + K].mean()
                                eval_mask[s0 + K:s1] = True
                            q_prefix = _quartiles(prefix_acc)
                            q_prefix[~keep] = -1

                            # то же со старым определением: доля верных по всем
                            # строкам учащегося, включая оценочные
                            all_acc = np.array([ty[s:e].mean() for s, e in zip(starts, ends)])
                            all_acc[~keep] = np.nan
                            q_all = _quartiles(all_acc)
                            q_all[~keep] = -1

                            for axis, q in (("ability_prefix", q_prefix),
                                            ("ability_all_rows", q_all)):
                                if K != 10 and axis == "ability_all_rows":
                                    continue        # старое определение — один раз
                                label = f"{axis}{K}" if axis == "ability_prefix" else axis
                                _one_axis(ece_rows, disp_rows, dataset, granularity,
                                          family, model, fold, probs, ty, groups,
                                          codes, starts, ends, eval_mask, q, label,
                                          axis == "ability_all_rows", args)

                        # оси, не производные от исхода: берутся из таблицы
                        # признаков, ошибка считается на той же оценочной части
                        if dump is not None:
                            keep10 = lengths >= 10 + MIN_EVAL_ROWS
                            eval10 = np.zeros(ty.size, dtype=bool)
                            for i in range(codes.size):
                                if keep10[i]:
                                    eval10[starts[i] + 10:ends[i]] = True
                            m = {}
                            for axis, col in DUMP_AXES.items():
                                if col not in dump.columns:
                                    continue
                                m = dict(zip(dump["uid"].to_numpy(),
                                             dump[col].to_numpy()))
                                q = np.array([int(m.get(int(c), -1)) for c in codes])
                                q[~keep10] = -1
                                _one_axis(ece_rows, disp_rows, dataset, granularity,
                                          family, model, fold, probs, ty, groups,
                                          codes, starts, ends, eval10, q, axis,
                                          False, args)
                        print(f"[{dataset:20s} {granularity:8s} {family:9s} "
                              f"{model:9s} f{fold}] строк {len(ece_rows)}")

    by = defaultdict(list)
    for i, r in enumerate(disp_rows):
        by[r["family_key"]].append(i)
    for key, idxs in by.items():
        res = holm_bonferroni([disp_rows[i]["p"] for i in idxs], alpha=args.alpha)
        for j, i in enumerate(idxs):
            disp_rows[i]["p_holm"] = res["adjusted"][j]
            disp_rows[i]["reject_holm"] = bool(res["reject"][j])
            disp_rows[i]["family_size"] = len(idxs)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("c4_subgroup_ece.csv", ece_rows),
                       ("c4_subgroup_disparity.csv", disp_rows)):
        if not rows:
            continue
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"записано {len(rows)} строк -> {OUT / name}")
    if skipped:
        agg = defaultdict(int)
        for ds, g, fam, mdl, why in skipped:
            agg[(ds, g, why)] += 1
        print(f"\nпропущено {len(skipped)} ячеек:")
        for (ds, g, why), n in sorted(agg.items()):
            print(f"  {ds:20s} {g:8s} x{n:2d}  {why}")


def _one_axis(ece_rows, disp_rows, dataset, granularity, family, model, fold,
              probs, ty, groups, codes, starts, ends, eval_mask, q, axis,
              selected_on_outcome, args):
    """Ошибка по группам одной оси и разрыв между крайними группами."""
    present = sorted(set(q.tolist()) - {-1})
    if len(present) < 2:
        return
    idx = np.flatnonzero(eval_mask)
    if idx.size == 0:
        return
    ty_e = ty[idx]
    groups_e = np.asarray(groups)[idx]
    for method in METHODS:
        cnt, sy, sp, codes_e = student_bin_stats(ty_e, probs[method][idx], groups_e)
        pos = {int(c): i for i, c in enumerate(codes_e)}
        qe = np.full(codes_e.size, -1, dtype=int)
        for i, c in enumerate(codes):
            j = pos.get(int(c))
            if j is not None:
                qe[j] = q[i]
        overall, n_all = _ece(cnt, sy, sp, np.ones(codes_e.size, dtype=bool))
        per_q = {}
        for qi in present:
            mask = qe == qi
            e, n = _ece(cnt, sy, sp, mask)
            if n < MIN_ROWS or int(mask.sum()) < MIN_STUDENTS:
                continue
            per_q[qi] = (e, n, int(mask.sum()))
            ece_rows.append({
                "dataset": dataset, "granularity": granularity, "family": family,
                "model": model, "fold": fold, "method": method, "axis": axis,
                "quartile": qi, "ece": e, "n_rows": int(n),
                "n_students": int(mask.sum()), "ece_overall": overall,
                "selected_on_outcome": selected_on_outcome,
            })
        if len(per_q) < 2:
            continue
        lo_q, hi_q = min(per_q), max(per_q)
        obs = per_q[lo_q][0] - per_q[hi_q][0]
        eq_ce = max(abs(v[0] - overall) for v in per_q.values())
        rng = np.random.default_rng(fold)
        mlo, mhi = qe == lo_q, qe == hi_q
        clo, ylo, plo = cnt[mlo], sy[mlo], sp[mlo]
        chi, yhi, phi = cnt[mhi], sy[mhi], sp[mhi]
        ilo, ihi = np.flatnonzero(mlo), np.flatnonzero(mhi)
        S = codes_e.size
        diffs = np.empty(args.n_boot)
        done = 0
        for _ in range(args.n_boot):
            pick = rng.integers(0, S, size=S)
            mult = np.bincount(pick, minlength=S).astype(float)
            wlo, whi = mult[ilo], mult[ihi]
            if wlo.sum() == 0 or whi.sum() == 0:
                continue
            a = ece_from_totals(wlo @ clo, wlo @ ylo, wlo @ plo)
            b = ece_from_totals(whi @ chi, whi @ yhi, whi @ phi)
            if not np.isfinite(a) or not np.isfinite(b):
                continue
            diffs[done] = float(a) - float(b)
            done += 1
        diffs = diffs[:done]
        ci_lo = float(np.quantile(diffs, args.alpha / 2))
        ci_hi = float(np.quantile(diffs, 1 - args.alpha / 2))
        p = float(min(1.0, 2.0 * (min(int(np.sum(diffs <= 0)),
                                      int(np.sum(diffs >= 0))) + 1) / (diffs.size + 1)))
        disp_rows.append({
            "dataset": dataset, "granularity": granularity, "family": family,
            "model": model, "fold": fold, "method": method, "axis": axis,
            "q_low": lo_q, "q_high": hi_q,
            "ece_low": per_q[lo_q][0], "ece_high": per_q[hi_q][0],
            "ece_overall": overall,
            "ratio_low_over_high": (per_q[lo_q][0] / per_q[hi_q][0]
                                    if per_q[hi_q][0] > 0 else float("nan")),
            "ratio_low_over_overall": (per_q[lo_q][0] / overall if overall > 0
                                       else float("nan")),
            "delta_ece": obs, "ci_low": ci_lo, "ci_high": ci_hi, "p": p,
            "equalized_ce": eq_ce,
            "n_students_low": per_q[lo_q][2], "n_students_high": per_q[hi_q][2],
            "n_eval_rows": int(n_all),
            "selected_on_outcome": selected_on_outcome,
            "family_key": f"{granularity}|{axis}|{method}|f{fold}",
        })


if __name__ == "__main__":
    main()
