"""Every table and every number the calibration paper quotes.

Nothing here is typed by hand: each table is printed from the artifacts, and
the scalar quantities the prose refers to (counts, ranges, extremes) are
written to `c4_numbers.json` so the number checker can compare the manuscript
against them instead of against a person's memory of them.

Usage:
    python -m scripts.make_c4_tables            # all tables
    python -m scripts.make_c4_tables --json     # only the scalars
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ktx import paths

A = paths.ARTIFACTS_DIR / "cross_dataset"
PRED = paths.ARTIFACTS_DIR / "predictions"
DEEP = ["dkt", "sakt", "akt", "simplekt"]
CLASSICAL = ["bkt", "pfa", "pfa_recency", "elorasch"]
METHODS = ["none", "platt", "isotonic", "temperature"]
RU = {"none": "без калибровки", "platt": "логистическая",
      "isotonic": "изотоническая", "temperature": "температурная"}
DS_RU = {"algebra2005": "Algebra-2005", "assist2009": "ASSISTments-2009",
         "assist2012": "ASSISTments-2012", "assist2015": "ASSISTments-2015",
         "assist2017": "ASSISTments-2017",
         "bridge2algebra2006": "Bridge-to-Algebra-2006", "ednet": "EdNet-KT1"}


def load(name):
    p = A / name
    if not p.exists():
        return []
    return list(csv.DictReader(open(p)))


def f(row, key):
    return float(row[key])


# --------------------------------------------------------------------------- #
def dataset_profile():
    """Rows per granularity, students, and concepts per question.

    Read from the prediction files when they are there, and from the previous
    `c4_numbers.json` when they are not. Without the fallback a run of this
    script in a repository that ships only the tables would replace a filled
    profile with an empty one, and the number checker — which reads the profile
    rather than the prediction files — would then fail on the very next command
    in the README.
    """
    out = {}
    for ds in DS_RU:
        p = PRED / ds / "dkt_fold0.npz"
        if not p.exists():
            continue
        d = np.load(p)
        n_q = int(d["y_true"].size)
        n_c = int(d["concept_y_true"].size)
        out[ds] = {
            "n_test_question": n_q,
            "n_test_concept": n_c,
            "kc_per_question": n_c / n_q,
            "n_students": int(np.unique(d["groups"]).size),
            "base_rate": float(np.mean(d["y_true"])),
            "has_question_level_valid": "valid_y_true_q_pykt" in d.files,
        }
    if not out:
        prev = A / "c4_numbers.json"
        if prev.exists():
            out = json.loads(prev.read_text()).get("dataset_profile", {})
    return out


def best_by(rows, keyfn, valfn=lambda r: float(r["ece"])):
    """Argmin of the value inside each group."""
    best = {}
    for r in rows:
        k = keyfn(r)
        if k not in best or valfn(r) < valfn(best[k]):
            best[k] = r
    return best


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    prof = dataset_profile()
    cells = load("c4_cell_ece.csv")
    meth = load("c4_method_contrasts.csv")
    models = load("c4_model_contrasts.csv")
    path = load("c4_path_contrasts.csv")
    sub = load("c4_subgroup_disparity.csv")
    ca = load("c4_concept_aware.csv")
    cac = load("c4_concept_aware_contrasts.csv")
    aggci = load("c4_aggregate_ci.csv")
    # интервал для того, что напечатано в строке таблицы, а не для ячейки:
    # ключ — (kind, набор, ключ строки, величина)
    CI = {(r["kind"], r["dataset"], r["key"], r["quantity"]): r for r in aggci}
    N: dict = {}

    def ci(kind, ds, key, quantity, fmt="{:.2f}"):
        """«точка [низ; верх]» или прочерк, если расчёт интервалов не проводился."""
        r = CI.get((kind, ds, key, quantity))
        if r is None:
            return "—"
        return (fmt.format(float(r["point"])) + " ["
                + fmt.format(float(r["ci_low"])) + "; "
                + fmt.format(float(r["ci_high"])) + "]")

    order = sorted(prof, key=lambda d: prof[d]["kc_per_question"])

    # ---------------------------------------------------------------- #
    print("\n### Таблица 1. Наборы данных\n")
    print("| набор | учащихся | строк на уровне заданий | строк на уровне компонентов | компонентов на задание | доля верных |")
    print("|---|---:|---:|---:|---:|---:|")
    for ds in order:
        p = prof[ds]
        print(f"| {DS_RU[ds]} | {p['n_students']} | {p['n_test_question']} | "
              f"{p['n_test_concept']} | {p['kc_per_question']:.4f} | {p['base_rate']:.3f} |")
    N["kc_per_question"] = {d: round(prof[d]["kc_per_question"], 4) for d in prof}
    N["n_students"] = {d: prof[d]["n_students"] for d in prof}
    # весь профиль — чтобы сверщик мог работать там, где файлов предсказаний нет
    N["dataset_profile"] = {d: {k: (round(v, 6) if isinstance(v, float) else v)
                                for k, v in prof[d].items()} for d in prof}
    N["datasets_without_question_level"] = [d for d in prof
                                            if not prof[d]["has_question_level_valid"]]

    # ---------------------------------------------------------------- #
    # 2. granularity
    print("\n### Таблица 2. Лучшая по калибровке нейросетевая модель на двух уровнях агрегирования\n")
    print("| набор | комп./задание | уровень компонентов | ECE | уровень заданий | ECE | ΔAUC |")
    print("|---|---:|---|---:|---|---:|---:|")
    deep_cells = [r for r in cells if r["family"] == "deep"]
    flips = []
    dauc = {}
    for ds in order:
        rows_c = [r for r in deep_cells if r["dataset"] == ds and r["granularity"] == "concept"]
        rows_q = [r for r in deep_cells if r["dataset"] == ds and r["granularity"] == "question"]
        if not rows_c:
            continue
        mean_c = defaultdict(list)
        for r in rows_c:
            mean_c[(r["model"], r["method"])].append(f(r, "ece"))
        agg_c = {k: float(np.mean(v)) for k, v in mean_c.items()}
        bc = min(agg_c, key=agg_c.get)
        if rows_q:
            mean_q = defaultdict(list)
            for r in rows_q:
                mean_q[(r["model"], r["method"])].append(f(r, "ece"))
            agg_q = {k: float(np.mean(v)) for k, v in mean_q.items()}
            bq = min(agg_q, key=agg_q.get)
            ac = np.mean([f(r, "auc") for r in rows_c if r["method"] == "none"])
            aq = np.mean([f(r, "auc") for r in rows_q if r["method"] == "none"])
            dauc[ds] = float(ac - aq)
            flips.append((ds, bc[0] != bq[0]))
            print(f"| {DS_RU[ds]} | {prof[ds]['kc_per_question']:.4f} | {bc[0]} + {RU[bc[1]]} | "
                  f"{agg_c[bc]:.5f} | {bq[0]} + {RU[bq[1]]} | {agg_q[bq]:.5f} | {dauc[ds]:+.4f} |")
        else:
            print(f"| {DS_RU[ds]} | {prof[ds]['kc_per_question']:.4f} | {bc[0]} + {RU[bc[1]]} | "
                  f"{agg_c[bc]:.5f} | — | — | — |")
    N["granularity_model_changes"] = sum(1 for _, x in flips if x)
    N["granularity_comparable_datasets"] = len(flips)
    N["auc_gap_concept_minus_question"] = {d: round(v, 4) for d, v in dauc.items()}

    # ---------------------------------------------------------------- #
    # 4. fit path
    if path:
        print("\n### Таблица 3. Согласованный и рассогласованный путь обучения калибровки\n")
        print("| набор | метод | ECE при рассогласовании | ECE при согласовании | "
              "отношение [95 % интервал] | учащихся | значимо (из 20 ячеек) |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for ds in order:
            for m in ("isotonic", "platt", "temperature"):
                s = [r for r in path if r["dataset"] == ds and r["method"] == m]
                if not s:
                    continue
                mm = float(np.mean([f(r, "ece_mismatched_fit") for r in s]))
                cc = float(np.mean([f(r, "ece_coherent_fit") for r in s]))
                sig = sum(1 for r in s if r["reject_holm"] == "True")
                n_st = int(float(s[0]["n_students"]))
                print(f"| {DS_RU[ds]} | {RU[m]} | {mm:.5f} | {cc:.5f} | "
                      f"{ci('path', ds, m, 'ratio')} | {n_st} | {sig}/{len(s)} |")
        iso_p = [r for r in path if r["method"] == "isotonic"]
        ratios = [f(r, "ratio") for r in iso_p]
        N["path_isotonic"] = {
            "cells": len(iso_p),
            "sig_worse_with_mismatch": sum(1 for r in iso_p
                                           if r["reject_holm"] == "True" and f(r, "delta_ece") > 0),
            "ratio_min": round(min(ratios), 2), "ratio_max": round(max(ratios), 2),
            "ratio_median": round(float(np.median(ratios)), 2),
        }
        # Отношение на набор — то же самое, что печатает таблица 3 и рисует
        # рисунок 1: отношение средних, а не среднее отношений. Две величины
        # расходятся вдвое на EdNet-KT1, и держать в сводке вторую значит
        # оставить читателю репозитория расхождение с таблицей.
        N["path_ratio_by_dataset"] = {}
        for ds in order:
            s = [r for r in iso_p if r["dataset"] == ds]
            if not s:
                continue
            cc = float(np.mean([f(r, "ece_coherent_fit") for r in s]))
            mm = float(np.mean([f(r, "ece_mismatched_fit") for r in s]))
            N["path_ratio_by_dataset"][ds] = round(mm / cc, 2) if cc else None
        if aggci:
            N["aggregate_ci_path"] = {
                r["dataset"] + "|" + r["key"]: {
                    "point": round(float(r["point"]), 5),
                    "ci_low": round(float(r["ci_low"]), 5),
                    "ci_high": round(float(r["ci_high"]), 5),
                    "n_students": int(float(r["n_students"])),
                    "n_cells": int(float(r["n_cells"])),
                }
                for r in aggci if r["kind"] == "path" and r["quantity"] == "ratio"}
            single = [r for r in aggci
                      if r["quantity"] in ("mean_mismatched", "mean_coherent", "ece")
                      and float(r["point"]) > 0]
            infl = [float(r["boot_mean"]) / float(r["point"]) for r in single]
            N["boot_fold_inflation"] = {
                "quantities": len(single),
                "max": round(max(infl), 2),
                "median": round(float(np.median(infl)), 2),
                "above_1_05": int(sum(1 for x in infl if x > 1.05)),
            }
            # У отношения сдвиг не сокращается нацело: сильнее завышается та
            # сторона, которая посчитана по меньшему числу учащихся или по
            # меньшей ошибке. У отношения путей это знаменатель, и отношение
            # в повторениях уходит вниз; у отношения «группа к общей» —
            # числитель, и оно уходит вверх. Обе крайности записаны, чтобы
            # утверждение в п. 5.3 можно было проверить, а не принять на слово.
            shift = {}
            for kind, name in (("path", "path_ratio"),
                               ("ability_group", "group_ratio")):
                s = [r for r in aggci if r["kind"] == kind
                     and r["quantity"] == "ratio" and float(r["point"]) > 0]
                if not s:
                    continue
                vals = [(float(r["boot_mean"]) / float(r["point"]),
                         r["dataset"]) for r in s]
                lo, hi = min(vals), max(vals)
                shift[name] = {"quantities": len(vals),
                               "min": round(lo[0], 2), "min_dataset": lo[1],
                               "max": round(hi[0], 2), "max_dataset": hi[1]}
            N["boot_ratio_shift"] = shift

    # ---------------------------------------------------------------- #
    # 3. isotonic vs raw
    print("\n### Таблица 4. Изотоническая калибровка против сырых вероятностей\n")
    iso = [r for r in meth if r["contrast"] == "iso_vs_none"]
    print("| срез | ячеек (набор × модель × разбиение) | поправка лучше | значимо лучше | значимо хуже |")
    print("|---|---:|---:|---:|---:|")
    for gran, fam in (("concept", "deep"), ("question", "deep"), ("question", "classical")):
        s = [r for r in iso if r["granularity"] == gran and r["family"] == fam]
        if not s:
            continue
        better = sum(1 for r in s if f(r, "delta_ece") < 0)
        sb = sum(1 for r in s if f(r, "delta_ece") < 0 and r["reject_holm"] == "True")
        sw = sum(1 for r in s if f(r, "delta_ece") > 0 and r["reject_holm"] == "True")
        label = {"concept": "уровень компонентов", "question": "уровень заданий"}[gran]
        famru = {"deep": "глубокие", "classical": "классические"}[fam]
        print(f"| {label}, {famru} | {len(s)} | {better} | {sb} | {sw} |")
        N[f"iso_vs_none_{gran}_{fam}"] = {"cells": len(s), "better": better,
                                          "sig_better": sb, "sig_worse": sw}
    N["iso_vs_none_sig_worse_total"] = sum(
        1 for r in iso if f(r, "delta_ece") > 0 and r["reject_holm"] == "True")
    N["iso_vs_none_cells_total"] = len(iso)

    # mean ECE before/after, and AUC preservation
    for gran, fam in (("question", "deep"), ("question", "classical"), ("concept", "deep")):
        s = [r for r in cells if r["granularity"] == gran and r["family"] == fam]
        if not s:
            continue
        by = defaultdict(dict)
        for r in s:
            by[(r["dataset"], r["model"], r["fold"])][r["method"]] = r
        none_ = [f(v["none"], "ece") for v in by.values()]
        iso_ = [f(v["isotonic"], "ece") for v in by.values()]
        dauc_ = [abs(f(v["isotonic"], "auc") - f(v["none"], "auc")) for v in by.values()]
        N[f"mean_ece_{gran}_{fam}"] = {"none": round(float(np.mean(none_)), 5),
                                       "isotonic": round(float(np.mean(iso_)), 5),
                                       "max_abs_delta_auc": round(float(np.max(dauc_)), 5)}

    # best method counts
    bm = Counter()
    for gran in ("concept", "question"):
        for fam in ("deep", "classical"):
            s = [r for r in cells if r["granularity"] == gran and r["family"] == fam]
            if not s:
                continue
            by = defaultdict(dict)
            for r in s:
                by[(r["dataset"], r["model"], r["fold"])][r["method"]] = f(r, "ece")
            for k, v in by.items():
                bm[(gran, fam, min(v, key=v.get))] += 1
    print("\n### Таблица 5. Лучший способ калибровки, число ячеек\n")
    print("| срез | " + " | ".join(RU[m] for m in METHODS) + " |")
    print("|---|" + "---:|" * len(METHODS))
    for gran, fam in (("concept", "deep"), ("question", "deep"), ("question", "classical")):
        row = [str(bm[(gran, fam, m)]) for m in METHODS]
        if sum(int(x) for x in row) == 0:
            continue
        label = {"concept": "уровень компонентов", "question": "уровень заданий"}[gran]
        famru = {"deep": "глубокие", "classical": "классические"}[fam]
        print(f"| {label}, {famru} | " + " | ".join(row) + " |")
        N[f"best_method_counts_{gran}_{fam}"] = {m: bm[(gran, fam, m)] for m in METHODS}

    # ---------------------------------------------------------------- #
    # 5. fragility
    print("\n### Таблица 6. Сравнение лучшей по калибровке модели со второй по величине ошибки\n")
    print("| уровень | сравнение | сравнений | значимо после Холма | наборов с одним победителем на всех 5 разбиениях |")
    print("|---|---|---:|---:|---:|")
    for gran, ct in (("concept", "within_deep"), ("question", "within_deep"),
                     ("question", "cross_family")):
        s = [r for r in models if r["granularity"] == gran and r["contrast_type"] == ct]
        if not s:
            continue
        sig = sum(1 for r in s if r["reject_holm"] == "True")
        win = defaultdict(set)
        for r in s:
            win[r["dataset"]].add(r["model_a"])
        stable = sum(1 for k, v in win.items() if len(v) == 1)
        ctru = {"within_deep": "между глубокими", "cross_family": "между семействами"}[ct]
        granru = {"concept": "компонентов", "question": "заданий"}[gran]
        print(f"| {granru} | {ctru} | {len(s)} | {sig} | {stable}/{len(win)} |")
        N[f"model_contrasts_{gran}_{ct}"] = {
            "contrasts": len(s), "sig_holm": sig,
            "datasets": len(win), "datasets_stable_winner": stable,
            "distinct_winners": {k: len(v) for k, v in sorted(win.items())},
            "p_min": round(min(f(r, "p") for r in s), 4),
            "p_max": round(max(f(r, "p") for r in s), 4)}
    cf = [r for r in models if r["contrast_type"] == "cross_family"]
    if cf:
        N["cross_family_classical_wins"] = sum(1 for r in cf if r["family_a"] == "classical")
        N["cross_family_total"] = len(cf)

    # winner table per fold
    print("\n### Таблица 7. Победитель по калибровке на каждом разбиении\n")
    print("| набор | " + " | ".join(f"разб. {k}" for k in range(5)) + " | различных |")
    print("|---|" + "---|" * 6)
    for ds in order:
        s = {r["fold"]: r["model_a"] for r in models
             if r["dataset"] == ds and r["granularity"] == "question"
             and r["contrast_type"] == "within_deep"}
        if not s:
            continue
        cells_ = [s.get(str(k), "—") for k in range(5)]
        print(f"| {DS_RU[ds]} | " + " | ".join(cells_) + f" | {len(set(cells_))} |")

    # how small is the gap that decides the winner, and how large is the
    # difference between the two inference passes over the same rows?
    gaps = {}
    for ds in order:
        rows_q = [r for r in cells if r["dataset"] == ds and r["family"] == "deep"
                  and r["granularity"] == "question"]
        if not rows_q:
            continue
        agg = defaultdict(list)
        for r in rows_q:
            agg[(r["model"], r["method"])].append(f(r, "ece"))
        per_model = {}
        for (mdl, meth), v in agg.items():
            m_ = float(np.mean(v))
            if mdl not in per_model or m_ < per_model[mdl]:
                per_model[mdl] = m_
        o = sorted(per_model.values())
        gaps[ds] = round(o[1] - o[0], 6)
    N["best_minus_runnerup_question_deep"] = gaps
    # Расхождение двух проходов вывода считается по самим предсказаниям; там, где
    # их нет, величина берётся из прошлой сводки, а не обнуляется молча.
    same = {}
    for ds in order:
        if prof[ds]["kc_per_question"] > 1.0001 or not prof[ds]["has_question_level_valid"]:
            continue
        worst = 0.0
        seen = False
        for mdl in DEEP:
            path_npz = PRED / ds / f"{mdl}_fold0.npz"
            if not path_npz.exists():
                continue
            dd = np.load(path_npz)
            if dd["concept_y_prob"].size != dd["y_prob"].size:
                continue
            seen = True
            worst = max(worst, float(np.max(np.abs(
                dd["concept_y_prob"].astype(np.float64) - dd["y_prob"].astype(np.float64)))))
        if seen:
            same[ds] = worst
    if same:
        N["two_passes_max_prob_gap_single_concept"] = {k: float(f"{v:.2e}")
                                                       for k, v in same.items()}
    else:
        prev = A / "c4_numbers.json"
        if prev.exists():
            keep = json.loads(prev.read_text()).get(
                "two_passes_max_prob_gap_single_concept")
            if keep:
                N["two_passes_max_prob_gap_single_concept"] = keep

    # ---------------------------------------------------------------- #
    # Панели рисунка 2 в числах. Подпись приглашает читателя сравнивать
    # расстояние до диагонали, значит эти расстояния должны быть посчитаны, а не
    # оставлены глазу: взвешенное по числу строк среднее модуля отклонения — это
    # та же ошибка калибровки, а наибольшее отклонение подпись называет прямо.
    bins = load("c4_reliability_bins.csv")
    if bins:
        panels = defaultdict(lambda: defaultdict(list))
        for r in bins:
            panels[f'{r["dataset"]}|{r["model"]}'][r["arm"]].append(
                (abs(f(r, "observed_frequency") - f(r, "mean_predicted")),
                 float(r["n_rows"])))
        N["fig2_panels"] = {
            key: {arm: {"weighted_abs_gap": round(
                            float(np.average([d for d, _ in v],
                                             weights=[n for _, n in v])), 5),
                        "max_abs_gap": round(max(d for d, _ in v), 4),
                        "bins": len(v)}
                  for arm, v in arms.items()}
            for key, arms in panels.items()}

    # ---------------------------------------------------------------- #
    # 6. subgroups
    if sub:
        print("\n### Таблица 8. Разница ошибки калибровки между крайними группами\n")
        print("| ось | ячеек | значимая разница (Холм) | доля | медиана |разницы| | наибольшее отношение |")
        print("|---|---:|---:|---:|---:|---:|")
        for axis in ("ability_prefix10", "ability_all_rows", "history", "rt"):
            s = [r for r in sub if r["axis"] == axis and r["method"] == "isotonic"]
            if not s:
                continue
            sig = sum(1 for r in s if r["reject_holm"] == "True")
            med = float(np.median([abs(f(r, "delta_ece")) for r in s]))
            ratios = [f(r, "ratio_low_over_high") for r in s
                      if np.isfinite(f(r, "ratio_low_over_high"))]
            axru = {"ability_prefix10": "способность по первым десяти ответам",
                    "ability_all_rows": "способность по всем ответам",
                    "history": "длина последовательности",
                    "rt": "время ответа"}[axis]
            print(f"| {axru} | {len(s)} | {sig} | {sig / len(s):.2f} | {med:.5f} | "
                  f"{max(ratios + [float('nan')]):.1f} |")
            N[f"subgroup_{axis}"] = {"cells": len(s), "sig": sig,
                                     "share_sig": round(sig / len(s), 3),
                                     "median_abs_delta": round(med, 5),
                                     "max_ratio": round(max(ratios), 2) if ratios else None}
        eq = defaultdict(list)
        for r in sub:
            if r["method"] == "isotonic" and r["axis"] == "ability_prefix10":
                eq[r["family"]].append(f(r, "equalized_ce"))
        N["equalized_ce_by_family"] = {k: round(float(np.mean(v)), 5) for k, v in eq.items()}

        # во сколько раз отбор по исходу завышает разрыв
        pair = defaultdict(dict)
        for r in sub:
            if r["method"] != "isotonic":
                continue
            k = (r["dataset"], r["granularity"], r["family"], r["model"], r["fold"])
            if r["axis"] in ("ability_prefix10", "ability_all_rows"):
                pair[k][r["axis"]] = abs(f(r, "delta_ece"))
        both = [(v["ability_all_rows"], v["ability_prefix10"]) for v in pair.values()
                if "ability_all_rows" in v and "ability_prefix10" in v]
        if both:
            a, b = zip(*both)
            N["selection_inflation"] = {
                "cells": len(both),
                "median_abs_delta_all_rows": round(float(np.median(a)), 5),
                "median_abs_delta_prefix": round(float(np.median(b)), 5),
                "ratio_of_medians": round(float(np.median(a)) / float(np.median(b)), 2),
                "larger_with_all_rows": int(sum(1 for x, y in both if x > y)),
            }

        # Таблица 9: ошибка по группам способности и её отношение к общей
        print("\n### Таблица 9. Ошибка калибровки в целом и по группам "
              "способности\n")
        sel = [r for r in load("c4_subgroup_ece.csv")
               if r["axis"] == "ability_prefix10" and r["method"] == "isotonic"
               and r["granularity"] == "question" and r["family"] == "deep"]
        by_ds = defaultdict(lambda: defaultdict(list))
        ov = defaultdict(list)
        for r in sel:
            by_ds[r["dataset"]][int(r["quartile"])].append(f(r, "ece"))
            ov[r["dataset"]].append(f(r, "ece_overall"))
        print("| набор | учащихся, всего / в группе 1 | в целом | группа 1 | "
              "группа 2 | группа 3 | группа 4 | отношение [95 % интервал] |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        t9 = {}
        n_by_ds = defaultdict(set)
        for r in sel:
            n_by_ds[r["dataset"]].add(int(r["n_students"]))
        for ds in order:
            if ds not in by_ds:
                continue
            o = float(np.mean(ov[ds]))
            g = [float(np.mean(by_ds[ds][q])) if by_ds[ds].get(q) else None
                 for q in range(4)]
            cells = " | ".join(f"{x:.5f}" if x is not None else "—" for x in g)
            row = CI.get(("ability_group", ds, "overall", "ece"))
            low = CI.get(("ability_group", ds, "group0", "ece"))
            n_st = int(float(row["n_students"])) if row else sum(n_by_ds[ds])
            # Интервал отношения построен по первой группе, а не по всей
            # оценочной части, и её размер объясняет его ширину. Столбец
            # называет оба числа, иначе третье правило протокола сообщения
            # выполняется только на словах.
            n_low = int(float(low["n_students"])) if low else None
            n_cell = f"{n_st} / {n_low}" if n_low else str(n_st)
            print(f"| {DS_RU[ds]} | {n_cell} | {o:.5f} | {cells} | "
                  f"{ci('ability_group', ds, 'ratio_low_over_overall', 'ratio', '{:.1f}')} |")
            t9[ds] = {"overall": round(o, 5),
                      "groups": [round(x, 5) if x is not None else None for x in g],
                      "ratio_low_over_overall": round(g[0] / o, 2),
                      "n_students_eval": n_st, "n_students_group1": n_low}
        N["table9_ability_groups"] = t9
        if aggci:
            N["aggregate_ci_ability"] = {
                r["dataset"] + "|" + r["key"]: {
                    "point": round(float(r["point"]), 5),
                    "ci_low": round(float(r["ci_low"]), 5),
                    "ci_high": round(float(r["ci_high"]), 5),
                    "boot_mean": round(float(r["boot_mean"]), 5),
                    "n_students": int(float(r["n_students"])),
                    "n_cells": int(float(r["n_cells"])),
                }
                for r in aggci if r["kind"] == "ability_group"}
        # does calibration reduce the disparity?
        pair = defaultdict(dict)
        for r in sub:
            pair[(r["dataset"], r["granularity"], r["family"], r["model"],
                  r["fold"], r["axis"])][r["method"]] = abs(f(r, "delta_ece"))
        both = [(v["none"], v["isotonic"]) for v in pair.values()
                if "none" in v and "isotonic" in v]
        if both:
            n_, i_ = zip(*both)
            N["disparity_none_vs_isotonic"] = {
                "cells": len(both),
                "mean_abs_delta_none": round(float(np.mean(n_)), 5),
                "mean_abs_delta_isotonic": round(float(np.mean(i_)), 5),
                "isotonic_reduces_in": int(sum(1 for a, b in both if b < a)),
            }

    # ---------------------------------------------------------------- #
    # 7. concept-aware
    if cac:
        print("\n### Таблица 10. Варианты калибровки против общей изотонической кривой\n")
        print("| вариант | ячеек | лучше общей | значимо лучше | значимо хуже |")
        print("|---|---:|---:|---:|---:|")
        for m in ("none", "concept_aware_lambda0", "concept_aware_eb", "difficulty_bucket"):
            s = [r for r in cac if r["method"] == m]
            if not s:
                continue
            better = sum(1 for r in s if f(r, "delta_ece") < 0)
            sb = sum(1 for r in s if f(r, "delta_ece") < 0 and r["reject_holm"] == "True")
            sw = sum(1 for r in s if f(r, "delta_ece") > 0 and r["reject_holm"] == "True")
            print(f"| {m} | {len(s)} | {better} | {sb} | {sw} |")
            N[f"concept_aware_{m}"] = {"cells": len(s), "better": better,
                                       "sig_better": sb, "sig_worse": sw}
        lam = Counter(r["best_lambda"] for r in ca)
        N["lambda_choices"] = dict(lam)

    out = A / "c4_numbers.json"
    out.write_text(json.dumps(N, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nwrote scalars -> {out}")


if __name__ == "__main__":
    main()
