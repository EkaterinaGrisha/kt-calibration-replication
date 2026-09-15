"""Figures for the calibration paper. Numbers come from the artifacts.

Three figures, each 13 cm wide at 300 dpi, saved as PNG and PDF. Black and
white throughout: the journal accepts the manuscript in black and white only, so
series are told apart by marker, line style and hatch rather than by colour.

  1. What the mismatched fit costs, against the number of concepts per
     question. One point per dataset; the vertical axis is the ratio of the
     calibration error obtained when the correction is learned on one
     question-level object and applied to another, to the error obtained when
     both sides agree.
  2. Reliability, before and after the correction, on three datasets that span
     the range of concepts per question.
  3. Calibration error inside quartiles of student ability, one panel per
     dataset, corrected predictions only.

No number is written into this file; everything is read from
`artifacts/cross_dataset/`.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ktx import paths
from ktx.calibration import IsotonicCalibration
from ktx.stats_fast import bin_index

A = paths.ARTIFACTS_DIR / "cross_dataset"
FIG = paths.ROOT / "research" / "paper" / "figures" if hasattr(paths, "ROOT") else None
FIG = (paths.ARTIFACTS_DIR.parent / "paper" / "figures")
FIG.mkdir(parents=True, exist_ok=True)
CM = 1 / 2.54
WIDTH = 13 * CM
DS_RU = {"algebra2005": "Algebra-2005", "assist2009": "ASSISTments-2009",
         "assist2012": "ASSISTments-2012", "assist2015": "ASSISTments-2015",
         "assist2017": "ASSISTments-2017",
         "bridge2algebra2006": "Bridge-2006", "ednet": "EdNet-KT1"}
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "savefig.dpi": 300, "figure.dpi": 300})


def load(name):
    return list(csv.DictReader(open(A / name)))


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / (stem + '.png')} and .pdf")


# --------------------------------------------------------------------------- #
def kc_per_question() -> dict:
    """Компонентов знания на задание — из сводки величин, а где её нет, из
    файлов предсказаний. Сводка лежит рядом с таблицами, поэтому рисунок
    строится и там, где самих предсказаний нет."""
    j = A / "c4_numbers.json"
    if j.exists():
        d = json.loads(j.read_text()).get("kc_per_question", {})
        if d:
            return {k: float(v) for k, v in d.items()}
    out = {}
    for ds in DS_RU:
        p = paths.ARTIFACTS_DIR / "predictions" / ds / "dkt_fold0.npz"
        if p.exists():
            z = np.load(p)
            out[ds] = z["concept_y_true"].size / z["y_true"].size
    return out


def fig1_path_cost():
    path = load("c4_path_contrasts.csv")
    prof = kc_per_question()
    by = defaultdict(lambda: defaultdict(list))
    for r in path:
        if r["method"] != "isotonic":
            continue
        by[r["dataset"]]["m"].append(float(r["ece_mismatched_fit"]))
        by[r["dataset"]]["c"].append(float(r["ece_coherent_fit"]))
    fig, ax = plt.subplots(figsize=(WIDTH, 7.5 * CM))
    xs, ys, names = [], [], []
    for ds, v in by.items():
        xs.append(prof[ds])
        ys.append(float(np.mean(v["m"])) / float(np.mean(v["c"])))
        names.append(DS_RU[ds])
    o = np.argsort(xs)
    xs, ys, names = np.array(xs)[o], np.array(ys)[o], np.array(names)[o]
    ax.axhline(1.0, color="0.6", lw=0.7, ls="--", zorder=1)
    ax.plot(xs, ys, "o-", color="black", ms=5, lw=1.1, zorder=3)
    # Три набора, у которых на задание приходится один компонент знания, лежат
    # почти в одной точке: подписать их по отдельности нельзя, надписи налезут
    # друг на друга и на ось. Поэтому они получают одну общую подпись в пустом
    # правом нижнем углу и выноску к точке; остальные подписаны по месту.
    close = np.flatnonzero(xs < 1.01)
    for i in range(xs.size):
        if i in close:
            continue
        # ломаная идёт вверх направо, поэтому там, где за точкой есть следующая
        # и она выше, подпись уходит под линию, а не ложится на неё
        rising = i + 1 < xs.size and ys[i + 1] > ys[i]
        ax.annotate(names[i], (xs[i], ys[i]), textcoords="offset points",
                    xytext=(7, -9 if rising else -1), ha="left", va="center",
                    fontsize=7, color="0.15")
    if close.size:
        ax.annotate("\n".join(names[close]),
                    xy=(float(xs[close[0]]), float(ys[close[0]])),
                    xytext=(0.58, 0.015), textcoords="axes fraction",
                    fontsize=7, color="0.15", ha="left", va="bottom",
                    bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"),
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45",
                                    shrinkA=2, shrinkB=4))
    ax.set_xlabel("компонентов знания на задание / knowledge components per question")
    ax.set_ylabel("во сколько раз хуже / error ratio")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.set_yticklabels(["1", "2", "5", "10", "20"])
    ax.set_xlim(0.9, 2.95)
    ax.set_ylim(0.55, 26)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "c4_fig1_path_cost")


FIG2_PANELS = [("assist2012", "akt"), ("assist2009", "dkt"), ("ednet", "simplekt")]
FIG2_BINS = "c4_reliability_bins.csv"


def reliability_bins():
    """Точки диаграмм надёжности: по корзине на строку.

    Рисунок 2 — единственный, который строится по самим вероятностям, а не по
    сводкам, и в репозитории без файлов предсказаний он раньше просто
    пропускался: один рисунок из трёх рецензент воспроизвести не мог. Поэтому
    посчитанные корзины сохраняются рядом с остальными таблицами, и рисунок
    строится из них, когда предсказаний нет. Сами вероятности при этом не
    публикуются: в корзине остаются только среднее предсказание, наблюдённая
    доля и число строк.
    """
    out, missing = [], []
    for ds, mdl in FIG2_PANELS:
        p = paths.ARTIFACTS_DIR / "predictions" / ds / f"{mdl}_fold0.npz"
        if not p.exists():
            missing.append(f"{ds}/{mdl}")
            continue
        d = np.load(p)
        y = d["y_true"].astype(float)
        probs = d["y_prob"].astype(float)
        iso = IsotonicCalibration().fit(d["valid_y_prob_q_pykt"].astype(float),
                                        d["valid_y_true_q_pykt"].astype(int))
        for arm, pr in (("none", probs), ("isotonic", iso.transform(probs))):
            b = bin_index(pr)
            cnt = np.bincount(b, minlength=10).astype(float)
            sy = np.bincount(b, weights=y, minlength=10)
            sp = np.bincount(b, weights=pr, minlength=10)
            for k in range(10):
                if cnt[k] <= 0:
                    continue
                out.append({"dataset": ds, "model": mdl, "arm": arm, "bin": k,
                            "mean_predicted": sp[k] / cnt[k],
                            "observed_frequency": sy[k] / cnt[k],
                            "n_rows": int(cnt[k])})
    if out and not missing:
        with open(A / FIG2_BINS, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print(f"записано {len(out)} строк -> {A / FIG2_BINS}")
    return out, missing


def fig2_reliability():
    rows, missing = reliability_bins()
    if missing:
        rows = load(FIG2_BINS)
        if not rows:
            print("рисунок 2 пропущен: нет ни предсказаний "
                  + ", ".join(missing) + f", ни таблицы {FIG2_BINS}")
            return
        print(f"рисунок 2 строится из {FIG2_BINS}: предсказаний нет")
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["dataset"], r["model"])][r["arm"]].append(
            (float(r["mean_predicted"]), float(r["observed_frequency"])))
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 5.2 * CM), sharey=True)
    MODEL_RU = {"dkt": "DKT", "sakt": "SAKT", "akt": "AKT", "simplekt": "simpleKT"}
    for ax, key in zip(axes, FIG2_PANELS):
        ds, mdl = key
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.7, ls="--")
        for arm, marker, ls, label in (("none", "s", ":", "до / before"),
                                       ("isotonic", "o", "-", "после / after")):
            pts = sorted(by[key][arm])
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=marker, linestyle=ls, color="black", ms=3.6,
                    lw=1.0, markerfacecolor=("white" if marker == "s" else "black"),
                    markeredgewidth=0.8, label=label)
        ax.set_title(f"{DS_RU[ds]}, {MODEL_RU.get(mdl, mdl)}", fontsize=7.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("предсказанная вероятность / predicted probability")
    axes[0].set_ylabel("наблюдённая доля / observed frequency")
    # кривые идут по диагонали, поэтому правый нижний угол свободен, а в
    # левом верхнем подпись ложилась на «после»
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "c4_fig2_reliability")


def fig3_subgroups():
    rows = load("c4_subgroup_ece.csv")
    sel = [r for r in rows if r["axis"] == "ability_prefix10"
           and r["method"] == "isotonic"
           and r["granularity"] == "question" and r["family"] == "deep"]
    by = defaultdict(lambda: defaultdict(list))
    for r in sel:
        by[r["dataset"]][int(r["quartile"])].append(float(r["ece"]))
    datasets = [d for d in ["assist2012", "assist2017", "bridge2algebra2006",
                            "assist2009", "algebra2005", "ednet"] if d in by]
    fig, ax = plt.subplots(figsize=(WIDTH, 7.6 * CM))
    w = 0.19
    # four greys plus four hatches: the bars must stay distinguishable
    # in a black-and-white print and for a reader who cannot see colour
    colours = ["0.15", "0.45", "0.72", "1.0"]
    hatches = ["", "///", "...", "xxx"]
    for qi in range(4):
        vals = [float(np.mean(by[d][qi])) if by[d].get(qi) else np.nan
                for d in datasets]
        lab = {0: "1 — слабейшие / weakest",
               1: "2 — ниже среднего / lower middle",
               2: "3 — выше среднего / upper middle",
               3: "4 — сильнейшие / strongest"}[qi]
        ax.bar(np.arange(len(datasets)) + (qi - 1.5) * w, vals, width=w,
               color=colours[qi], hatch=hatches[qi], edgecolor="black",
               linewidth=0.5, label=f"группа {lab}")
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels([DS_RU[d] for d in datasets], rotation=20, ha="right",
                       fontsize=7)
    ax.set_ylabel("ошибка калибровки / calibration error")
    # легенда над полем: при четырёх группах она иначе накрывает самые высокие
    # столбцы, а именно они и есть предмет рисунка
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), handlelength=1.6, columnspacing=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "c4_fig3_subgroups")


if __name__ == "__main__":
    fig1_path_cost()
    fig2_reliability()
    fig3_subgroups()
