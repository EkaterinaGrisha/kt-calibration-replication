"""Exact but fast cluster bootstrap for the calibration error.

Why this module exists
----------------------
`stats.paired_bootstrap` recomputes the metric on a freshly gathered index
array for every resample. For the expected calibration error that is wasteful:
the metric depends on the resample only through per-bin totals, and the bin of
a row never changes, because the bin is a function of the predicted
probability alone. Accumulating (count, sum of labels, sum of probabilities)
per student and per bin once turns each resample into a sum of a few thousand
rows of ten numbers instead of half a million gathers. On the largest cell here
that is the difference between a run of hours and a run of seconds, which is
what makes a five-fold bootstrap affordable at all.

The result is not an approximation. The same random draw produces the same
resample as `stats.paired_bootstrap`, and the arithmetic differs only in the
order of summation; the accompanying test asserts agreement to 1e-12.

The binning convention follows `netcal.metrics.ECE`, which the rest of the
project uses: ten equal-width bins on [0, 1], a value falling on an internal
edge going to the upper bin. That is `searchsorted(..., side="right")`, not
`floor(p * n_bins)`: the latter disagrees on values whose product with ten is
representable just below an integer, and on real prediction arrays the two
differ in the fifth decimal of the calibration error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def bin_index(p: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Bin index per row, identical to netcal's equal-width binning."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.searchsorted(edges, np.asarray(p, dtype=np.float64),
                                   side="right") - 1, 0, n_bins - 1)


def ece_from_totals(cnt: np.ndarray, sy: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """Expected calibration error from per-bin totals.

    Accepts either one set of totals (shape (n_bins,)) or a stack of them
    (shape (..., n_bins)) and returns a scalar or an array accordingly.
    """
    cnt = np.asarray(cnt, dtype=np.float64)
    total = cnt.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.abs(np.where(cnt > 0, sy / cnt, 0.0) - np.where(cnt > 0, sp / cnt, 0.0))
    out = np.sum(np.where(cnt > 0, cnt, 0.0) * gap, axis=-1) / total[..., 0]
    return out


def student_bin_stats(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                      n_bins: int = 10):
    """Per-student, per-bin (count, sum of labels, sum of probabilities).

    Returns the three (n_students, n_bins) matrices and the student codes in
    the order the matrices use.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    codes, inv = np.unique(np.asarray(groups).ravel(), return_inverse=True)
    b = bin_index(p, n_bins)
    flat = inv * n_bins + b
    size = codes.size * n_bins
    cnt = np.bincount(flat, minlength=size).astype(np.float64)
    sy = np.bincount(flat, weights=y, minlength=size)
    sp = np.bincount(flat, weights=p, minlength=size)
    shape = (codes.size, n_bins)
    return cnt.reshape(shape), sy.reshape(shape), sp.reshape(shape), codes


@dataclass
class BootResult:
    ece_a: float
    ece_b: float
    diff: float           # a - b; negative means A has the smaller error
    ci_low: float
    ci_high: float
    p_value: float
    n_students: int
    n_rows: int
    n_boot_used: int


def paired_cluster_bootstrap_ece(y, prob_a, prob_b, groups, n_boot: int = 2000,
                                 alpha: float = 0.05, seed: int = 0,
                                 n_bins: int = 10) -> BootResult:
    """Cluster bootstrap over students for the difference of two calibration
    errors measured on the same rows.

    The two probability vectors are two post-hoc treatments of one model's
    output, so the resample is shared: the same students enter both sides.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    ca, ya_, pa_, codes = student_bin_stats(y, prob_a, groups, n_bins)
    cb, yb_, pb_, _ = student_bin_stats(y, prob_b, groups, n_bins)
    n_groups = codes.size

    obs_a = float(ece_from_totals(ca.sum(0), ya_.sum(0), pa_.sum(0)))
    obs_b = float(ece_from_totals(cb.sum(0), yb_.sum(0), pb_.sum(0)))

    # Same draw as stats.paired_bootstrap: one vector of student indices per
    # iteration, in the same order, so the two implementations resample
    # identically for a given seed.
    rng = np.random.default_rng(seed)
    label_total = y.sum()
    diffs = np.empty(n_boot, dtype=np.float64)
    per_student_pos = ya_.sum(1)          # labels are the same on both sides
    per_student_n = ca.sum(1)
    done = 0
    for _ in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        m = np.bincount(pick, minlength=n_groups).astype(np.float64)
        # degenerate resample guard, matching the reference implementation:
        # a resample with a single label class is skipped, not recorded.
        pos = float(m @ per_student_pos)
        tot = float(m @ per_student_n)
        if pos <= 0 or pos >= tot:
            continue
        ea = ece_from_totals(m @ ca, m @ ya_, m @ pa_)
        eb = ece_from_totals(m @ cb, m @ yb_, m @ pb_)
        diffs[done] = ea - eb
        done += 1
    diffs = diffs[:done]

    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    # Шаг сетки и пол значения — как в stats.paired_bootstrap: сдвиг на
    # единицу не даёт получить ноль там, где ноль означал бы уверенность,
    # которой в двух тысячах повторений нет.
    p = float(min(1.0, 2.0 * (min(int(np.sum(diffs <= 0)),
                                  int(np.sum(diffs >= 0))) + 1) / (diffs.size + 1)))
    del label_total
    return BootResult(obs_a, obs_b, obs_a - obs_b, lo, hi, p,
                      n_groups, int(y.size), done)


def unpaired_cluster_bootstrap_ece(y_a, prob_a, groups_a, y_b, prob_b, groups_b,
                                   n_boot: int = 2000, alpha: float = 0.05,
                                   seed: int = 0, n_bins: int = 10) -> BootResult:
    """Cluster bootstrap for two calibration errors measured on different rows.

    Used for contrasts between families: the deep models are evaluated on the
    rows pyKT keeps, the classical ones on their own rows, and the two sets
    differ by the first interaction of every student. Students shared by both
    sides are resampled jointly, and the observed difference is centred on that
    shared subsample so that the interval brackets the quantity it describes.
    """
    ca, ya_, pa_, codes_a = student_bin_stats(y_a, prob_a, groups_a, n_bins)
    cb, yb_, pb_, codes_b = student_bin_stats(y_b, prob_b, groups_b, n_bins)
    shared = np.intersect1d(codes_a, codes_b)
    if shared.size == 0:
        raise ValueError("no shared students between the two sides")
    ia = np.searchsorted(codes_a, shared)
    ib = np.searchsorted(codes_b, shared)
    ca, ya_, pa_ = ca[ia], ya_[ia], pa_[ia]
    cb, yb_, pb_ = cb[ib], yb_[ib], pb_[ib]
    n_units = shared.size

    obs_a = float(ece_from_totals(ca.sum(0), ya_.sum(0), pa_.sum(0)))
    obs_b = float(ece_from_totals(cb.sum(0), yb_.sum(0), pb_.sum(0)))

    rng = np.random.default_rng(seed)
    pos_a, n_a = ya_.sum(1), ca.sum(1)
    pos_b, n_b = yb_.sum(1), cb.sum(1)
    diffs = np.empty(n_boot, dtype=np.float64)
    done = 0
    for _ in range(n_boot):
        pick = rng.integers(0, n_units, size=n_units)
        m = np.bincount(pick, minlength=n_units).astype(np.float64)
        pa_pos, pa_tot = float(m @ pos_a), float(m @ n_a)
        pb_pos, pb_tot = float(m @ pos_b), float(m @ n_b)
        if pa_pos <= 0 or pa_pos >= pa_tot or pb_pos <= 0 or pb_pos >= pb_tot:
            continue
        diffs[done] = (ece_from_totals(m @ ca, m @ ya_, m @ pa_)
                       - ece_from_totals(m @ cb, m @ yb_, m @ pb_))
        done += 1
    diffs = diffs[:done]

    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    # Шаг сетки и пол значения — как в stats.paired_bootstrap: сдвиг на
    # единицу не даёт получить ноль там, где ноль означал бы уверенность,
    # которой в двух тысячах повторений нет.
    p = float(min(1.0, 2.0 * (min(int(np.sum(diffs <= 0)),
                                  int(np.sum(diffs >= 0))) + 1) / (diffs.size + 1)))
    return BootResult(obs_a, obs_b, obs_a - obs_b, lo, hi, p,
                      int(n_units), int(np.asarray(y_a).size + np.asarray(y_b).size), done)
