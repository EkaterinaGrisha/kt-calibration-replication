"""The fast calibration bootstrap must agree with the reference implementation.

`stats_fast` exists only to make the five-fold bootstrap affordable. It earns
that place only if it returns what `stats.paired_bootstrap` returns, so the
agreement is asserted on real prediction files rather than on synthetic data:
the disagreement that motivated the module (bin edges) shows up only on
probabilities that land exactly on an edge.
"""
from __future__ import annotations

import numpy as np
import pytest

from ktx import paths
from ktx.calibration import CALIBRATORS
from ktx.stats import ece_metric, paired_bootstrap
from ktx.stats_fast import (
    ece_from_totals,
    paired_cluster_bootstrap_ece,
    student_bin_stats,
    unpaired_cluster_bootstrap_ece,
)

CELL = paths.ARTIFACTS_DIR / "predictions" / "assist2009" / "dkt_fold0.npz"


def _synthetic():
    """Стенд на случай, когда файлов предсказаний рядом нет.

    Он воспроизводит то, из-за чего наивная формула корзины расходится с
    библиотечной: часть вероятностей ложится ровно на границу корзины. Поэтому
    равенство двух реализаций проверяется и здесь, а не только на реальной
    ячейке, — иначе в чистом репозитории тест молча пропускался бы.
    """
    rng = np.random.default_rng(20260906)
    n_students, per = 300, 60
    groups = np.repeat(np.arange(n_students), per)
    skill = rng.normal(0, 0.8, n_students)[groups]
    p_raw = 1 / (1 + np.exp(-(skill + rng.normal(0, 0.5, groups.size))))
    edges = np.linspace(0, 1, 11)
    on_edge = rng.random(groups.size) < 0.02
    p_raw[on_edge] = rng.choice(edges[1:-1], size=int(on_edge.sum()))
    y = (rng.random(groups.size) < p_raw * 0.9 + 0.05).astype(int)
    half = groups.size // 2
    p_iso = CALIBRATORS["isotonic"]().fit(p_raw[:half], y[:half]).transform(p_raw)
    return y, p_raw, p_iso, groups


def _cell():
    if not CELL.exists():
        return _synthetic()
    d = np.load(CELL)
    y = d["y_true"].astype(int)
    p_none = d["y_prob"].astype(float)
    vy = d["valid_y_true_q_pykt"].astype(int)
    vp = d["valid_y_prob_q_pykt"].astype(float)
    p_iso = CALIBRATORS["isotonic"]().fit(vp, vy).transform(p_none)
    return y, p_none, p_iso, np.asarray(d["groups"])


def test_ece_from_totals_matches_netcal():
    y, p_none, p_iso, groups = _cell()
    for p in (p_none, p_iso):
        cnt, sy, sp, _ = student_bin_stats(y, p, groups)
        fast = float(ece_from_totals(cnt.sum(0), sy.sum(0), sp.sum(0)))
        assert fast == pytest.approx(ece_metric(y, p), abs=1e-12)


def test_paired_bootstrap_matches_reference():
    y, p_none, p_iso, groups = _cell()
    ref = paired_bootstrap(y, p_iso, p_none, ece_metric, n_boot=150,
                           groups=groups, seed=7, metric_name="ece")
    fast = paired_cluster_bootstrap_ece(y, p_iso, p_none, groups,
                                        n_boot=150, seed=7)
    assert fast.diff == pytest.approx(ref.diff, abs=1e-12)
    assert fast.ci_low == pytest.approx(ref.ci_low, abs=1e-12)
    assert fast.ci_high == pytest.approx(ref.ci_high, abs=1e-12)
    assert fast.p_value == pytest.approx(ref.p_value, abs=1e-12)


def test_unpaired_reduces_to_paired_when_sides_coincide():
    """Feeding the same rows to both sides of the unpaired routine must give
    the same observed difference as the paired one; only the interval may
    differ, since the two draws are independent."""
    y, p_none, p_iso, groups = _cell()
    up = unpaired_cluster_bootstrap_ece(y, p_iso, groups, y, p_none, groups,
                                        n_boot=100, seed=3)
    pa = paired_cluster_bootstrap_ece(y, p_iso, p_none, groups, n_boot=100, seed=3)
    assert up.diff == pytest.approx(pa.diff, abs=1e-12)
    assert up.n_students == pa.n_students
