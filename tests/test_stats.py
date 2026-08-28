import numpy as np
import pytest

from devrl.stats import bootstrap_ci, iqm, mann_whitney, time_to_threshold


def test_iqm_is_mean_of_middle_half():
    # middle 50% of [0,1,2,...,7] is [2,3,4,5]
    assert iqm(np.arange(8.0)) == pytest.approx(3.5)


def test_iqm_robust_to_outlier():
    vals = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0])
    assert iqm(vals) == pytest.approx(1.0)


def test_bootstrap_ci_contains_mean_and_orders():
    rng = np.random.default_rng(0)
    vals = rng.normal(5.0, 1.0, size=200)
    lo, hi = bootstrap_ci(vals, n_boot=2000, rng=rng)
    assert lo < np.mean(vals) < hi
    assert hi - lo < 0.6  # tight for n=200


def test_mann_whitney_detects_separation():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 4)
    b = a + 10.0
    res = mann_whitney(a, b)
    assert res["p"] < 1e-4


def test_mann_whitney_null_is_insignificant():
    rng = np.random.default_rng(1)
    a = rng.normal(size=30)
    b = rng.normal(size=30)
    assert mann_whitney(a, b)["p"] > 0.01


def test_time_to_threshold_first_crossing():
    steps = [100, 200, 300, 400]
    vals = [0.1, 0.5, 0.9, 0.8]
    assert time_to_threshold(steps, vals, 0.9) == 300


def test_time_to_threshold_censored_returns_none():
    assert time_to_threshold([100, 200], [0.1, 0.2], 0.9) is None
