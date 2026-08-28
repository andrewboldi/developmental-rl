"""Statistical tools for multi-seed RL comparisons."""

import numpy as np
from scipy.stats import mannwhitneyu


def iqm(values):
    """Interquartile mean: mean of the middle 50% of sorted values."""
    v = np.sort(np.asarray(values, dtype=float))
    k = len(v) // 4
    return float(np.mean(v[k:len(v) - k])) if len(v) > 3 else float(np.mean(v))


def bootstrap_ci(values, n_boot=10000, ci=95, rng=None, statistic=np.mean):
    """Percentile bootstrap CI for a statistic (default: the mean)."""
    rng = rng or np.random.default_rng()
    v = np.asarray(values, dtype=float)
    stats = np.array([statistic(rng.choice(v, size=len(v), replace=True))
                      for _ in range(n_boot)])
    alpha = (100 - ci) / 2
    return float(np.percentile(stats, alpha)), float(np.percentile(stats, 100 - alpha))


def mann_whitney(a, b):
    """Two-sided Mann-Whitney U test."""
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return {"u": float(u), "p": float(p)}


def time_to_threshold(steps, vals, threshold):
    """First step at which vals reaches threshold; None if censored (never)."""
    for s, v in zip(steps, vals):
        if v >= threshold:
            return s
    return None
