"""Multi-seed experiment harness: parallel runs and JSON export."""

import json
import multiprocessing
import os
from pathlib import Path

import numpy as np


def run_seeds(fn, n_seeds, n_jobs=None):
    """Run fn(seed) for seed in 0..n_seeds-1 in parallel; results in seed order."""
    n_jobs = n_jobs or min(n_seeds, max(1, (os.cpu_count() or 2) - 2))
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(n_jobs) as pool:
        return pool.map(fn, range(n_seeds))


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return super().default(o)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, cls=_NumpyEncoder, indent=1))
