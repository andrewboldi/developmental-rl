import json

import numpy as np

from devrl.run import run_seeds, save_json


def _work(seed):
    rng = np.random.default_rng(seed)
    return {"seed": seed, "value": float(rng.random()), "arr": np.arange(3)}


def test_run_seeds_returns_one_result_per_seed_in_order():
    out = run_seeds(_work, n_seeds=6, n_jobs=3)
    assert [o["seed"] for o in out] == list(range(6))
    # different seeds -> different values
    assert len({o["value"] for o in out}) == 6


def test_run_seeds_is_reproducible():
    a = run_seeds(_work, n_seeds=4, n_jobs=2)
    b = run_seeds(_work, n_seeds=4, n_jobs=4)
    assert [x["value"] for x in a] == [y["value"] for y in b]


def test_save_json_handles_numpy(tmp_path):
    p = tmp_path / "out.json"
    save_json(p, {"a": np.float64(1.5), "b": np.arange(2), "c": [np.int64(3)]})
    loaded = json.loads(p.read_text())
    assert loaded == {"a": 1.5, "b": [0, 1], "c": [3]}
