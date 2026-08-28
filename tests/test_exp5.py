import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    if "exp5_growing" in sys.modules:
        return sys.modules["exp5_growing"]
    spec = importlib.util.spec_from_file_location(
        "exp5_growing", ROOT / "experiments" / "exp5_growing.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp5_growing"] = mod
    spec.loader.exec_module(mod)
    return mod


exp5 = _load()

MICRO = dict(budget=1200, eval_every=400, n_eval=1)
ALL = ("adult-walk", "adult-balance-first", "grow-linear", "grow-adaptive",
       "grow-jump", "grow-linear-walk")


# ------------------------------------------------------------------ schedules

def test_conditions_registry_is_the_six_from_design():
    assert tuple(exp5.CONDITIONS) == ALL


def test_size_schedule_adult_conditions_always_one():
    B = 120000
    for cond in ("adult-walk", "adult-balance-first"):
        for t in (0, 1, 60000, B):
            assert exp5.size_schedule(cond, t, B) == 1.0


def test_size_schedule_grow_linear_ramps_over_first_60pct():
    B = 120000
    for cond in ("grow-linear", "grow-linear-walk"):
        assert exp5.size_schedule(cond, 0, B) == pytest.approx(0.5)
        assert exp5.size_schedule(cond, int(0.3 * B), B) == pytest.approx(0.75)
        assert exp5.size_schedule(cond, int(0.6 * B), B) == pytest.approx(1.0)
        assert exp5.size_schedule(cond, B, B) == pytest.approx(1.0)


def test_size_schedule_grow_jump_is_a_step_function():
    B = 120000
    assert exp5.size_schedule("grow-jump", 0, B) == 0.5
    assert exp5.size_schedule("grow-jump", int(0.6 * B) - 1, B) == 0.5
    assert exp5.size_schedule("grow-jump", int(0.6 * B), B) == 1.0
    assert exp5.size_schedule("grow-jump", B, B) == 1.0


def test_walk_start_balance_first_is_30pct_of_budget():
    B = 120000
    assert exp5.walk_start("adult-walk", B) == 0
    assert exp5.walk_start("grow-linear-walk", B) == 0
    for cond in ("adult-balance-first", "grow-linear", "grow-adaptive",
                 "grow-jump"):
        assert exp5.walk_start(cond, B) == int(0.3 * B)


# ------------------------------------------------------------------- run_one

@pytest.fixture(scope="module")
def micro_results():
    return {cond: exp5.run_one(0, cond=cond, **MICRO) for cond in ALL}


def test_run_one_training_budget_is_exact_for_every_condition(micro_results):
    for cond, res in micro_results.items():
        assert res["train_steps"] == MICRO["budget"], cond


def test_run_one_eval_checkpoints_identical_across_conditions(micro_results):
    for res in micro_results.values():
        assert res["eval_steps"] == [400, 800, 1200]


def test_run_one_curve_shapes_and_sanity(micro_results):
    for cond, res in micro_results.items():
        n = len(res["eval_steps"])
        assert len(res["eval_returns"]) == n
        assert len(res["damage_at_checkpoint"]) == n
        assert len(res["size_at_checkpoint"]) == n
        assert all(np.isfinite(res["eval_returns"]))
        d = res["damage_at_checkpoint"]
        assert all(d[i] <= d[i + 1] for i in range(n - 1))
        assert d[-1] == pytest.approx(res["total_damage"])


def test_run_one_fall_accounting(micro_results):
    for cond, res in micro_results.items():
        assert res["n_falls"] == len(res["falls"])
        assert res["total_damage"] == pytest.approx(
            sum(f[2] for f in res["falls"]))
        for step, size, damage in res["falls"]:
            assert 1 <= step <= MICRO["budget"]
            assert 0.5 <= size <= 1.0
            assert damage == pytest.approx(size ** 4)


def test_run_one_sizes_match_condition(micro_results):
    for cond in ("adult-walk", "adult-balance-first"):
        assert all(s == 1.0 for s in micro_results[cond]["size_at_checkpoint"])
    for cond in ("grow-linear", "grow-adaptive", "grow-jump",
                 "grow-linear-walk"):
        sizes = micro_results[cond]["size_at_checkpoint"]
        assert sizes[0] < 1.0
        assert all(0.5 <= s <= 1.0 for s in sizes)
        assert all(sizes[i] <= sizes[i + 1] for i in range(len(sizes) - 1))


def test_run_one_competence_fields_consistent(micro_results):
    for cond, res in micro_results.items():
        if res["steps_to_competence"] is None:
            assert res["censored"] is True
            assert res["damage_at_competence"] == pytest.approx(
                res["total_damage"])
        else:
            assert res["censored"] is False
            assert res["steps_to_competence"] in res["eval_steps"]


def test_run_one_trace_is_final_adult_greedy_policy(micro_results):
    for cond, res in micro_results.items():
        tr = res["trace"]
        n = len(tr["theta"])
        assert 1 <= n <= 200
        assert len(tr["target"]) == n and len(tr["action"]) == n
        assert isinstance(tr["fell"], bool)
        assert all(abs(th) <= np.pi / 5 + 0.5 for th in tr["theta"])
        assert set(tr["target"]) <= {-0.15, 0.0, 0.15}


def test_run_one_is_deterministic_given_seed():
    a = exp5.run_one(3, cond="grow-adaptive", **MICRO)
    b = exp5.run_one(3, cond="grow-adaptive", **MICRO)
    assert a["eval_returns"] == b["eval_returns"]
    assert a["total_damage"] == b["total_damage"]
    assert a["falls"] == b["falls"]


def test_evaluate_is_deterministic_and_checkpoint_seeded():
    Q = np.zeros((765, 5))
    r1 = exp5.evaluate(Q, seed=0, k=1, n_eval=2)
    r2 = exp5.evaluate(Q, seed=0, k=1, n_eval=2)
    assert r1 == r2
    assert np.isfinite(r1)


# ---------------------------------------------------------------- statistics

def test_holm_bonferroni_adjustment_known_example():
    adj = exp5.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one_and_handles_single_p():
    assert exp5.holm([0.2]) == pytest.approx([0.2])
    assert exp5.holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])


# ---------------------------------------------------------------- aggregation

def _fake(seed, cond, budget=1200):
    rng = np.random.default_rng([hashable_idx(cond), seed])
    steps = [400, 800, 1200]
    grow = cond.startswith("grow")
    returns = [50.0 * (i + 1) + 20 * rng.random() + (100 if grow else 0)
               for i in range(3)]
    falls = [[int(rng.integers(1, budget + 1)), 0.5 if grow else 1.0,
              0.5 ** 4 if grow else 1.0] for _ in range(200)]
    total = float(sum(f[2] for f in falls))
    crossed = seed % 2 == 0
    return {
        "seed": seed, "cond": cond, "train_steps": budget,
        "eval_steps": steps, "eval_returns": returns,
        "damage_at_checkpoint": [total * f for f in (0.3, 0.7, 1.0)],
        "size_at_checkpoint": [0.5, 0.75, 1.0] if grow else [1.0] * 3,
        "steps_to_competence": 800 if crossed else None,
        "censored": not crossed,
        "damage_at_competence": total * 0.7 if crossed else total,
        "final_perf": returns[-1], "total_damage": total,
        "n_falls": len(falls), "falls": falls,
        "size_dense": {"steps": steps, "s": [0.5, 0.75, 1.0]},
        "trace": {"theta": [0.0] * 200, "target": [0.0] * 200,
                  "action": [2] * 200, "fell": False},
    }


def hashable_idx(cond):
    return list(ALL).index(cond)


@pytest.fixture(scope="module")
def fake_output():
    results = {cond: [_fake(s, cond) for s in range(4)] for cond in ALL}
    return exp5.aggregate(results, budget=1200, n_boot=60)


def test_aggregate_has_full_output_contract(fake_output):
    for key in ("experiment", "hypothesis", "config", "conditions", "curves",
                "tests", "conclusion", "viz"):
        assert key in fake_output, key
    assert fake_output["experiment"] == "exp5_growing"


def test_aggregate_conditions_carry_per_seed_raw_metrics(fake_output):
    for cond in ALL:
        c = fake_output["conditions"][cond]
        for key in ("steps_to_competence", "damage_at_competence",
                    "final_perf", "total_damage", "n_falls", "censored"):
            assert len(c[key]) == 4, (cond, key)
        assert c["censored_frac"] == pytest.approx(0.5)


def test_aggregate_curves_have_ci_bounds(fake_output):
    for cond in ALL:
        cv = fake_output["curves"][cond]
        assert cv["steps"] == [400, 800, 1200]
        assert len(cv["iqm"]) == len(cv["lo"]) == len(cv["hi"]) == 3
        assert all(l <= h for l, h in zip(cv["lo"], cv["hi"]))


def test_aggregate_tests_are_holm_corrected(fake_output):
    tests = fake_output["tests"]
    assert len(tests) >= 6
    for t in tests:
        for key in ("name", "metric", "a", "b", "iqm_a", "iqm_b", "u", "p",
                    "p_holm", "significant", "direction_ok"):
            assert key in t, (t.get("name"), key)
        assert t["p_holm"] >= t["p"] - 1e-12


def test_aggregate_conclusion(fake_output):
    c = fake_output["conclusion"]
    assert isinstance(c["supported"], bool)
    assert isinstance(c["summary"], str) and len(c["summary"]) > 40


def test_aggregate_viz_block_is_rich(fake_output):
    viz = fake_output["viz"]
    for key in ("size_schedules", "fall_events", "theta_traces",
                "damage_vs_competence", "meta"):
        assert key in viz, key
    for cond in ALL:
        assert len(viz["fall_events"][cond]) <= 500  # subsampled (800 raw)
        assert len(viz["fall_events"][cond]) > 0
        sched = viz["size_schedules"][cond]
        assert len(sched["steps"]) == len(sched["iqm"]) == 3
        tr = viz["theta_traces"][cond]
        assert len(tr["theta"]) == 200 and "seed" in tr
        dvc = viz["damage_vs_competence"][cond]
        assert len(dvc["damage_iqm"]) == len(dvc["return_iqm"]) == 3
    assert viz["meta"]["threshold"] == exp5.THRESHOLD
    assert set(viz["meta"]["conditions"]) == set(ALL)
