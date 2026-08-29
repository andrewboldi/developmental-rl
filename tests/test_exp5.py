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
       "grow-jump", "grow-linear-walk", "grow-adaptive-walk")


# ------------------------------------------------------------------ schedules

def test_conditions_registry_is_the_seven_from_design_v2():
    # v2 amendment: the missing factorial cell grow-adaptive-walk completes
    # the growth x task-staging factorial.
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


def test_size_schedule_adaptive_conditions_are_event_driven():
    for cond in ("grow-adaptive", "grow-adaptive-walk"):
        with pytest.raises(ValueError):
            exp5.size_schedule(cond, 0, 120000)


def test_walk_start_balance_first_is_30pct_of_budget():
    B = 120000
    for cond in ("adult-walk", "grow-linear-walk", "grow-adaptive-walk"):
        assert exp5.walk_start(cond, B) == 0
    for cond in ("adult-balance-first", "grow-linear", "grow-adaptive",
                 "grow-jump"):
        assert exp5.walk_start(cond, B) == int(0.3 * B)


def test_physics_variant_registry():
    assert exp5.VARIANTS["default"] == (2, 0)
    assert exp5.VARIANTS["tau3"] == (3, 0)     # muscle torque ~ L^3
    assert exp5.VARIANTS["damp2"] == (2, 2)    # size-scaled damping b = s^2
    assert set(exp5.ROBUST_CONDITIONS) <= set(ALL)
    assert "grow-adaptive-walk" in exp5.ROBUST_CONDITIONS


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
                 "grow-linear-walk", "grow-adaptive-walk"):
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


def test_run_one_reports_time_on_target(micro_results):
    # guard against the standing-only local optimum: eval records on-target
    # fractions, including the lean-only fraction (target != 0), which a pure
    # stander cannot earn.
    for cond, res in micro_results.items():
        n = len(res["eval_steps"])
        assert len(res["eval_ontarget"]) == n
        assert len(res["eval_lean_ontarget"]) == n
        for x in res["eval_ontarget"] + res["eval_lean_ontarget"]:
            assert 0.0 <= x <= 1.0
        assert 0.0 <= res["final_ontarget"] <= 1.0
        assert 0.0 <= res["final_lean_ontarget"] <= 1.0
        if res["steps_to_competence"] is None:
            assert res["ontarget_at_competence"] == pytest.approx(
                res["eval_ontarget"][-1])
            assert res["lean_ontarget_at_competence"] == pytest.approx(
                res["eval_lean_ontarget"][-1])
        else:
            i = res["eval_steps"].index(res["steps_to_competence"])
            assert res["ontarget_at_competence"] == pytest.approx(
                res["eval_ontarget"][i])
            assert res["lean_ontarget_at_competence"] == pytest.approx(
                res["eval_lean_ontarget"][i])


def test_run_one_durable_competence_consistent(micro_results):
    for cond, res in micro_results.items():
        d = res["steps_to_durable"]
        if d is not None:
            assert res["steps_to_competence"] is not None
            assert d >= res["steps_to_competence"]
            assert d in res["eval_steps"]


def test_run_one_is_deterministic_given_seed():
    a = exp5.run_one(3, cond="grow-adaptive", **MICRO)
    b = exp5.run_one(3, cond="grow-adaptive", **MICRO)
    assert a["eval_returns"] == b["eval_returns"]
    assert a["total_damage"] == b["total_damage"]
    assert a["falls"] == b["falls"]


def test_run_one_variant_changes_training_physics(micro_results):
    base = micro_results["grow-linear"]
    tau3 = exp5.run_one(0, cond="grow-linear", variant="tau3", **MICRO)
    damp2 = exp5.run_one(0, cond="grow-linear", variant="damp2", **MICRO)
    assert base["variant"] == "default"
    assert tau3["variant"] == "tau3" and damp2["variant"] == "damp2"
    assert (tau3["eval_returns"] != base["eval_returns"]
            or tau3["falls"] != base["falls"])
    assert (damp2["eval_returns"] != base["eval_returns"]
            or damp2["falls"] != base["falls"])


def test_run_one_variant_deterministic():
    a = exp5.run_one(2, cond="grow-adaptive-walk", variant="damp2", **MICRO)
    b = exp5.run_one(2, cond="grow-adaptive-walk", variant="damp2", **MICRO)
    assert a["eval_returns"] == b["eval_returns"]
    assert a["falls"] == b["falls"]


def test_run_indexed_applies_seed_offset():
    # confirmatory protocol: seeds 100..(100+N-1), disjoint from tuning 0-9
    direct = exp5.run_one(101, cond="adult-walk", **MICRO)
    idx = exp5._run_indexed(1, cond="adult-walk", seed_offset=100, **MICRO)
    assert idx["seed"] == 101
    assert idx["eval_returns"] == direct["eval_returns"]
    assert idx["total_damage"] == direct["total_damage"]


def test_evaluate_is_deterministic_and_reports_tracking():
    Q = np.zeros((765, 5))
    r1 = exp5.evaluate(Q, seed=0, k=1, n_eval=2)
    r2 = exp5.evaluate(Q, seed=0, k=1, n_eval=2)
    assert r1 == r2
    assert np.isfinite(r1["ret"])
    assert 0.0 <= r1["ontarget"] <= 1.0
    assert 0.0 <= r1["lean_ontarget"] <= 1.0


# ---------------------------------------------------------------- statistics

def test_holm_bonferroni_adjustment_known_example():
    adj = exp5.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one_and_handles_single_p():
    assert exp5.holm([0.2]) == pytest.approx([0.2])
    assert exp5.holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])


def test_welch_matches_scipy_and_guards_degenerate_input():
    from scipy.stats import ttest_ind
    a = [1.0, 2.0, 3.0, 4.0]
    b = [3.0, 4.0, 5.0, 7.0]
    w = exp5.welch(a, b)
    t, p = ttest_ind(a, b, equal_var=False)
    assert w["t"] == pytest.approx(float(t))
    assert w["p"] == pytest.approx(float(p))
    assert exp5.welch([1.0, 1.0], [1.0, 1.0])["p"] == 1.0


def test_time_to_durable_requires_consecutive_checkpoints():
    steps = [400, 800, 1200, 1600]
    assert exp5.time_to_durable(steps, [600, 400, 600, 600], 500.0) == 1200
    assert exp5.time_to_durable(steps, [600, 600, 0, 0], 500.0) == 400
    assert exp5.time_to_durable(steps, [600, 400, 600, 400], 500.0) is None
    assert exp5.time_to_durable(steps, [0, 0, 0, 0], 500.0) is None


def test_parity_verdict_rules_never_accept_the_null():
    # diff = growth - adult (positive = slower); margin = +20% of adult IQM
    assert exp5.parity_verdict(1000, 99999, 15000, True) == "refuted"
    assert exp5.parity_verdict(1000, 9000, 15000, True) == "boundary"
    assert exp5.parity_verdict(-5000, 9000, 15000, False) == "supported"
    assert exp5.parity_verdict(-5000, 99999, 15000, False) == "null"


def test_directional_verdict_rules():
    assert exp5.directional_verdict(True, True, True) == "supported"
    assert exp5.directional_verdict(True, True, False) == "refuted"
    assert exp5.directional_verdict(True, False, True) == "boundary"
    assert exp5.directional_verdict(False, True, False) == "boundary"
    assert exp5.directional_verdict(False, False, True) == "null"
    assert exp5.directional_verdict(False, False, False) == "null"


def test_combine_verdicts():
    assert exp5.combine_verdicts(["supported", "supported"]) == "supported"
    assert exp5.combine_verdicts(["supported", "refuted"]) == "boundary"
    assert exp5.combine_verdicts(["refuted", "null"]) == "refuted"
    assert exp5.combine_verdicts(["boundary", "null"]) == "boundary"
    assert exp5.combine_verdicts(["null", "null"]) == "null"


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
    ontgt = [0.1 * (i + 1) for i in range(3)]
    lean = [0.05 * (i + 1) for i in range(3)]
    ci = 1 if crossed else 2
    return {
        "seed": seed, "cond": cond, "train_steps": budget,
        "variant": "default",
        "eval_steps": steps, "eval_returns": returns,
        "eval_ontarget": ontgt, "eval_lean_ontarget": lean,
        "damage_at_checkpoint": [total * f for f in (0.3, 0.7, 1.0)],
        "size_at_checkpoint": [0.5, 0.75, 1.0] if grow else [1.0] * 3,
        "steps_to_competence": 800 if crossed else None,
        "steps_to_durable": 800 if crossed else None,
        "censored": not crossed,
        "damage_at_competence": total * 0.7 if crossed else total,
        "ontarget_at_competence": ontgt[ci], "lean_ontarget_at_competence": lean[ci],
        "final_perf": returns[-1], "total_damage": total,
        "final_ontarget": ontgt[-1], "final_lean_ontarget": lean[-1],
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
    rob = {v: {cond: [_fake(s, cond) for s in range(2)]
               for cond in exp5.ROBUST_CONDITIONS}
           for v in ("tau3", "damp2")}
    return exp5.aggregate(results, budget=1200, n_boot=60, robustness=rob)


def test_aggregate_has_full_output_contract(fake_output):
    for key in ("experiment", "hypothesis", "config", "conditions", "curves",
                "tests", "conclusion", "viz", "robustness"):
        assert key in fake_output, key
    assert fake_output["experiment"] == "exp5_growing"


def test_aggregate_conditions_carry_per_seed_raw_metrics(fake_output):
    for cond in ALL:
        c = fake_output["conditions"][cond]
        for key in ("steps_to_competence", "damage_at_competence",
                    "final_perf", "total_damage", "n_falls", "censored",
                    "ontarget_at_competence", "lean_ontarget_at_competence",
                    "final_ontarget", "final_lean_ontarget",
                    "steps_to_durable"):
            assert len(c[key]) == 4, (cond, key)
        assert c["censored_frac"] == pytest.approx(0.5)
        assert c["durable_frac"] == pytest.approx(0.5)


def test_aggregate_curves_have_ci_bounds(fake_output):
    for cond in ALL:
        cv = fake_output["curves"][cond]
        assert cv["steps"] == [400, 800, 1200]
        assert len(cv["iqm"]) == len(cv["lo"]) == len(cv["hi"]) == 3
        assert all(l <= h for l, h in zip(cv["lo"], cv["hi"]))


def test_aggregate_tests_are_holm_corrected_within_family(fake_output):
    tests = fake_output["tests"]
    fams = {t["family"] for t in tests}
    assert fams == {"primary", "secondary"}
    assert sum(t["family"] == "primary" for t in tests) == 4
    assert sum(t["family"] == "secondary" for t in tests) == 8
    for t in tests:
        for key in ("name", "family", "metric", "a", "b", "iqm_a", "iqm_b",
                    "u", "p", "p_holm", "significant", "direction_ok",
                    "t_welch", "p_welch", "p_welch_holm", "significant_welch"):
            assert key in t, (t.get("name"), key)
        assert t["p_holm"] >= t["p"] - 1e-12
        assert t["p_welch_holm"] >= t["p_welch"] - 1e-12


def test_aggregate_primary_family_is_walk_direct_pairing(fake_output):
    prim = [t for t in fake_output["tests"] if t["family"] == "primary"]
    pairs = {(t["a"], t["b"], t["metric"]) for t in prim}
    assert pairs == {
        ("grow-linear-walk", "adult-walk", "damage_at_competence"),
        ("grow-adaptive-walk", "adult-walk", "damage_at_competence"),
        ("grow-linear-walk", "adult-walk", "steps_to_competence"),
        ("grow-adaptive-walk", "adult-walk", "steps_to_competence"),
    }


def test_aggregate_steps_tests_carry_equivalence_ci(fake_output):
    for t in fake_output["tests"]:
        if t["metric"] == "steps_to_competence":
            sd = t["steps_diff"]
            for key in ("iqm_diff", "lo", "hi", "margin", "margin_frac"):
                assert key in sd, (t["name"], key)
            assert sd["lo"] <= sd["hi"]
            assert sd["margin"] > 0
        else:
            assert "censoring_sensitivity" in t
            cs = t["censoring_sensitivity"]
            for key in ("drop_censored_p", "worst_case_p",
                        "n_censored_a", "n_censored_b"):
                assert key in cs, (t["name"], key)


def test_aggregate_conclusion_has_per_claim_verdicts(fake_output):
    c = fake_output["conclusion"]
    assert "supported" not in c          # boolean replaced by per-claim verdicts
    assert isinstance(c["summary"], str) and len(c["summary"]) > 40
    claims = c["claims"]
    assert [cl["key"] for cl in claims] == [
        "damage_at_competence", "steps_parity", "gradualism", "balance_first"]
    for cl in claims:
        assert cl["verdict"] in ("supported", "refuted", "null", "boundary")
        assert isinstance(cl["claim"], str) and len(cl["claim"]) > 10
        assert isinstance(cl["evidence"], str) and len(cl["evidence"]) > 10


def test_aggregate_robustness_block(fake_output):
    rob = fake_output["robustness"]
    assert set(rob) == {"tau3", "damp2"}
    for variant, blk in rob.items():
        assert blk["n_seeds"] == 2
        assert "adult-walk" in blk["conditions"]  # comparator from main arm
        for cond in exp5.ROBUST_CONDITIONS:
            assert cond in blk["conditions"]
            assert len(blk["conditions"][cond]["damage_at_competence"]) == 2
        assert len(blk["tests"]) == len(exp5.ROBUST_CONDITIONS)
        for t in blk["tests"]:
            for key in ("p", "p_holm", "p_welch", "p_welch_holm"):
                assert key in t
        assert len(blk["damage_ordering"]) == len(exp5.ROBUST_CONDITIONS) + 1
        assert "note" in blk


def test_aggregate_viz_block_is_rich(fake_output):
    viz = fake_output["viz"]
    for key in ("size_schedules", "fall_events", "theta_traces",
                "damage_vs_competence", "ontarget", "robustness", "meta"):
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
        ot = viz["ontarget"][cond]
        assert len(ot["lean_iqm"]) == len(ot["ontarget_iqm"]) == 3
    assert viz["meta"]["threshold"] == exp5.THRESHOLD
    assert set(viz["meta"]["conditions"]) == set(ALL)
    assert "-5" in viz["meta"]["reward"]         # fall penalty disclosed
    assert "damping" in viz["meta"]              # default damping documented
