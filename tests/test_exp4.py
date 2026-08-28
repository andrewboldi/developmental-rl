import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from devrl.agents.distill import (EpisodicMemory, apply_advice, extract_advice,
                                  halflife_schedule)
from devrl.envs.trapgrid import TrapGrid

_spec = importlib.util.spec_from_file_location(
    "exp4_generations",
    Path(__file__).resolve().parents[1] / "experiments" / "exp4_generations.py")
exp4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp4)

# ~75x reduced budget for fast structural tests; halflives scale with life
TINY = dict(exp4.FULL, life=200, halflife=66, slow_halflife=333, n_boot=100)


# ---------------------------------------------------------------- distill.py

def test_memory_keeps_top_k_by_return():
    m = EpisodicMemory(k=3)
    for ret in [1.0, 5.0, 3.0, 10.0, 2.0]:
        m.add(ret, [(0, 0)])
    assert [r for r, _ in m.episodes] == [10.0, 5.0, 3.0]
    assert m.best_return() == 10.0


def test_memory_tie_breaks_keep_earlier_episode():
    # ties keep the earlier episode: a trapped agent's tied-return memories
    # are its earliest, most exploratory meanders, not its distilled habit
    m = EpisodicMemory(k=1)
    m.add(10.0, [(0, 0), (1, 1), (2, 2)])
    m.add(10.0, [(3, 3), (4, 4)])  # same return, arrives later
    assert m.episodes[0][1] == [(0, 0), (1, 1), (2, 2)]


def test_memory_stores_sa_sequences_and_empty_best():
    m = EpisodicMemory()
    assert m.best_return() == 0.0
    m.add(0.3, [(7, 1), (8, 2)])
    assert m.episodes[0] == (0.3, [(7, 1), (8, 2)])


def test_extract_advice_dedups_preserving_best_first_order():
    m = EpisodicMemory(k=3)
    m.add(10.0, [(0, 1), (2, 3), (0, 1)])  # duplicate inside the episode
    m.add(5.0, [(2, 3), (4, 0), (5, 1)])  # duplicate across episodes
    assert extract_advice(m, cap=100) == [(0, 1), (2, 3), (4, 0), (5, 1)]


def test_extract_advice_caps_pairs_best_episode_first():
    m = EpisodicMemory(k=2)
    m.add(1.0, [(s, 1) for s in range(50)])
    m.add(9.0, [(s, 0) for s in range(50)])
    adv = extract_advice(m, cap=10)
    assert adv == [(s, 0) for s in range(10)]  # best episode survives the cap


def test_apply_advice_sets_only_advised_pairs():
    Q = np.zeros((6, 4))
    out = apply_advice(Q, [(1, 2), (3, 0)], value=5.0)
    assert out is Q and Q[1, 2] == 5.0 and Q[3, 0] == 5.0
    assert Q.sum() == 10.0


def test_halflife_schedule_matches_design_constants():
    lr = halflife_schedule(0.3, 5000)
    eps = halflife_schedule(0.4, 5000)
    assert lr(0) == 0.3 and eps(0) == 0.4
    assert lr(5000) == pytest.approx(0.15)
    assert eps(10000) == pytest.approx(0.1)
    assert lr(15000) == pytest.approx(0.0375)  # rigid by end of life


# ---------------------------------------------------------- experiment script

def test_full_config_pins_design_numbers():
    c = exp4.FULL
    assert c["gens"] == 5 and c["life"] == 15000
    assert c["gens"] * c["life"] == 75000  # matched total budget
    assert c["halflife"] == 5000 and c["slow_halflife"] == 25000
    assert c["lr0"] == 0.3 and c["eps0"] == 0.4
    assert c["gamma"] == 0.99 and c["cap"] == 120
    assert c["memory_k"] == 3 and c["advice_cap"] == 100
    assert c["advice_value"] == 5.0 and c["n_boot"] == 10000
    s = exp4.SMOKE
    assert s["life"] == 1500 and s["halflife"] == 500 and s["slow_halflife"] == 2500
    assert exp4.CONDITIONS == ["generational-distill", "weight-copy",
                               "one-long-life", "one-long-life-slow",
                               "no-inheritance"]


def test_advised_student_walks_straight_to_the_big_goal():
    # the core mechanism: advice pairs alone steer a fresh greedy policy home
    env = TrapGrid()
    pairs = [(env.state_of((5, 1)), TrapGrid.UP)]
    pairs += [(env.state_of((4, c)), TrapGrid.RIGHT) for c in range(1, 13)]
    pairs += [(env.state_of((4, 13)), TrapGrid.DOWN)]
    m = EpisodicMemory()
    m.add(10.0, pairs)
    Q = np.zeros((env.n_states, env.n_actions))
    apply_advice(Q, extract_advice(m), value=5.0)
    ret, big, traj = exp4.greedy_rollout(Q)
    assert big and ret == 10.0 and len(traj) == 15  # 14 steps, optimal


def test_greedy_rollout_times_out_on_blank_q():
    ret, big, traj = exp4.greedy_rollout(np.zeros((165, 4)))
    assert ret == 0.0 and not big and len(traj) == 121  # cap-120 truncation


def test_run_seed_budgets_matched_and_structured():
    r = exp4._run_seed(0, TINY)
    assert set(r) == set(exp4.CONDITIONS)
    for cond in exp4.CONDITIONS:
        res = r[cond]
        assert res["steps_trained"] == TINY["gens"] * TINY["life"]  # exact
        assert len(res["per_gen"]) == TINY["gens"]
        for g in res["per_gen"]:
            for key in ("greedy_return", "big_goal", "greedy_steps", "q_start",
                        "best_memory_return", "gap"):
                assert np.isfinite(g[key])
        assert res["final_traj"][0] == [5, 1]  # eval always starts at S
    advice = r["generational-distill"]["advice_by_gen"]
    assert len(advice) == TINY["gens"]
    for adv in advice:
        assert 0 < len(adv) <= TINY["advice_cap"]
        assert len(set(adv)) == len(adv)  # deduplicated
        assert all(0 <= s < 165 and 0 <= a < 4 for s, a in adv)
    assert "advice_by_gen" not in r["weight-copy"]


def test_run_seed_is_deterministic_per_seed():
    assert exp4._run_seed(3, TINY) == exp4._run_seed(3, TINY)
    assert exp4._run_seed(0, TINY) != exp4._run_seed(1, TINY)


def test_holm_bonferroni_adjustment():
    assert np.allclose(exp4._holm([0.01, 0.04, 0.03, 0.5]),
                       [0.04, 0.09, 0.09, 0.5])
    assert list(exp4._holm([0.9, 0.8])) == [1.0, 1.0]


def test_aggregate_meets_output_contract(tmp_path):
    results = [exp4._run_seed(s, TINY) for s in range(2)]
    out = exp4._aggregate(results, TINY, seeds=2, wall_s=0.0)
    for key in ("experiment", "hypothesis", "config", "conditions", "curves",
                "tests", "conclusion", "viz"):
        assert key in out
    assert out["experiment"] == "exp4_generations"
    # curves: checkpoint steps + IQM + CI bounds per condition
    gr = out["curves"]["greedy_return"]
    assert gr["steps"] == [200, 400, 600, 800, 1000]
    for cond in exp4.CONDITIONS:
        block = gr["conditions"][cond]
        for arr in (block["iqm"], block["ci_lo"], block["ci_hi"], block["mean"]):
            assert len(arr) == TINY["gens"] and np.all(np.isfinite(arr))
    # per-seed raw metrics present
    assert len(out["conditions"]["weight-copy"]["seeds"]) == 2
    assert out["conditions"]["weight-copy"]["seeds"][0]["seed"] == 0
    # tests: 4 primary comparisons vs distill, Holm-adjusted
    comps = out["tests"]["comparisons"]
    assert [c["b"] for c in comps] == exp4.CONDITIONS[1:]
    for c in comps:
        assert c["a"] == "generational-distill"
        assert c["p_holm"] >= c["p"] - 1e-12 and isinstance(c["significant"], bool)
    assert isinstance(out["conclusion"]["supported"], bool)
    assert out["conclusion"]["summary"]
    # viz: layout, curves, advice trajectories for a representative seed
    viz = out["viz"]
    grid = viz["grid"]
    assert grid["start"] == [5, 1] and grid["goal"] == [5, 13]
    assert len(grid["candies"]) == 3 and len(grid["map"]) == 11
    assert viz["representative_seed"] in (0, 1)
    assert len(viz["advice_by_generation"]) == TINY["gens"]
    for gen_adv in viz["advice_by_generation"]:
        for p in gen_adv["pairs"]:
            assert set(p) == {"s", "rc", "a", "action"}
    assert set(viz["final_greedy_paths"]) == set(exp4.CONDITIONS)
    # whole object is JSON-serializable via the shared encoder
    from devrl.run import save_json
    save_json(tmp_path / "exp4.json", out)
    loaded = json.loads((tmp_path / "exp4.json").read_text())
    assert loaded["viz"]["grid"]["goal"] == [5, 13]
