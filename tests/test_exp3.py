import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from devrl.agents.linearq import LinearQ
from devrl.envs.piano import PianoPiece
from experiments.exp3_variation import (aggregate, greedy_score, holm,
                                        make_schedule, run_seed)


# ---------------------------------------------------------------- LinearQ

def make_agent(**kw):
    defaults = dict(n_features=3, n_actions=2, lr=0.5, gamma=0.5, eps=0.0,
                    rng=np.random.default_rng(0))
    defaults.update(kw)
    return LinearQ(**defaults)


def test_q_is_linear_in_features():
    ag = make_agent()
    assert np.allclose(ag.q(np.array([1.0, 1.0, 1.0])), [0.0, 0.0])  # zero init
    ag.W[1] = [1.0, 2.0, 0.0]
    assert np.allclose(ag.q(np.array([0.5, 1.0, 0.0])), [0.0, 2.5])


def test_terminal_update_is_hand_computed_sgd_step():
    ag = make_agent()  # lr = 0.5
    phi = np.array([2.0, 0.0, 1.0])
    ag.update(phi, 0, 1.0, None, done=True)
    # td = 1 - 0 = 1;  w_0 += 0.5 * 1 * phi  (gradient step, scaled by phi)
    assert np.allclose(ag.W[0], [1.0, 0.0, 0.5])
    assert np.allclose(ag.W[1], 0.0)  # only the taken action's row moves
    ag.update(phi, 0, 1.0, None, done=True)
    # q = 2*1 + 1*0.5 = 2.5; td = -1.5; w_0 += 0.5 * (-1.5) * phi
    assert np.allclose(ag.W[0], [1.0 - 1.5, 0.0, 0.5 - 0.75])


def test_bootstrap_update_uses_max_q_of_next_features():
    ag = make_agent(lr=0.1, gamma=0.5)
    ag.W[1] = [0.0, 0.0, 2.0]
    phi, phi2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    ag.update(phi, 0, 0.5, phi2, done=False)
    # target = 0.5 + 0.5 * max(0, 2) = 1.5; td = 1.5; w_0 += 0.1 * 1.5 * phi
    assert np.allclose(ag.W[0], [0.15, 0.0, 0.0])
    assert np.allclose(ag.W[1], [0.0, 0.0, 2.0])


def test_shared_feature_updates_leak_across_states():
    # The contextual-interference mechanism: two states share feature 0, so
    # an update in one changes the value of the other.
    ag = make_agent()
    ag.update(np.array([1.0, 1.0, 0.0]), 0, 1.0, None, done=True)
    assert ag.q(np.array([1.0, 0.0, 1.0]))[0] == pytest.approx(0.5)


def test_greedy_prefers_higher_q():
    ag = make_agent()
    ag.W[1] = [1.0, 0.0, 0.0]
    assert ag.greedy(np.array([1.0, 0.0, 0.0])) == 1


def test_greedy_breaks_ties_randomly():
    ag = make_agent(n_actions=3)
    acts = {ag.greedy(np.array([1.0, 0.0, 0.0])) for _ in range(100)}
    assert acts == {0, 1, 2}


def test_eps_one_explores_uniformly():
    ag = make_agent(eps=1.0)
    ag.W[0] = [100.0, 0.0, 0.0]
    acts = [ag.act(np.array([1.0, 0.0, 0.0])) for _ in range(500)]
    assert 0.4 < np.mean(acts) < 0.6


def test_learns_contextual_bandit():
    ag = make_agent(n_features=2, eps=0.3, lr=0.2, rng=np.random.default_rng(5))
    c0, c1 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    for _ in range(300):
        for phi, good in ((c0, 0), (c1, 1)):
            a = ag.act(phi)
            ag.update(phi, a, float(a == good), None, done=True)
    assert ag.greedy(c0) == 0 and ag.greedy(c1) == 1


# ---------------------------------------------------------------- schedules

def test_blocked_schedule_is_three_contiguous_phases():
    s = make_schedule("blocked", 90, np.random.default_rng(0))
    assert len(s) == 90
    assert list(s[:30]) == [0] * 30
    assert list(s[30:60]) == [1] * 30
    assert list(s[60:]) == [2] * 30


def test_blocked_schedule_requires_divisible_budget():
    with pytest.raises(ValueError):
        make_schedule("blocked", 91, np.random.default_rng(0))


def test_interleaved_schedule_uniform_and_seeded():
    s = make_schedule("interleaved", 900, np.random.default_rng(0))
    assert len(s) == 900 and set(s.tolist()) == {0, 1, 2}
    counts = np.bincount(s, minlength=3) / 900
    assert np.all(np.abs(counts - 1 / 3) < 0.06)  # statistical: ~4 sd
    s2 = make_schedule("interleaved", 900, np.random.default_rng(0))
    assert np.array_equal(s, s2)


# ---------------------------------------------------------------- eval

def test_greedy_score_perfect_weights_reach_one():
    env = PianoPiece(rng=np.random.default_rng(0))
    ag = LinearQ(env.n_features, env.n_actions, rng=np.random.default_rng(0))
    # position features alone can encode a single passage perfectly
    for i in range(12):
        ag.W[env.correct_key(0, i), 18 + 4 + i] = 10.0
    assert greedy_score(env, ag, 0, np.random.default_rng(1)) == 1.0


def test_greedy_score_untrained_agent_sits_at_chance():
    env = PianoPiece(rng=np.random.default_rng(0))
    ag = LinearQ(env.n_features, env.n_actions, rng=np.random.default_rng(0))
    scores = [greedy_score(env, ag, 0, np.random.default_rng(k))
              for k in range(60)]
    assert abs(np.mean(scores) - 1 / 8) < 0.04  # all-tie greedy = uniform


# ---------------------------------------------------------------- run_seed

def test_budgets_match_exactly_and_eval_is_free():
    res = run_seed(0, episodes=30, eval_every=10)
    b, i = res["blocked"], res["interleaved"]
    # eval rollouts happened at every checkpoint but consumed zero budget
    assert b["train_steps"] == i["train_steps"] == 30 * 12
    assert b["checkpoint_episodes"] == i["checkpoint_episodes"] == [0, 10, 20, 30]


def test_run_seed_is_deterministic():
    a = run_seed(3, episodes=30, eval_every=15)
    b = run_seed(3, episodes=30, eval_every=15)
    assert a == b  # pure-python payload, bitwise reproducible


def test_run_seed_metric_shapes_and_ranges():
    r = run_seed(0, episodes=30, eval_every=10)
    assert set(r["structure"]) >= {"motif_table", "passages", "exceptions",
                                   "correct_keys"}
    for cond in ("blocked", "interleaved"):
        c = r[cond]
        assert len(c["acquisition_curve"]) == 4
        assert len(c["eval_scores"]) == 4
        assert all(len(row) == 4 for row in c["eval_scores"])
        assert all(0.0 <= v <= 1.0 for row in c["eval_scores"] for v in row)
        assert set(c["retention"]) == {"A", "B", "C", "mean"}
        assert 0.0 <= c["transfer"] <= 1.0
        assert len(c["practice_curve"]) == 3
        assert set(c["final_rollouts"]) == {"A", "B", "C", "novel"}
        assert all(len(v) == 12 for v in c["final_rollouts"].values())


def test_acquisition_tracks_currently_practiced_passage():
    r = run_seed(1, episodes=30, eval_every=10)
    b = r["blocked"]  # phases: A = ep 0-9, B = 10-19, C = 20-29
    ev = b["eval_scores"]
    assert b["acquisition_curve"][1] == ev[1][0]  # after A phase -> score on A
    assert b["acquisition_curve"][2] == ev[2][1]  # after B phase -> score on B
    assert b["acquisition_curve"][3] == ev[3][2]  # after C phase -> score on C
    i = r["interleaved"]  # practices all three at once
    assert i["acquisition_curve"][2] == pytest.approx(
        np.mean(i["eval_scores"][2][:3]))


def test_retention_and_transfer_come_from_final_checkpoint():
    r = run_seed(2, episodes=30, eval_every=10)
    for cond in ("blocked", "interleaved"):
        c = r[cond]
        last = c["eval_scores"][-1]
        assert c["retention"]["A"] == last[0]
        assert c["retention"]["C"] == last[2]
        assert c["retention"]["mean"] == pytest.approx(np.mean(last[:3]))
        assert c["transfer"] == last[3]


# ---------------------------------------------------------------- stats/output

def test_holm_adjustment_hand_computed():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])  # capped at 1


def test_output_contract_keys_and_json_clean():
    results = [run_seed(s, episodes=12, eval_every=6) for s in range(2)]
    out = aggregate(results, {"n_boot": 200})
    for key in ("experiment", "hypothesis", "conditions", "curves", "tests",
                "conclusion", "viz"):
        assert key in out
    assert out["experiment"] == "exp3_variation"
    assert set(out["conditions"]) == {"blocked", "interleaved"}
    assert len(out["conditions"]["blocked"]["seeds"]) == 2
    s0 = out["conditions"]["interleaved"]["seeds"][0]
    for key in ("seed", "retention", "transfer", "acquisition_mean",
                "acquisition_curve", "eval_scores", "train_steps",
                "final_rollouts"):
        assert key in s0
    # three Holm-corrected primary comparisons
    prim = [t for t in out["tests"].values() if t["family"] == "primary"]
    assert len(prim) == 3
    assert all("p_holm" in t and "significant" in t for t in prim)
    assert isinstance(out["conclusion"]["supported"], bool)
    assert isinstance(out["conclusion"]["summary"], str)
    # curves ride the shared checkpoint grid
    ck = results[0]["blocked"]["checkpoint_episodes"]
    assert out["curves"]["episodes"] == ck
    assert out["curves"]["steps"] == [e * 12 for e in ck]
    for metric in ("acquisition", "retention_A", "retention_B", "retention_C",
                   "retention_mean", "transfer"):
        for cond in ("blocked", "interleaved"):
            series = out["curves"]["metrics"][metric][cond]
            assert len(series["iqm"]) == len(ck)
            assert len(series["lo"]) == len(series["hi"]) == len(ck)
    # viz has everything the website needs, and the whole payload is
    # pure-python and finite (json refuses NaN/inf and numpy types)
    viz = out["viz"]
    for key in ("example_structure", "retention_bars", "transfer_bars",
                "crossover", "curves", "practice_curves", "example_rollouts",
                "phase_boundaries_episodes", "per_seed_points"):
        assert key in viz
    assert len(viz["example_structure"]["passages"]) == 4
    assert len(viz["example_structure"]["motif_table"]) == 6
    json.dumps(out, allow_nan=False)
