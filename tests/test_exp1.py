"""EXP1 protocol tests: env statistics, budget accounting, eval purity,
analysis helpers, and the output contract of the blindfold experiment."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from devrl.agents.dyna import DynaQ
from devrl.agents.qlearning import QLearner
from devrl.agents.touchnav import TouchDynaQ, TouchNavigator
from devrl.envs.gridhome import HOME_A, HOME_B, GridHome
from devrl.run import save_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import exp1_blindfold as x1  # noqa: E402

OPEN = """
#######
#S....#
#.....#
#....G#
#######
"""

CORRIDOR = """
#########
#S.....G#
#########
"""


def tiny_config():
    cfg = x1.make_config(smoke=True)
    cfg.update(train_steps=600, eval_every=200, eval_episodes=4,
               blind_episodes=4, n_traj=2, n_boot=300)
    return cfg


# ---------------------------------------------------------------- env pins

def test_homes_share_shell_start_and_goal():
    # blindfold transfer needs state indices to line up between homes
    a = GridHome(HOME_A, rng=np.random.default_rng(0))
    b = GridHome(HOME_B, rng=np.random.default_rng(0))
    assert a.shape == b.shape == (11, 13)
    assert a.start == b.start and a.goal == b.goal
    assert a.state_of(a.start) == b.state_of(b.start)


def test_slip_statistics_match_spec():
    # slip=0.1: intended direction w.p. 0.9, each perpendicular w.p. 0.05,
    # opposite direction never
    env = GridHome(OPEN, slip=0.1, rng=np.random.default_rng(0))
    n = 4000
    counts = {(2, 4): 0, (1, 3): 0, (3, 3): 0, (2, 2): 0}
    for _ in range(n):
        env.pos = (2, 3)  # interior cell, all four neighbours open
        s2, _, _, _ = env.step(GridHome.RIGHT)
        counts[env.pos_of(s2)] += 1
    assert counts[(2, 2)] == 0  # never the opposite way
    assert abs(counts[(2, 4)] / n - 0.9) < 0.02
    assert abs((counts[(1, 3)] + counts[(3, 3)]) / n - 0.1) < 0.02


def test_empirical_bump_probability_matches_slip_mixture():
    # corridor cell (1,1), action UP: bump unless the 0.05 slip-to-RIGHT
    # fires, so P(bump) = 0.95; the touch model must recover it
    env = GridHome(CORRIDOR, slip=0.1, rng=np.random.default_rng(1))
    ag = TouchDynaQ(n_states=env.n_states, n_actions=4, planning_steps=0,
                    rng=np.random.default_rng(2))
    s = env.state_of((1, 1))
    n = 4000
    for _ in range(n):
        env.reset()
        _, _, _, info = env.step(GridHome.UP)
        ag.observe_touch(s, GridHome.UP, info["bump"])
    assert abs(ag.bump_prob(s, GridHome.UP) - 0.95) < 0.02


# ------------------------------------------------- training and eval purity

def test_training_budget_accounting_is_exact():
    cfg = tiny_config()
    agent, ckpts, curve = x1.train_agent("dynaq", cfg, seed=0)
    assert isinstance(agent, TouchDynaQ)
    assert agent.age == cfg["train_steps"]  # one update per env step, exactly
    assert ckpts == [200, 400, 600]
    assert len(curve) == 3 and all(0.0 <= v <= 1.0 for v in curve)
    q_agent, q_ckpts, _ = x1.train_agent("qlearning", cfg, seed=0)
    assert isinstance(q_agent, QLearner) and not isinstance(q_agent, DynaQ)
    assert q_agent.age == cfg["train_steps"]
    assert q_ckpts == ckpts  # identical eval schedule across conditions


def test_eval_is_deterministic_and_does_not_mutate_q():
    cfg = tiny_config()
    Q = np.zeros((11 * 13, 4))
    before = Q.copy()
    r1 = x1.eval_greedy(Q, HOME_A, cfg, seed=0, ckpt_idx=0)
    r2 = x1.eval_greedy(Q, HOME_A, cfg, seed=0, ckpt_idx=0)
    assert r1 == r2
    assert np.array_equal(Q, before)


def test_blindfold_phase_trains_nothing():
    cfg = tiny_config()
    agent, _, _ = x1.train_agent("dynaq", cfg, seed=0)
    q0, age0 = agent.Q.copy(), agent.age
    seen0 = len(agent.seen)
    rng_state0 = agent.rng.bit_generator.state
    for name in x1.COND_ORDER:
        x1.run_blind_condition(name, agent, cfg, seed=0)
    assert np.array_equal(agent.Q, q0)
    assert agent.age == age0 and len(agent.seen) == seen0
    assert agent.rng.bit_generator.state == rng_state0


def test_blind_touch_navigation_succeeds_in_trained_corridor():
    # after learning model+touch in a slip-noisy corridor, the blindfolded
    # navigator must still walk to the goal (the H1 mechanism, in miniature)
    env = GridHome(CORRIDOR, slip=0.1, rng=np.random.default_rng(3))
    # gamma=0.9 keeps a crisp Q margin between "step right" and "bump a
    # wall" in the corridor, so the test isolates the navigator mechanism
    ag = TouchDynaQ(n_states=env.n_states, n_actions=4, planning_steps=10,
                    lr=0.2, gamma=0.9, eps=0.3, rng=np.random.default_rng(4))
    s, ep_len = env.reset(), 0
    for _ in range(4000):
        a = ag.act(s)
        s2, r, done, info = env.step(a)
        ag.observe_touch(s, a, info["bump"])
        ag.update(s, a, r, s2, done)
        ep_len += 1
        if done or ep_len >= 20:
            s, ep_len = env.reset(), 0
        else:
            s = s2
    wins = 0
    for ep in range(20):
        env = GridHome(CORRIDOR, slip=0.1, rng=np.random.default_rng((5, ep)))
        nav = TouchNavigator(ag, ag.Q, env.reset(), touch=ag)
        for _ in range(20):
            a = nav.act()
            _, _, done, info = env.step(a)  # true state never shown to nav
            nav.advance(a, info["bump"])
            if done:
                wins += 1
                break
    assert wins >= 16


# ------------------------------------------------------------ stats helpers

def test_holm_adjustment_step_down():
    assert x1.holm([0.01, 0.02, 0.04]) == pytest.approx([0.03, 0.04, 0.04])
    assert x1.holm([0.4]) == pytest.approx([0.4])
    assert x1.holm([0.9, 0.8]) == [1.0, 1.0]


def test_censor_times_assigns_budget_plus_one():
    vals, frac = x1.censor_times([None, 500, None], budget=1000)
    assert vals == [1001, 500, 1001]
    assert frac == pytest.approx(2 / 3)


def test_safe_mann_whitney_handles_all_ties_without_nan():
    res = x1.safe_mann_whitney([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert res["p"] == 1.0 and not math.isnan(res["u"])
    res = x1.safe_mann_whitney([0.0, 0.1, 0.2], [5.0, 5.1, 5.2])
    assert res["p"] < 0.2  # normal path delegates to devrl.stats


def test_belief_entropy_bits():
    assert x1.belief_entropy_bits(np.array([0.25] * 4)) == pytest.approx(2.0)
    assert x1.belief_entropy_bits(np.array([0.0, 1.0])) == pytest.approx(0.0)


# ------------------------------------------------------------ configuration

def test_config_pins_design_constants():
    cfg = x1.make_config(smoke=False)
    assert cfg["train_steps"] == 40_000 and cfg["eval_every"] == 1000
    assert cfg["slip"] == 0.1 and cfg["gamma"] == 0.97 and cfg["cap"] == 60
    assert cfg["planning_steps"] == 20 and cfg["threshold"] == 0.9
    assert cfg["blind_episodes"] == 30 and cfg["n_traj"] == 3
    smoke = x1.make_config(smoke=True)
    assert smoke["train_steps"] == 4000  # ~10x reduced budget


# ------------------------------------------------- per-seed and full output

def _assert_no_nan(x, path="root"):
    if isinstance(x, dict):
        for k, v in x.items():
            _assert_no_nan(v, f"{path}.{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            _assert_no_nan(v, f"{path}[{i}]")
    elif isinstance(x, float):
        assert not math.isnan(x), f"NaN at {path}"


def test_run_seed_structure_and_full_contract(tmp_path):
    cfg = tiny_config()
    results = [x1.run_seed(s, cfg) for s in range(2)]
    r = results[0]

    assert set(r["curves"]) == {"qlearning-A", "dynaq-A"}
    assert all(len(c) == len(r["checkpoints"]) for c in r["curves"].values())
    assert set(r["blind"]) == set(x1.COND_ORDER)
    for name, m in r["blind"].items():
        assert 0.0 <= m["success_rate"] <= 1.0
        assert 1 <= m["mean_steps"] <= cfg["cap"]
    assert set(r["trajectories"]) == set(x1.TRAJ_CONDS)
    for name, trajs in r["trajectories"].items():
        assert 1 <= len(trajs) <= cfg["n_traj"]
        for traj in trajs:
            path = traj["path"]
            assert traj["steps"] + 1 == len(path)  # per-step rows + terminal
            assert path[-1]["action"] is None
            for row in path[:-1]:
                assert isinstance(row["action"], int) and len(row["true"]) == 2
                if name == "random-A":
                    assert row["believed"] is None
                else:
                    assert len(row["believed"]) == 2
                    assert row["entropy_bits"] >= 0.0

    out = x1.aggregate(results, cfg, smoke=True, wall_clock=1.0)
    for key in ("experiment", "hypothesis", "config", "conditions", "curves",
                "tests", "conclusion", "viz"):
        assert key in out
    assert set(out["conditions"]) == set(x1.COND_ORDER) | {"qlearning-A",
                                                           "dynaq-A"}
    for cond, block in out["conditions"].items():
        for vals in block["per_seed"].values():
            assert len(vals) == 2
    cv = out["curves"]
    assert cv["checkpoints"] == results[0]["checkpoints"]
    for cond in ("qlearning-A", "dynaq-A"):
        band = cv["conditions"][cond]
        n = len(cv["checkpoints"])
        assert len(band["iqm"]) == len(band["ci_lo"]) == len(band["ci_hi"]) == n
        assert all(lo <= hi for lo, hi in zip(band["ci_lo"], band["ci_hi"]))
    names = {t["name"] for t in out["tests"]}
    assert {"dyna_faster_than_q_t90", "blindA_beats_blindB",
            "touch_beats_pure_deadreckoning"} <= names
    for t in out["tests"]:
        assert "u" in t and "p" in t
        if t["family"] == "primary":
            assert 0.0 <= t["p_holm"] <= 1.0
    assert isinstance(out["conclusion"]["supported"], bool)
    assert out["conclusion"]["summary"]
    viz = out["viz"]
    for home, map_str in (("A", HOME_A), ("B", HOME_B)):
        h = viz["homes"][home]
        assert h["ascii"] == map_str.strip().splitlines()
        assert h["shortest_path_len"] == GridHome(map_str).shortest_path_len()
        assert [0, 0] in h["walls"]  # border corner is always a wall
        assert len(h["walls"]) > 20
    assert len(viz["blind_summary"]) == len(x1.COND_ORDER)
    assert set(viz["blind_trajectories"]) == set(x1.TRAJ_CONDS)
    assert viz["sample_efficiency"]["threshold"] == cfg["threshold"]

    _assert_no_nan(out)
    p = tmp_path / "exp1.json"
    save_json(p, out)
    loaded = json.loads(p.read_text())
    assert loaded["experiment"] == "exp1_blindfold"
