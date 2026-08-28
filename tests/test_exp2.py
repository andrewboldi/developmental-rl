"""Tests for the EXP2 harness: curriculum schedule, start states, exact budget
accounting, eval purity (greedy rollouts must not touch training state), and
the shared output contract."""

import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import exp2_microtask as exp2  # noqa: E402

from devrl.envs.soccer import CARRIED, SoccerGrid  # noqa: E402
from devrl.run import run_seeds  # noqa: E402


def test_phase_schedule_is_reverse_curriculum():
    b = 60_000
    assert [exp2.phase_at("whole", s, b)
            for s in (0, 11_999, 12_000, 24_000, 59_999)] == ["game"] * 5
    for cond in ("drills-varied", "drills-fixed"):
        got = [exp2.phase_at(cond, s, b)
               for s in (0, 11_999, 12_000, 23_999, 24_000, 59_999)]
        assert got == ["shoot", "shoot", "dribble", "dribble", "game", "game"]


def test_varied_drill_start_states_cover_their_regions():
    rng = np.random.default_rng(0)
    shots = [exp2.start_state("drills-varied", "shoot", rng) for _ in range(300)]
    assert all(ball == CARRIED for _, ball in shots)  # spawn already carrying
    assert all(c in (8, 9, 10) and 0 <= r <= 6 for (r, c), _ in shots)
    assert len({cell for cell, _ in shots}) > 10  # varied, not blocked

    dribbles = [exp2.start_state("drills-varied", "dribble", rng) for _ in range(300)]
    assert all(ball == (3, 5) for _, ball in dribbles)  # ball waits at center
    assert all(0 <= c <= 4 and 0 <= r <= 6 for (r, c), _ in dribbles)
    assert len({cell for cell, _ in dribbles}) > 10


def test_fixed_drill_start_states_are_blocked():
    rng = np.random.default_rng(0)
    assert {exp2.start_state("drills-fixed", "shoot", rng)
            for _ in range(50)} == {((3, 9), CARRIED)}
    assert {exp2.start_state("drills-fixed", "dribble", rng)
            for _ in range(50)} == {((3, 2), (3, 5))}


def test_game_phase_starts_from_kickoff_in_all_conditions():
    rng = np.random.default_rng(0)
    for cond in exp2.CONDITIONS:
        assert exp2.start_state(cond, "game", rng) == ((3, 0), (3, 5))


def test_training_budget_is_exact_and_checkpoints_matched():
    runs = {c: exp2._train(seed=0, condition=c, budget=600, eval_every=200, cap=50)
            for c in exp2.CONDITIONS}
    for r in runs.values():
        assert r["train_steps"] == 600  # budgets matched EXACTLY
        assert r["checkpoints"] == [0, 200, 400, 600]
        assert len(r["curve"]) == 4
        assert all(0.0 <= v <= 1.0 for v in r["curve"])


def test_eval_frequency_does_not_affect_training():
    # Greedy eval runs on its own env and rng: evaluating 4x as often must
    # leave the learned Q-table bit-identical.
    a = exp2._train(seed=3, condition="drills-varied", budget=400, eval_every=100, cap=40)
    b = exp2._train(seed=3, condition="drills-varied", budget=400, eval_every=400, cap=40)
    assert np.array_equal(a["Q"], b["Q"])


def test_greedy_eval_is_deterministic_and_pure():
    Q = np.zeros((SoccerGrid.n_states, SoccerGrid.n_actions))
    r1 = exp2.greedy_eval(Q, np.random.default_rng(7), n_episodes=5, cap=30)
    r2 = exp2.greedy_eval(Q, np.random.default_rng(7), n_episodes=5, cap=30)
    assert r1 == r2 == 0.0  # all-zero Q: argmax walks UP forever, never scores
    assert not Q.any()  # eval never mutates the Q-table


def _toward(src, dst):
    if src[0] > dst[0]:
        return SoccerGrid.UP
    if src[0] < dst[0]:
        return SoccerGrid.DOWN
    if src[1] < dst[1]:
        return SoccerGrid.RIGHT
    return SoccerGrid.LEFT


def _optimal_q():
    """Hand-built Q encoding the optimal kickoff policy: fetch the ball,
    dribble to the goal center (shot p=1 there), shoot."""
    env = SoccerGrid(rng=np.random.default_rng(0))
    Q = np.zeros((SoccerGrid.n_states, SoccerGrid.n_actions))
    for cell in range(SoccerGrid.n_cells):
        rc = divmod(cell, SoccerGrid.W)
        if rc != SoccerGrid.KICKOFF_BALL:
            s = env.state_of(rc, SoccerGrid.KICKOFF_BALL)
            Q[s, _toward(rc, SoccerGrid.KICKOFF_BALL)] = 1.0
        a = SoccerGrid.SHOOT if rc == SoccerGrid.GOAL_CENTER else _toward(rc, SoccerGrid.GOAL_CENTER)
        Q[env.state_of(rc, CARRIED), a] = 1.0
    return Q


def test_greedy_eval_scores_with_optimal_policy():
    success = exp2.greedy_eval(_optimal_q(), np.random.default_rng(0),
                               n_episodes=20, cap=100)
    assert success == 1.0  # point-blank shots are deterministic goals


def test_greedy_traj_records_path_ball_and_outcome():
    traj = exp2.greedy_traj(_optimal_q(), np.random.default_rng(0), cap=100)
    assert traj["scored"] is True
    assert len(traj["steps"]) == 11  # 5 to the ball, 5 to the goal mouth, 1 shot
    assert traj["steps"][0] == {"agent": [3, 0], "ball": [3, 5], "a": SoccerGrid.RIGHT}
    assert traj["steps"][5]["ball"] == "carried"
    assert traj["steps"][-1]["a"] == SoccerGrid.SHOOT
    assert traj["steps"][-1]["shot_p"] == 1.0
    assert traj["final"]["ball"] == "scored"


def test_greedy_traj_respects_step_cap():
    Q = np.zeros((SoccerGrid.n_states, SoccerGrid.n_actions))
    traj = exp2.greedy_traj(Q, np.random.default_rng(0), cap=17)
    assert traj["scored"] is False and len(traj["steps"]) == 17


def test_trajectories_sampled_at_quarter_half_full_budget():
    out = exp2._train(seed=1, condition="whole", budget=400, eval_every=200, cap=30)
    assert set(out["trajs"]) == {"25", "50", "100"}
    assert [out["trajs"][k]["step"] for k in ("25", "50", "100")] == [100, 200, 400]


def test_t90_imputation_and_censoring():
    ckpts = [0, 100, 200]
    curves = [[0.0, 0.95, 1.0], [0.0, 0.5, 0.89]]
    t90, imputed, n_cens = exp2.t90_stats(ckpts, curves, budget=200)
    assert t90 == [100, None]
    assert imputed == [100, 201]  # censored -> budget + 1 (conservative)
    assert n_cens == 1


def test_holm_adjustment_is_monotone_and_capped():
    assert np.allclose(exp2.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert list(exp2.holm([0.9, 0.8])) == [1.0, 1.0]


def test_mann_whitney_guard_handles_fully_tied_samples():
    # e.g. every seed censored in both conditions during a smoke run
    res = exp2._mw([201, 201], [201, 201])
    assert res["p"] == 1.0 and np.isfinite(res["u"])


def test_run_seed_is_picklable_across_fork():
    fn = partial(exp2.run_seed, condition="whole", budget=200, eval_every=100, cap=20)
    out = run_seeds(fn, n_seeds=2, n_jobs=2)
    assert [o["seed"] for o in out] == [0, 1]
    assert all("Q" not in o for o in out)  # JSON-safe payload only


def test_assemble_meets_output_contract():
    results = {c: [exp2.run_seed(s, condition=c, budget=400, eval_every=100, cap=30)
                   for s in range(2)] for c in exp2.CONDITIONS}
    out = exp2.assemble(results, budget=400, n_boot=200)
    for key in ("experiment", "hypothesis", "conditions", "curves", "tests",
                "conclusion", "viz"):
        assert key in out
    assert out["experiment"] == "exp2_microtask"

    ckpts = out["curves"]["checkpoints"]
    assert ckpts == [0, 100, 200, 300, 400]
    for cond in exp2.CONDITIONS:
        c = out["conditions"][cond]
        assert len(c["final_success"]) == 2 and len(c["time_to_90"]) == 2
        assert c["n_censored"] + sum(t is not None for t in c["time_to_90"]) == 2
        cv = out["curves"]["conditions"][cond]
        assert len(cv["iqm"]) == len(cv["ci_lo"]) == len(cv["ci_hi"]) == len(ckpts)
        for k in ("iqm", "ci_lo", "ci_hi"):
            assert np.isfinite(np.asarray(cv[k])).all()

    assert len(out["tests"]["primary"]) == 3
    for t in out["tests"]["primary"]:
        assert np.isfinite(t["p"]) and np.isfinite(t["p_holm"])
    assert {"supported", "summary"} <= set(out["conclusion"])

    viz = out["viz"]
    pitch = viz["pitch"]
    assert pitch["W"] == 11 and pitch["H"] == 7
    assert pitch["goal_cells"] == [[2, 10], [3, 10], [4, 10]]
    assert len(pitch["shot_p"]) == 7 and all(len(row) == 11 for row in pitch["shot_p"])
    assert set(viz["trajectories"]) == set(exp2.CONDITIONS)
    for cond in exp2.CONDITIONS:
        assert set(viz["trajectories"][cond]) == {"25", "50", "100"}
        assert len(viz["phases"][cond]) == (1 if cond == "whole" else 3)
    assert viz["eval_curves"]["checkpoints"] == ckpts
