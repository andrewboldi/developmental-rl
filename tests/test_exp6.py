"""Tests for the EXP6 self-coach harness: the memory-start schedules, the
SelfCoach episodic memory (top-10 by return, ties -> earlier; scoring-gated
fallback to all visited states), snapshot restart fidelity, bit-for-bit
protocol fidelity of the whole/teacher arms to EXP2, exact budget accounting,
eval purity, practice-start heatmaps, the m=2 primary family with the claim-2
equivalence margin, and the shared output contract."""

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import exp2_microtask as exp2  # noqa: E402
import exp6_selfcoach as exp6  # noqa: E402

from devrl.envs.soccer import CARRIED, SCORED, SoccerGrid  # noqa: E402
from devrl.run import run_seeds  # noqa: E402

KICKOFF = (SoccerGrid.KICKOFF_AGENT, SoccerGrid.KICKOFF_BALL)


def _env():
    return SoccerGrid(rng=np.random.default_rng(0))


def test_conditions_and_shared_protocol_constants():
    assert exp6.CONDITIONS == ("whole", "teacher-drills",
                               "self-drills", "self-drills-late")
    # protocol constants are EXP2's, imported (single source of truth)
    assert (exp6.BUDGET, exp6.EVAL_EVERY, exp6.CAP) == (60_000, 2_000, 100)
    assert (exp6.LR, exp6.GAMMA, exp6.EPS) == (exp2.LR, exp2.GAMMA, exp2.EPS)
    assert exp6.THRESHOLD == exp2.THRESHOLD == 0.9
    assert exp6.MEMORY_K == 10 and exp6.MEMORY_P == 0.75
    assert exp6.SELF_FRAC == exp2.SHOOT_FRAC + exp2.DRIBBLE_FRAC == 0.4


def test_memory_p_schedules():
    b = 60_000
    # self-drills: hard phase boundary at 40% of budget
    assert exp6.memory_p("self-drills", 0, b) == 0.75
    assert exp6.memory_p("self-drills", 23_999, b) == 0.75
    assert exp6.memory_p("self-drills", 24_000, b) == 0.0
    assert exp6.memory_p("self-drills", 59_999, b) == 0.0
    # self-drills-late: linear 0.75 -> 0 across the WHOLE budget, no boundary
    assert exp6.memory_p("self-drills-late", 0, b) == 0.75
    assert abs(exp6.memory_p("self-drills-late", 24_000, b) - 0.45) < 1e-12
    assert abs(exp6.memory_p("self-drills-late", 30_000, b) - 0.375) < 1e-12
    assert exp6.memory_p("self-drills-late", 59_999, b) > 0.0
    assert exp6.memory_p("self-drills-late", 60_000, b) == 0.0
    for cond in ("whole", "teacher-drills"):
        for s in (0, 12_000, 30_000):
            assert exp6.memory_p(cond, s, b) == 0.0


def test_selfcoach_fallback_until_first_scoring_episode():
    coach = exp6.SelfCoach()
    assert not coach.has_scoring_episode()
    assert coach.practice_states() == []
    coach.observe(100)
    coach.observe(50)
    coach.end_episode(0.0, [(100, 0), (50, 1)])
    assert not coach.has_scoring_episode()
    assert coach.practice_states() == [50, 100]  # all visited, sorted
    coach.observe(7)  # the visited set keeps growing mid-episode
    assert coach.practice_states() == [7, 50, 100]
    coach.end_episode(1.0, [(200, 2)])
    assert coach.has_scoring_episode()
    # memory now holds the scoring episode AND the earliest zero episode:
    # the union covers both ("those top episodes"), not the visited set
    assert coach.practice_states() == [50, 100, 200]
    assert 7 not in coach.practice_states()


def test_selfcoach_memory_keeps_first_ten_scoring_episodes():
    # binary returns + ties->earlier: the memory freezes on the first 10 goals
    coach = exp6.SelfCoach()
    for i in range(15):
        coach.end_episode(1.0, [(i, 0)])
    assert coach.practice_states() == list(range(10))


def test_selfcoach_sample_start_is_uniform_over_the_union():
    coach = exp6.SelfCoach()
    coach.end_episode(1.0, [(10, 0), (20, 1), (30, 2), (20, 3)])  # dupes collapse
    rng = np.random.default_rng(0)
    draws = [coach.sample_start(rng) for _ in range(3000)]
    counts = {s: draws.count(s) for s in (10, 20, 30)}
    assert set(draws) == {10, 20, 30}
    assert all(c > 800 for c in counts.values())  # roughly uniform
    # empty coach: nothing to sample
    assert exp6.SelfCoach().sample_start(np.random.default_rng(0)) is None
    # determinism: same rng seed -> same draw sequence
    d1 = [coach.sample_start(np.random.default_rng(5)) for _ in range(10)]
    d2 = [coach.sample_start(np.random.default_rng(5)) for _ in range(10)]
    assert d1 == d2


def test_self_start_kinds_and_rng_discipline():
    # empty coach: kickoff even when the coin fires
    coach = exp6.SelfCoach()
    assert exp6.self_start(coach, 1.0, np.random.default_rng(0)) == (*KICKOFF, "kickoff")
    # p=0 consumes NO randomness (game phase leaves the stream untouched)
    rng = np.random.default_rng(42)
    assert exp6.self_start(coach, 0.0, rng) == (*KICKOFF, "kickoff")
    assert rng.random() == np.random.default_rng(42).random()
    # pre-scoring: fallback restarts from a visited state
    env = _env()
    s_vis = env.state_of((2, 3), (5, 6))
    coach.observe(s_vis)
    coach.end_episode(0.0, [(s_vis, 0)])
    agent, ball, kind = exp6.self_start(coach, 1.0, np.random.default_rng(1))
    assert kind == "fallback" and (agent, ball) == ((2, 3), (5, 6))
    # post-scoring: memory restarts from a remembered moment
    s_mem = env.state_of((4, 9), CARRIED)
    coach.end_episode(1.0, [(s_mem, 4)])
    starts = {exp6.self_start(coach, 1.0, np.random.default_rng(i))
              for i in range(40)}
    assert all(kind == "memory" for _, _, kind in starts)
    assert ((4, 9), CARRIED, "memory") in starts       # the scoring moment
    assert ((2, 3), (5, 6), "memory") in starts        # the earliest-zero moment
    # p between 0 and 1: some kickoffs too
    kinds = {exp6.self_start(coach, 0.5, np.random.default_rng(i))[2]
             for i in range(40)}
    assert kinds == {"memory", "kickoff"}


def test_snapshot_roundtrip_for_visited_states():
    # every pre-action state decodes to a legal restart that re-encodes exactly
    env = _env()
    rng = np.random.default_rng(1)
    s, seen = env.reset(), []
    for _ in range(400):
        seen.append(s)
        s, _, done, _ = env.step(int(rng.integers(5)))
        if done:
            s = env.reset()
    for s in set(seen):
        agent, ball = env.decode(s)
        assert ball != SCORED
        assert env.reset(agent=agent, ball=ball) == s


def test_heat_windows_cover_the_practice_phase_in_thirds():
    assert exp6.heat_windows("self-drills", 600) == (
        (0, 80, "early"), (80, 160, "mid"), (160, 240, "end"))
    assert exp6.heat_windows("self-drills-late", 600) == (
        (0, 200, "early"), (200, 400, "mid"), (400, 600, "end"))
    assert exp6.heat_windows("self-drills", 60_000)[2] == (16_000, 24_000, "end")
    assert exp6.heat_windows("whole", 600) is None
    assert exp6.heat_windows("teacher-drills", 600) is None


def test_whole_and_teacher_arms_replicate_exp2_bit_for_bit():
    for c6, c2 in (("whole", "whole"), ("teacher-drills", "drills-varied")):
        a = exp6._train(seed=5, condition=c6, budget=600, eval_every=200, cap=50)
        b = exp2._train(seed=5, condition=c2, budget=600, eval_every=200, cap=50)
        assert a["curve"] == b["curve"]
        assert np.array_equal(a["Q"], b["Q"])
        assert a["checkpoints"] == b["checkpoints"]


def test_training_budget_is_exact_and_checkpoints_matched():
    runs = {c: exp6._train(seed=0, condition=c, budget=600, eval_every=200, cap=50)
            for c in exp6.CONDITIONS}
    for r in runs.values():
        assert r["train_steps"] == 600  # budgets matched EXACTLY, all arms
        assert r["checkpoints"] == [0, 200, 400, 600]
        assert len(r["curve"]) == 4
        assert all(0.0 <= v <= 1.0 for v in r["curve"])


def test_eval_frequency_does_not_affect_training_or_coach():
    a = exp6._train(seed=3, condition="self-drills", budget=400, eval_every=100, cap=40)
    b = exp6._train(seed=3, condition="self-drills", budget=400, eval_every=400, cap=40)
    assert np.array_equal(a["Q"], b["Q"])
    assert a["self_stats"] == b["self_stats"]
    assert a["heat"] == b["heat"]


def test_self_mechanism_changes_training_and_is_accounted():
    self_run = exp6._train(seed=3, condition="self-drills",
                           budget=400, eval_every=400, cap=40)
    whole_run = exp6._train(seed=3, condition="whole",
                            budget=400, eval_every=400, cap=40)
    st = self_run["self_stats"]
    assert st["n_memory_starts"] + st["n_fallback_starts"] > 0
    assert not np.array_equal(self_run["Q"], whole_run["Q"])
    # heatmap counts every self-chosen (non-kickoff) start, and only those
    heat = self_run["heat"]["windows"]
    n_events = sum(w["n_memory"] + w["n_fallback"] for w in heat)
    assert n_events == st["n_memory_starts"] + st["n_fallback_starts"]
    for w in heat:
        assert sum(map(sum, w["agent_counts"])) == w["n_memory"] + w["n_fallback"]
        assert sum(map(sum, w["ball_counts"])) + w["ball_carried"] \
            == w["n_memory"] + w["n_fallback"]
    # whole/teacher arms carry no self machinery
    assert "self_stats" not in whole_run and "heat" not in whole_run


def test_self_coach_engages_memory_after_first_goal():
    # seed pinned from the dev range (0-50; confirmatory seeds are 100+): a
    # goal happens well inside 4k steps, after which memory starts must fire
    out = exp6._train(seed=exp6.DEV_DEMO_SEED, condition="self-drills",
                      budget=4_000, eval_every=2_000, cap=100)
    st = out["self_stats"]
    assert st["first_score_step"] is not None
    assert 0 < st["first_score_step"] <= 4_000
    assert st["n_memory_starts"] > 0
    assert st["n_fallback_starts"] > 0  # exploration restarts happened first


def test_run_seed_is_json_safe_and_picklable_across_fork():
    fn = partial(exp6.run_seed, condition="self-drills",
                 budget=200, eval_every=100, cap=20)
    out = run_seeds(fn, n_seeds=2, n_jobs=2)
    assert [o["seed"] for o in out] == [0, 1]
    for o in out:
        assert "Q" not in o and "coach" not in o
        json.dumps(o)  # JSON-safe payload


def test_run_seed_at_applies_seed_offset():
    fn = partial(exp6.run_seed_at, offset=100, condition="whole",
                 budget=200, eval_every=100, cap=20)
    out = run_seeds(fn, n_seeds=2, n_jobs=2)
    assert [o["seed"] for o in out] == [100, 101]  # fresh confirmatory seeds
    ref = exp6.run_seed(101, condition="whole", budget=200, eval_every=100, cap=20)
    assert out[1]["curve"] == ref["curve"]  # offset shifts the seed, nothing else


def test_phase_blocks_carry_the_memory_schedule():
    blocks = {c: exp6.phase_blocks6(c, 60_000) for c in exp6.CONDITIONS}
    assert blocks["whole"] == [{"phase": "game", "start": 0, "end": 60_000}]
    assert [b["phase"] for b in blocks["teacher-drills"]] == ["shoot", "dribble", "game"]
    assert blocks["self-drills"] == [
        {"phase": "self-practice", "start": 0, "end": 24_000, "p_memory": 0.75},
        {"phase": "game", "start": 24_000, "end": 60_000}]
    assert blocks["self-drills-late"] == [
        {"phase": "self-practice-annealed", "start": 0, "end": 60_000,
         "p_memory_start": 0.75, "p_memory_end": 0.0}]


def test_match_analysis_diff_ci_and_margin():
    rng_a = [10_000] * 10 + [12_000] * 10
    rng_b = [10_500] * 10 + [11_500] * 10
    m = exp6.match_analysis(rng_a, rng_b, n_boot=500)
    assert m["diff_iqm"] == exp6.stats.iqm(rng_a) - exp6.stats.iqm(rng_b)
    assert m["ci_lo"] <= m["diff_iqm"] <= m["ci_hi"]
    assert m["margin"] == 0.2 * exp6.stats.iqm(rng_b)
    assert m["margin_frac"] == 0.2
    # identical samples: diff 0, CI straddles (or touches) 0
    m0 = exp6.match_analysis([1000, 2000, 3000], [1000, 2000, 3000], n_boot=300)
    assert m0["diff_iqm"] == 0.0
    assert m0["ci_lo"] <= 0.0 <= m0["ci_hi"]


def _mk_primary(comparison, iqm_a, iqm_b, sig_mw, sig_welch):
    a, b = comparison.split(" vs ")
    return {"comparison": comparison,
            "metric": "time_to_90 (steps; censored -> budget+1)",
            "u": 0.0, "p": 0.001 if sig_mw else 0.5,
            "p_holm": 0.005 if sig_mw else 1.0,
            "welch_t": 0.0, "welch_p": 0.001 if sig_welch else 0.5,
            "welch_p_holm": 0.005 if sig_welch else 1.0,
            "iqm": {a: iqm_a, b: iqm_b},
            "faster": a if iqm_a < iqm_b else (b if iqm_b < iqm_a else "tie"),
            "significant": sig_mw, "welch_significant": sig_welch}


def _mk_conditions(iqms):
    return {c: {"n_seeds": 30, "n_censored": 0, "time_to_90_iqm": v}
            for c, v in iqms.items()}


def _mk_match(diff, ci_hi, teacher_iqm):
    return {"diff_iqm": diff, "ci_lo": diff - 2000.0, "ci_hi": ci_hi,
            "margin": 0.2 * teacher_iqm, "margin_frac": 0.2}


def _claims(sig1_mw, sig1_w, self_iqm, teach_iqm, sig2_mw, sig2_w, ci_hi,
            whole_iqm=40_000):
    conds = _mk_conditions({"whole": whole_iqm, "teacher-drills": teach_iqm,
                            "self-drills": self_iqm,
                            "self-drills-late": self_iqm + 1000})
    primary = [
        _mk_primary("self-drills vs whole", self_iqm, whole_iqm, sig1_mw, sig1_w),
        _mk_primary("self-drills vs teacher-drills", self_iqm, teach_iqm,
                    sig2_mw, sig2_w)]
    match = _mk_match(self_iqm - teach_iqm, ci_hi, teach_iqm)
    return exp6.build_claims(primary, match, conds)


def test_build_claims_claim1_directional_verdicts():
    # both stats significant, self faster -> supported
    c = _claims(True, True, 15_000, 15_500, False, False, 2000.0)
    assert c[0]["verdict"] == "supported"
    # both significant, whole faster -> refuted
    c = _claims(True, True, 45_000, 15_500, False, False, 2000.0,
                whole_iqm=20_000)
    assert c[0]["verdict"] == "refuted"
    # one-legged detection -> boundary; none -> null
    assert _claims(True, False, 15_000, 15_500, False, False,
                   2000.0)[0]["verdict"] == "boundary"
    assert _claims(False, False, 15_000, 15_500, False, False,
                   2000.0)[0]["verdict"] == "null"


def test_build_claims_claim2_match_verdicts():
    # no detection, CI within +20% margin -> supported (equivalence shown)
    c = _claims(True, True, 15_500, 15_000, False, False, ci_hi=2000.0)
    assert c[1]["verdict"] == "supported"
    assert "margin" in c[1]["evidence"]
    # no detection, CI beyond margin -> null (underpowered, never equivalence)
    c = _claims(True, True, 15_500, 15_000, False, False, ci_hi=9000.0)
    assert c[1]["verdict"] == "null"
    # teacher detectably faster, CI beyond margin -> refuted, value quantified
    c = _claims(True, True, 25_000, 15_000, True, True, ci_hi=12_000.0)
    assert c[1]["verdict"] == "refuted"
    assert "10000" in c[1]["evidence"].replace(",", "")
    # teacher detectably faster (even one-legged) but bounded within margin
    c = _claims(True, True, 16_500, 15_000, True, False, ci_hi=2500.0)
    assert c[1]["verdict"] == "boundary"
    # self significantly faster than the teacher on both stats -> supported
    c = _claims(True, True, 11_000, 15_000, True, True, ci_hi=-1000.0)
    assert c[1]["verdict"] == "supported"
    # one-legged detection with self faster -> boundary
    c = _claims(True, True, 14_000, 15_000, True, False, ci_hi=500.0)
    assert c[1]["verdict"] == "boundary"
    for claim in c:
        assert {"claim", "verdict", "evidence"} <= set(claim)
        assert claim["verdict"] in {"supported", "refuted", "null", "boundary"}


def test_assemble_meets_output_contract():
    results = {c: [exp6.run_seed(s, condition=c, budget=400, eval_every=100, cap=30)
                   for s in range(2)] for c in exp6.CONDITIONS}
    out = exp6.assemble(results, budget=400, n_boot=200)
    for key in ("experiment", "hypothesis", "config", "conditions", "curves",
                "tests", "conclusion", "viz"):
        assert key in out
    assert out["experiment"] == "exp6_selfcoach"

    ckpts = out["curves"]["checkpoints"]
    assert ckpts == [0, 100, 200, 300, 400]
    for cond in exp6.CONDITIONS:
        c = out["conditions"][cond]
        assert len(c["final_success"]) == 2 and len(c["time_to_90"]) == 2
        assert c["n_censored"] + sum(t is not None for t in c["time_to_90"]) == 2
        cv = out["curves"]["conditions"][cond]
        assert len(cv["iqm"]) == len(cv["ci_lo"]) == len(cv["ci_hi"]) == len(ckpts)
        for k in ("iqm", "ci_lo", "ci_hi"):
            assert np.isfinite(np.asarray(cv[k])).all()
    for cond in ("self-drills", "self-drills-late"):
        stats_list = out["conditions"][cond]["self_stats"]
        assert len(stats_list) == 2
        assert {"first_score_step", "n_memory_starts", "n_fallback_starts",
                "n_kickoff_starts"} <= set(stats_list[0])
    for cond in ("whole", "teacher-drills"):
        assert "self_stats" not in out["conditions"][cond]

    tests = out["tests"]
    assert [t["comparison"] for t in tests["primary"]] == [
        "self-drills vs whole", "self-drills vs teacher-drills"]
    for t in tests["primary"]:
        for k in ("p", "p_holm", "welch_p", "welch_p_holm"):
            assert np.isfinite(t[k])
        assert isinstance(t["significant"], bool)
        assert isinstance(t["welch_significant"], bool)
    assert {"diff_iqm", "ci_lo", "ci_hi", "margin"} <= set(tests["match_analysis"])
    assert [t["comparison"] for t in tests["secondary_t90_context"]] == [
        "teacher-drills vs whole", "self-drills-late vs whole",
        "self-drills vs self-drills-late"]
    assert len(tests["secondary_final_success"]) == 2

    conclusion = out["conclusion"]
    assert {"claims", "summary"} <= set(conclusion)
    assert "supported" not in conclusion  # per-claim verdicts, no boolean
    assert len(conclusion["claims"]) == 2
    for c in conclusion["claims"]:
        assert {"claim", "verdict", "evidence"} <= set(c)
        assert c["verdict"] in {"supported", "refuted", "null", "boundary"}

    viz = out["viz"]
    assert viz["pitch"]["W"] == 11 and viz["pitch"]["H"] == 7
    assert set(viz["trajectories"]) == set(exp6.CONDITIONS)
    for cond in exp6.CONDITIONS:
        assert set(viz["trajectories"][cond]) == {"25", "50", "100"}
    assert [r["condition"] for r in viz["t90_table"]] == list(exp6.CONDITIONS)
    for row in viz["t90_table"]:
        assert {"t90_iqm", "ci", "n_censored", "n_seeds",
                "final_success_iqm"} <= set(row)
    heat = viz["practice_heatmaps"]
    assert set(heat) == {"self-drills", "self-drills-late"}
    for cond, block in heat.items():
        assert block["seed"] == viz["viz_seed"][cond]
        assert [w["label"] for w in block["windows"]] == ["early", "mid", "end"]
        for w in block["windows"]:
            assert len(w["agent_counts"]) == 7
            assert all(len(row) == 11 for row in w["agent_counts"])
        assert len(block["mean_agent_counts"]) == 3
        assert all(len(g) == 7 and all(len(r) == 11 for r in g)
                   for g in block["mean_agent_counts"])
    assert viz["phases"]["self-drills"][0]["end"] == 160  # 40% of 400
    assert viz["eval_curves"]["checkpoints"] == ckpts
    assert viz["eval_curves"]["threshold"] == 0.9
    assert set(viz["self_coach"]["first_score_steps"]) == {
        "self-drills", "self-drills-late"}
    json.dumps(out, default=float)  # whole payload is JSON-exportable
