import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from devrl.agents.distill import (EpisodicMemory, TransitionLog, apply_advice,
                                  extract_advice, halflife_schedule,
                                  replay_pretrain)
from devrl.envs.trapgrid import TrapGrid

_spec = importlib.util.spec_from_file_location(
    "exp4_generations",
    Path(__file__).resolve().parents[1] / "experiments" / "exp4_generations.py")
exp4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp4)

# ~75x reduced budget for fast structural tests; halflives scale with life
TINY = dict(exp4.FULL, life=200, halflife=66, slow_halflife=333, n_boot=100)

VERDICTS = {"supported", "refuted", "null", "boundary"}


# ---------------------------------------------------------------- distill.py

def test_memory_keeps_top_k_by_return():
    m = EpisodicMemory(k=3)
    for ret in [1.0, 5.0, 3.0, 10.0, 2.0]:
        m.add(ret, [(0, 0)])
    assert [r for r, _ in m.episodes] == [10.0, 5.0, 3.0]
    assert m.best_return() == 10.0


def test_memory_tie_breaks_keep_earlier_episode_by_default():
    # registered primary rule: ties keep the earlier episode (a trapped
    # agent's tied-return memories are its earliest exploratory meanders)
    m = EpisodicMemory(k=1)
    m.add(10.0, [(0, 0), (1, 1), (2, 2)])
    m.add(10.0, [(3, 3), (4, 4)])  # same return, arrives later
    assert m.episodes[0][1] == [(0, 0), (1, 1), (2, 2)]
    assert EpisodicMemory(k=1).tie_break == "earliest"


def test_memory_tie_break_shortest_keeps_shortest_episode():
    # robustness rule mandated by verification: ties keep the SHORTEST episode
    m = EpisodicMemory(k=1, tie_break="shortest")
    m.add(10.0, [(0, 0), (1, 1), (2, 2)])
    m.add(10.0, [(3, 3), (4, 4)])  # same return, shorter, later
    assert m.episodes[0][1] == [(3, 3), (4, 4)]


def test_memory_tie_break_shortest_equal_lengths_keep_earlier():
    m = EpisodicMemory(k=1, tie_break="shortest")
    m.add(5.0, [(0, 0), (1, 1)])
    m.add(5.0, [(2, 2), (3, 3)])  # same return, same length, later
    assert m.episodes[0][1] == [(0, 0), (1, 1)]


def test_memory_tie_break_shortest_still_ranks_by_return_first():
    m = EpisodicMemory(k=2, tie_break="shortest")
    m.add(1.0, [(0, 0)])
    m.add(10.0, [(1, 1), (2, 2), (3, 3)])
    assert [r for r, _ in m.episodes] == [10.0, 1.0]
    assert m.best_return() == 10.0


def test_memory_rejects_unknown_tie_break():
    with pytest.raises(ValueError):
        EpisodicMemory(tie_break="longest")


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


def test_transition_log_records_full_lifetime():
    log = TransitionLog()
    assert len(log) == 0
    log.add(0, 1, 0.0, 5, False)
    log.add(5, 2, 10.0, 9, True)
    assert len(log) == 2
    assert log.transitions == [(0, 1, 0.0, 5, False), (5, 2, 10.0, 9, True)]
    assert isinstance(log.transitions[1][4], bool)  # done stored as bool


def test_transition_log_shuffled_is_rng_permutation_of_everything():
    log = TransitionLog()
    for i in range(50):
        log.add(i, i % 4, float(i), i + 1, False)
    out = log.shuffled(np.random.default_rng(0))
    assert sorted(out) == sorted(log.transitions)  # nothing lost, nothing added
    assert out != log.transitions  # actually shuffled
    assert out == log.shuffled(np.random.default_rng(0))  # rng-deterministic
    assert out != log.shuffled(np.random.default_rng(1))
    assert log.transitions[0] == (0, 0, 0.0, 1, False)  # log itself untouched


def test_transition_log_sample_uniform_without_replacement():
    log = TransitionLog()
    for i in range(200):
        log.add(i, 0, 0.0, i + 1, False)
    got = log.sample(np.random.default_rng(3), 100)
    assert len(got) == 100
    assert len(set(got)) == 100  # distinct log entries: no replacement
    assert set(got) <= set(log.transitions)
    assert got == log.sample(np.random.default_rng(3), 100)  # rng-deterministic
    assert got != log.sample(np.random.default_rng(4), 100)


def test_replay_pretrain_is_one_pass_of_q_updates_at_fixed_lr():
    Q = np.zeros((6, 2))
    trans = [
        (0, 1, 10.0, 5, True),   # terminal: target = r, no bootstrap
        (1, 0, 0.0, 0, False),   # bootstraps through Q[0] AFTER the 1st update
        (1, 0, 0.0, 0, False),   # sequential pass: sees its own prior effect
    ]
    out = replay_pretrain(Q, trans, lr=0.5, gamma=0.9)
    assert out is Q  # in place
    assert Q[0, 1] == pytest.approx(5.0)  # 0 + 0.5*(10 - 0)
    # 2nd: target 0.9*5=4.5 -> 2.25; 3rd: 2.25 + 0.5*(4.5-2.25) = 3.375
    assert Q[1, 0] == pytest.approx(3.375)
    assert Q.sum() == pytest.approx(5.0 + 3.375)  # only replayed pairs move


def test_replay_pretrain_touches_no_agent_state():
    # the plasticity symmetry with advice priming: replay writes Q only —
    # the student's age (hence its lr/eps schedules) is untouched
    ag = exp4._agent_for("reset-replay-full", exp4.FULL, np.random.default_rng(0))
    replay_pretrain(ag.Q, [(0, 1, 10.0, 5, True)] * 50,
                    lr=exp4.FULL["lr0"], gamma=exp4.FULL["gamma"])
    assert ag.age == 0
    assert ag.current_lr() == exp4.FULL["lr0"]
    assert ag.current_eps() == exp4.FULL["eps0"]
    assert ag.Q[0, 1] != 0.0


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
    # v3: raw-replay dose is matched to the advice bottleneck by construction
    assert c["replay_dose"] == c["advice_cap"] == 100
    s = exp4.SMOKE
    assert s["life"] == 1500 and s["halflife"] == 500 and s["slow_halflife"] == 2500
    assert s["replay_dose"] == 100


def test_conditions_registered_order_keeps_originals_then_added_arms():
    # original five first, v2 arms next, v3 reset-replay arms appended: every
    # pre-existing condition keeps its rng stream index, so the v2 arms are
    # bit-identical reruns and only the two new arms draw fresh streams
    assert exp4.CONDITIONS == [
        "generational-distill", "weight-copy", "one-long-life",
        "one-long-life-slow", "no-inheritance",
        "generational-distill-shortest", "random-advice",
        "optimistic-init", "constant-eps-life",
        "reset-replay-full", "reset-replay-100"]
    assert exp4.REPLAY_CONDITIONS == ("reset-replay-full", "reset-replay-100")


def test_agent_for_constant_eps_life_keeps_eps_while_lr_decays():
    ag = exp4._agent_for("constant-eps-life", exp4.FULL, np.random.default_rng(0))
    ag.age = 75000
    assert ag.current_eps() == 0.4  # exploration never dies
    assert ag.current_lr() == pytest.approx(0.3 * 2.0 ** (-15))  # lr decays as usual
    assert np.all(ag.Q == 0.0)


def test_agent_for_optimistic_init_blesses_entire_table():
    ag = exp4._agent_for("optimistic-init", exp4.FULL, np.random.default_rng(0))
    assert np.all(ag.Q == 5.0)  # Q0 = 5.0 everywhere
    ag.age = 5000
    assert ag.current_lr() == pytest.approx(0.15)  # standard decay
    assert ag.current_eps() == pytest.approx(0.2)


def test_agent_for_standard_and_slow_conditions():
    ag = exp4._agent_for("generational-distill", exp4.FULL, np.random.default_rng(0))
    assert np.all(ag.Q == 0.0)
    ag.age = 5000
    assert ag.current_lr() == pytest.approx(0.15)
    assert ag.current_eps() == pytest.approx(0.2)
    slow = exp4._agent_for("one-long-life-slow", exp4.FULL, np.random.default_rng(0))
    slow.age = 25000
    assert slow.current_lr() == pytest.approx(0.15)  # slow halflife 25k
    # v3 replay students are standard fresh students: Q0=0, standard decay —
    # plasticity exactly matched to the distill student
    for cond in ("reset-replay-full", "reset-replay-100"):
        rp = exp4._agent_for(cond, exp4.FULL, np.random.default_rng(0))
        assert np.all(rp.Q == 0.0) and rp.age == 0
        rp.age = 5000
        assert rp.current_lr() == pytest.approx(0.15)
        assert rp.current_eps() == pytest.approx(0.2)


def test_random_advice_pairs_are_valid_dedup_and_capped():
    adv = exp4._random_advice(np.random.default_rng(0), 100)
    assert len(adv) == 100 and len(set(adv)) == 100
    env = TrapGrid()
    for s, a in adv:
        assert isinstance(s, int) and isinstance(a, int)
        assert 0 <= a < TrapGrid.n_actions
        rc = env.pos_of(s)
        assert env.grid[rc] != "#"  # actable cells only
        assert rc not in env.candies and rc != env.goal  # non-terminal


def test_random_advice_is_rng_driven():
    a1 = exp4._random_advice(np.random.default_rng(1), 100)
    a2 = exp4._random_advice(np.random.default_rng(1), 100)
    a3 = exp4._random_advice(np.random.default_rng(2), 100)
    assert a1 == a2 and a1 != a3


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
    for cond in ("generational-distill", "generational-distill-shortest",
                 "random-advice"):
        advice = r[cond]["advice_by_gen"]
        assert len(advice) == TINY["gens"]
        for adv in advice:
            assert 0 < len(adv) <= TINY["advice_cap"]
            assert len(set(adv)) == len(adv)  # deduplicated
            assert all(0 <= s < 165 and 0 <= a < 4 for s, a in adv)
    # random-advice always ships the full optimism dose
    assert all(len(a) == TINY["advice_cap"]
               for a in r["random-advice"]["advice_by_gen"])
    assert "advice_by_gen" not in r["weight-copy"]
    assert "advice_by_gen" not in r["one-long-life"]


def test_replay_conditions_are_budget_free_and_dose_matched():
    r = exp4._run_seed(0, TINY)
    life, gens = TINY["life"], TINY["gens"]
    full, dose = r["reset-replay-full"], r["reset-replay-100"]
    # budget symmetry: replay updates are free the same way advice priming
    # is free — they consume no env steps, so the trained budget is exact
    assert full["steps_trained"] == dose["steps_trained"] == gens * life
    # gen 1 has no teacher and replays nothing (matching distill's
    # advice-free gen 1); afterwards the full arm replays the teacher's
    # ENTIRE lifetime log, the dose arm exactly replay_dose transitions
    assert full["replayed_by_gen"] == [0] + [life] * (gens - 1)
    assert dose["replayed_by_gen"] == [0] + [TINY["replay_dose"]] * (gens - 1)
    for res in (full, dose):
        assert "advice_by_gen" not in res  # raw channel, no curated advice
        assert "log" not in res and "transitions" not in res  # never serialized
    assert "replayed_by_gen" not in r["generational-distill"]
    assert "replayed_by_gen" not in r["one-long-life"]


def test_live_records_the_exact_transition_stream():
    # the log IS the lived experience: replaying the logged actions through a
    # twin deterministic env reproduces every (s, a, r, s2, done) tuple
    env = TrapGrid(cap=TINY["cap"])
    agent = exp4._agent_for("reset-replay-full", TINY, np.random.default_rng(7))
    log = TransitionLog()
    exp4._live(agent, env, EpisodicMemory(3), 50, log=log)
    assert len(log) == 50  # one entry per env step, cap-truncations included
    twin = TrapGrid(cap=TINY["cap"])
    s = twin.reset()
    for ls, la, lrew, ls2, ldone in log.transitions:
        assert ls == s
        s2, r, done, info = twin.step(la)
        assert (ls2, lrew, ldone) == (s2, r, done)
        s = twin.reset() if done or info["truncated"] else s2


def test_replay_wiring_fresh_student_initial_lr_right_dose(monkeypatch):
    # replay_pretrain must be called once per generation after the first,
    # on the STILL-FRESH student's zero Q table (before it lives), at the
    # student's initial lr, with the registered dose
    calls = []
    real = exp4.replay_pretrain

    def spy(Q, transitions, lr, gamma):
        calls.append({"n": len(transitions), "lr": lr, "gamma": gamma,
                      "fresh": bool(np.all(Q == 0.0))})
        return real(Q, transitions, lr, gamma)

    monkeypatch.setattr(exp4, "replay_pretrain", spy)
    exp4._run_condition("reset-replay-full", np.random.default_rng(0), TINY)
    assert [c["n"] for c in calls] == [TINY["life"]] * (TINY["gens"] - 1)
    exp4._run_condition("reset-replay-100", np.random.default_rng(0), TINY)
    assert [c["n"] for c in calls[TINY["gens"] - 1:]] == (
        [TINY["replay_dose"]] * (TINY["gens"] - 1))
    for c in calls:
        assert c["lr"] == TINY["lr0"] and c["gamma"] == TINY["gamma"]
        assert c["fresh"]  # fresh Q at replay time: full plasticity preserved


def test_run_seed_is_deterministic_per_seed():
    assert exp4._run_seed(3, TINY) == exp4._run_seed(3, TINY)
    assert exp4._run_seed(0, TINY) != exp4._run_seed(1, TINY)


def test_seed_offset_shifts_the_seed_stream():
    # offset k + index i must equal plain seed k+i: fresh confirmatory seeds
    # 100..159 are the same streams a --seeds 160 run would call 100..159
    assert exp4._run_seed(0, TINY, offset=100) == exp4._run_seed(100, TINY)
    assert exp4._run_seed(0, TINY, offset=100) != exp4._run_seed(0, TINY)


def test_holm_bonferroni_adjustment():
    assert np.allclose(exp4._holm([0.01, 0.04, 0.03, 0.5]),
                       [0.04, 0.09, 0.09, 0.5])
    assert list(exp4._holm([0.9, 0.8])) == [1.0, 1.0]


def test_welch_safe_matches_scipy_and_handles_degenerate():
    from scipy.stats import ttest_ind
    a, b = [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]
    w = exp4._welch_safe(a, b)
    t, p = ttest_ind(a, b, equal_var=False)
    assert w["t"] == pytest.approx(float(t)) and w["p"] == pytest.approx(float(p))
    # both groups constant and equal: no evidence either way
    assert exp4._welch_safe([0.3] * 5, [0.3] * 5) == {"t": 0.0, "p": 1.0}
    # both constant, different: deterministic separation (t untestable -> None,
    # never Infinity, which would break strict JSON)
    zz = exp4._welch_safe([1.0] * 5, [2.0] * 5)
    assert zz["p"] == 0.0 and zz["t"] is None
    # one degenerate side is fine for Welch
    one = exp4._welch_safe([0.3] * 5, [0.0, 10.0, 0.3, 10.0, 0.0])
    assert np.isfinite(one["t"]) and 0.0 <= one["p"] <= 1.0


def _comp(iqm_a, iqm_b, sig_mw, sig_w, mean_a=None, mean_b=None):
    return {"iqm_a": iqm_a, "iqm_b": iqm_b,
            "mean_a": iqm_a if mean_a is None else mean_a,
            "mean_b": iqm_b if mean_b is None else mean_b,
            "significant_mw": sig_mw, "significant_welch": sig_w}


def test_comp_verdict_rules():
    assert exp4._comp_verdict(_comp(8.75, 0.3, True, True)) == "supported"
    assert exp4._comp_verdict(_comp(0.1, 0.3, True, True)) == "refuted"
    assert exp4._comp_verdict(_comp(8.75, 0.3, True, False)) == "boundary"
    assert exp4._comp_verdict(_comp(8.75, 0.3, False, True)) == "boundary"
    assert exp4._comp_verdict(_comp(0.3, 0.3, False, False)) == "null"
    # equal IQMs fall back to means for direction
    assert exp4._comp_verdict(
        _comp(0.3, 0.3, True, True, mean_a=1.9, mean_b=0.3)) == "supported"


def test_combine_verdicts_for_conjunctive_claims():
    comb = exp4._combine_verdicts
    assert comb("supported", "supported") == "supported"
    assert comb("supported", "null") == "boundary"
    assert comb("null", "supported") == "boundary"
    assert comb("refuted", "supported") == "refuted"
    assert comb("supported", "refuted") == "refuted"
    assert comb("null", "null") == "null"
    assert comb("boundary", "supported") == "boundary"


def test_primary_and_robustness_families_cover_the_mandated_comparisons():
    # v3 enlarges the primary family (m=9) with the two reset-replay
    # identifiability controls; the tie-break robustness family is unchanged
    assert exp4.PRIMARY_COMPARISONS == [
        ("generational-distill", "weight-copy"),
        ("generational-distill", "one-long-life"),
        ("generational-distill", "one-long-life-slow"),
        ("generational-distill", "no-inheritance"),
        ("generational-distill", "random-advice"),
        ("generational-distill", "optimistic-init"),
        ("generational-distill", "reset-replay-full"),
        ("generational-distill", "reset-replay-100"),
        ("constant-eps-life", "one-long-life"),
    ]
    assert exp4.ROBUSTNESS_COMPARISONS == [
        ("generational-distill-shortest", "weight-copy"),
        ("generational-distill-shortest", "one-long-life"),
        ("generational-distill-shortest", "one-long-life-slow"),
        ("generational-distill-shortest", "no-inheritance"),
        ("generational-distill-shortest", "random-advice"),
        ("generational-distill-shortest", "optimistic-init"),
        ("generational-distill", "generational-distill-shortest"),
    ]


def test_aggregate_meets_output_contract(tmp_path):
    results = [exp4._run_seed(s, TINY, offset=100) for s in range(2)]
    out = exp4._aggregate(results, TINY, seeds=2, wall_s=0.0, offset=100)
    for key in ("experiment", "hypothesis", "config", "conditions", "curves",
                "tests", "conclusion", "viz"):
        assert key in out
    assert out["experiment"] == "exp4_generations"
    assert out["config"]["seed_offset"] == 100
    assert out["config"]["eval_protocol"]
    # v3 budget-symmetry accounting documented in config: replay updates are
    # free (no env steps) exactly the way advice priming is free
    note = out["config"]["free_inheritance"]
    assert "no env steps" in note and "advice priming" in note
    # per-seed replay doses surface in the JSON (the raw log itself never does)
    rf = out["conditions"]["reset-replay-full"]["seeds"][0]
    assert rf["replayed_by_gen"] == [0] + [TINY["life"]] * (TINY["gens"] - 1)
    r100 = out["conditions"]["reset-replay-100"]["seeds"][0]
    assert r100["replayed_by_gen"] == [0] + [TINY["replay_dose"]] * (TINY["gens"] - 1)
    # curves: checkpoint steps + IQM + CI bounds for every condition
    gr = out["curves"]["greedy_return"]
    assert gr["steps"] == [200, 400, 600, 800, 1000]
    for cond in exp4.CONDITIONS:
        block = gr["conditions"][cond]
        for arr in (block["iqm"], block["ci_lo"], block["ci_hi"], block["mean"]):
            assert len(arr) == TINY["gens"] and np.all(np.isfinite(arr))
    # binary metric flagged so the site plots the mean, not a trimmed proportion
    assert out["curves"]["big_goal"]["stat_note"]
    # per-seed raw metrics present, seed numbers carry the offset
    assert len(out["conditions"]["weight-copy"]["seeds"]) == 2
    assert out["conditions"]["weight-copy"]["seeds"][0]["seed"] == 100
    # tests: dual-test primary family + tie-break robustness family
    comps = out["tests"]["comparisons"]
    assert [(c["a"], c["b"]) for c in comps] == exp4.PRIMARY_COMPARISONS
    rob = out["tests"]["robustness_comparisons"]
    assert [(c["a"], c["b"]) for c in rob] == exp4.ROBUSTNESS_COMPARISONS
    for c in comps + rob:
        for key in ("iqm_a", "iqm_b", "mean_a", "mean_b", "u", "p_mw",
                    "p_mw_holm", "t", "p_welch", "p_welch_holm",
                    "significant_mw", "significant_welch", "significant"):
            assert key in c
        assert c["p"] == c["p_mw"] and c["p_holm"] == c["p_mw_holm"]  # aliases
        assert c["p_mw_holm"] >= c["p_mw"] - 1e-12
        assert c["p_welch_holm"] >= c["p_welch"] - 1e-12
        assert c["significant"] == (c["significant_mw"] and c["significant_welch"])
    # conclusion: per-claim verdicts replace the old boolean
    con = out["conclusion"]
    assert "supported" not in con
    assert [c["claim"] for c in con["claims"]] == list(exp4.CLAIM_NAMES)
    for c in con["claims"]:
        assert c["verdict"] in VERDICTS and c["evidence"]
    assert con["summary"]
    # claim 5 evidence names both replay controls and carries the registered
    # interpretation sentence for its verdict pair
    ev5 = con["claims"][4]["evidence"]
    assert "reset-replay-full" in ev5 and "reset-replay-100" in ev5
    assert any(phrase in ev5 for phrase in
               ("content selection", "dose efficiency", "not identifiable",
                "fails", "matched dose"))
    # descriptive guard against the "replay arms never found the goal"
    # misreading: goal-bearing hand-off counts are reported for both replay
    # arms alongside distill's
    assert "Discovery-vs-consolidation" in ev5
    # viz: layout, curves, advice trajectories, paths for ALL conditions
    viz = out["viz"]
    grid = viz["grid"]
    assert grid["start"] == [5, 1] and grid["goal"] == [5, 13]
    assert len(grid["candies"]) == 3 and len(grid["map"]) == 11
    assert viz["representative_seed"] in (100, 101)
    assert len(viz["advice_by_generation"]) == TINY["gens"]
    for gen_adv in viz["advice_by_generation"]:
        for p in gen_adv["pairs"]:
            assert set(p) == {"s", "rc", "a", "action"}
    assert set(viz["final_greedy_paths"]) == set(exp4.CONDITIONS)
    assert set(viz["per_gen_representative"]) == set(exp4.CONDITIONS)
    assert set(viz["condition_labels"]) == set(exp4.CONDITIONS)
    # whole object survives a STRICT JSON round-trip via the shared encoder
    from devrl.run import save_json
    save_json(tmp_path / "exp4.json", out)
    text = (tmp_path / "exp4.json").read_text()
    assert "Infinity" not in text and "NaN" not in text
    loaded = json.loads(text)
    assert loaded["viz"]["grid"]["goal"] == [5, 13]


def test_claims_are_the_five_mandated_ones():
    assert exp4.CLAIM_NAMES == (
        "peak-experience distillation ratchets across generations",
        "the bottleneck beats weight copying",
        "advice content matters beyond optimism scatter",
        "plasticity decay is what strands the long life",
        "curated advice beats raw experience inheritance at matched plasticity")


def test_replay_story_covers_every_registered_interpretation():
    # the honesty clause is registered as a fixed map from the two comparison
    # verdicts to the interpretation sentence — including the narrowing to
    # dose efficiency when the full raw log matches curated advice
    story = exp4._replay_story
    assert "content selection" in story("supported", "supported")
    for v_full in ("null", "boundary"):
        assert "dose efficiency" in story(v_full, "supported")
        assert "not identifiable" in story(v_full, "null")
        assert "not identifiable" in story(v_full, "boundary")
        assert "not identifiable" in story(v_full, "refuted")
    for v_100 in ("supported", "null", "boundary", "refuted"):
        assert "fails" in story("refuted", v_100)
    assert "matched dose" in story("supported", "null")
    assert "matched dose" in story("supported", "boundary")
    assert "matched dose" in story("supported", "refuted")


def test_smoke_aggregate_carries_nonconfirmatory_note():
    results = [exp4._run_seed(s, TINY) for s in range(2)]
    smoke = exp4._aggregate(results, TINY, seeds=2, wall_s=0.0, smoke=True)
    assert "not confirmatory" in smoke["conclusion"]["note"]
    assert smoke["config"]["smoke"] is True
    full = exp4._aggregate(results, TINY, seeds=2, wall_s=0.0)
    assert "note" not in full["conclusion"]
    assert full["config"]["smoke"] is False
