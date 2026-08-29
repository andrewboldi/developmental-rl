import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import ttest_ind

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from devrl.agents.linearq import LinearQ
from devrl.envs.piano import PianoPiece
from experiments.exp3_variation import (BLOCK_ORDERS, CONDITIONS,
                                        CONTROL_CONDITIONS, MAIN_CONDITIONS,
                                        _seed_job, aggregate, claim_verdict,
                                        greedy_score, holm, make_schedule,
                                        mechanism_verdict, run_seed, welch)


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

def test_block_orders_are_the_six_permutations():
    assert BLOCK_ORDERS == tuple(itertools.permutations(range(3)))


def test_blocked_schedule_is_three_contiguous_phases():
    s = make_schedule("blocked", 90, np.random.default_rng(0))
    assert len(s) == 90
    assert list(s[:30]) == [0] * 30
    assert list(s[30:60]) == [1] * 30
    assert list(s[60:]) == [2] * 30


def test_blocked_schedule_respects_order():
    # v2 counterbalancing: the block order is a parameter
    s = make_schedule("blocked", 9, np.random.default_rng(0), order=(2, 0, 1))
    assert list(s) == [2, 2, 2, 0, 0, 0, 1, 1, 1]


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


def test_local_features_keep_novel_passage_q_at_zero():
    # The mechanism-control guarantee: with pair-onehot features, training on
    # passages 0-2 provably never touches any novel-passage weight, so
    # nomotif transfer sits at exact chance (greedy over all-zero Q).
    ss = np.random.SeedSequence(3)
    env = PianoPiece(rng=np.random.default_rng(ss), feature_map="local")
    ag = LinearQ(env.n_features, env.n_actions, lr=0.4, gamma=0.9, eps=0.1,
                 rng=np.random.default_rng(0))
    for ep in range(30):
        s = env.reset(ep % 3)
        done = False
        while not done:
            phi = env.features(*s)
            a = ag.act(phi)
            s2, r, done, _ = env.step(a)
            ag.update(phi, a, r, None if done else env.features(*s2), done)
            s = s2
    assert np.abs(ag.W).sum() > 0.0  # training did learn something
    for i in range(env.passage_len):
        assert np.all(ag.q(env.features(3, i)) == 0.0)


# ---------------------------------------------------------------- run_seed

def test_run_seed_has_all_four_conditions():
    r = run_seed(0, episodes=6, eval_every=3)
    assert set(CONDITIONS) <= set(r)
    assert MAIN_CONDITIONS == ("blocked", "interleaved")
    assert CONTROL_CONDITIONS == ("blocked-nomotif", "interleaved-nomotif")


def test_block_order_counterbalanced_by_seed_mod_six():
    for seed in (0, 1, 4, 5, 6, 103):
        r = run_seed(seed, episodes=6, eval_every=3)
        assert r["block_order"] == list(BLOCK_ORDERS[seed % 6])


def test_budgets_match_exactly_and_eval_is_free():
    res = run_seed(0, episodes=30, eval_every=10)
    steps = {res[c]["train_steps"] for c in CONDITIONS}
    # eval rollouts happened at every checkpoint but consumed zero budget
    assert steps == {30 * 12}
    for c in CONDITIONS:
        assert res[c]["checkpoint_episodes"] == [0, 10, 20, 30]


def test_run_seed_is_deterministic():
    a = run_seed(3, episodes=30, eval_every=15)
    b = run_seed(3, episodes=30, eval_every=15)
    assert a == b  # pure-python payload, bitwise reproducible


def test_seed_job_applies_offset():
    # --seed-offset support: pool index i runs true seed i + offset
    assert _seed_job(3, offset=100, episodes=6, eval_every=3) == \
        run_seed(103, episodes=6, eval_every=3)


def test_run_seed_metric_shapes_and_ranges():
    r = run_seed(0, episodes=30, eval_every=10)
    assert set(r["structure"]) >= {"motif_table", "passages", "exceptions",
                                   "correct_keys"}
    assert sorted(r["block_order"]) == [0, 1, 2]
    for cond in CONDITIONS:
        c = r[cond]
        assert len(c["acquisition_curve"]) == 4
        assert len(c["eval_scores"]) == 4
        assert all(len(row) == 4 for row in c["eval_scores"])
        assert all(0.0 <= v <= 1.0 for row in c["eval_scores"] for v in row)
        assert set(c["retention"]) == {"A", "B", "C", "mean", "last", "earlier"}
        assert 0.0 <= c["transfer"] <= 1.0
        assert len(c["practice_curve"]) == 3
        assert set(c["final_rollouts"]) == {"A", "B", "C", "novel"}
        assert all(len(v) == 12 for v in c["final_rollouts"].values())


def test_acquisition_tracks_currently_practiced_passage():
    r = run_seed(1, episodes=30, eval_every=10)
    order = r["block_order"]
    assert order == list(BLOCK_ORDERS[1])  # (0, 2, 1) — a non-identity order
    b = r["blocked"]  # phases follow the counterbalanced order
    ev = b["eval_scores"]
    assert b["acquisition_curve"][1] == ev[1][order[0]]
    assert b["acquisition_curve"][2] == ev[2][order[1]]
    assert b["acquisition_curve"][3] == ev[3][order[2]]
    i = r["interleaved"]  # practices all three at once
    assert i["acquisition_curve"][2] == pytest.approx(
        np.mean(i["eval_scores"][2][:3]))


def test_retention_and_transfer_come_from_final_checkpoint():
    r = run_seed(2, episodes=30, eval_every=10)
    for cond in CONDITIONS:
        c = r[cond]
        last = c["eval_scores"][-1]
        assert c["retention"]["A"] == last[0]
        assert c["retention"]["C"] == last[2]
        assert c["retention"]["mean"] == pytest.approx(np.mean(last[:3]))
        assert c["transfer"] == last[3]


def test_retention_last_and_earlier_are_order_relative():
    r = run_seed(4, episodes=30, eval_every=10)  # order (2, 0, 1)
    o = r["block_order"]
    assert o == [2, 0, 1]
    for cond in CONDITIONS:  # same per-seed order aligns all conditions
        c = r[cond]
        final = c["eval_scores"][-1]
        assert c["retention"]["last"] == final[o[2]]
        assert c["retention"]["earlier"] == pytest.approx(
            np.mean([final[o[0]], final[o[1]]]))


# ---------------------------------------------------------------- stats/output

def test_holm_adjustment_hand_computed():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])  # capped at 1


def test_welch_matches_scipy_and_handles_degenerate_groups():
    a, b = [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 6.0]
    got = welch(a, b)
    ref = ttest_ind(a, b, equal_var=False)
    assert got["t"] == pytest.approx(float(ref.statistic))
    assert got["p"] == pytest.approx(float(ref.pvalue))
    # both groups constant and equal (e.g. saturated retention): no evidence
    same = welch([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert same["p"] == 1.0 and np.isfinite(same["t"])
    # both constant but different: maximal evidence, still JSON-finite
    diff = welch([1.0, 1.0, 1.0], [0.5, 0.5, 0.5])
    assert diff["p"] == 0.0 and np.isfinite(diff["t"])


def _fake_test(direction_ok, sig_mw, sig_welch):
    return {"direction_ok": direction_ok, "significant_mw": sig_mw,
            "significant_welch": sig_welch}


def test_claim_verdict_registered_rules():
    assert claim_verdict(_fake_test(True, True, True)) == "supported"
    assert claim_verdict(_fake_test(True, False, False)) == "boundary"
    assert claim_verdict(_fake_test(True, True, False)) == "boundary"
    assert claim_verdict(_fake_test(True, False, True)) == "boundary"
    assert claim_verdict(_fake_test(False, True, True)) == "refuted"
    assert claim_verdict(_fake_test(False, False, False)) == "null"
    assert claim_verdict(_fake_test(False, True, False)) == "null"


def _fake_mech(p_ret, pw_ret, p_tra, pw_tra, tra_b=0.125, tra_i=0.125,
               ret_b=1.0, ret_i=1.0):
    return {
        "nomotif_retention_gap": {
            "p": p_ret, "p_welch": pw_ret,
            "iqm": {"blocked-nomotif": ret_b, "interleaved-nomotif": ret_i}},
        "nomotif_transfer_gap": {
            "p": p_tra, "p_welch": pw_tra,
            "iqm": {"blocked-nomotif": tra_b, "interleaved-nomotif": tra_i}},
    }


def test_mechanism_verdict_registered_rules():
    # crossover vanishes: gaps n.s. on both tests, transfer at chance
    assert mechanism_verdict(_fake_mech(0.8, 0.9, 0.5, 0.6)) == "supported"
    # crossover survives without shared slots: doubly significant gap in the
    # main-pair direction (interleaved > blocked) refutes the mechanism claim
    assert mechanism_verdict(
        _fake_mech(0.001, 0.002, 0.5, 0.6, ret_b=0.8, ret_i=0.95)) == "refuted"
    assert mechanism_verdict(
        _fake_mech(0.8, 0.9, 0.001, 0.002, tra_b=0.2, tra_i=0.4)) == "refuted"
    # off-chance transfer, or a one-test-only / reversed gap: boundary
    assert mechanism_verdict(
        _fake_mech(0.8, 0.9, 0.5, 0.6, tra_b=0.4, tra_i=0.42)) == "boundary"
    assert mechanism_verdict(_fake_mech(0.001, 0.9, 0.5, 0.6)) == "boundary"
    assert mechanism_verdict(
        _fake_mech(0.001, 0.002, 0.5, 0.6, ret_b=0.95, ret_i=0.8)) == "boundary"


def test_output_contract_keys_and_json_clean():
    results = [run_seed(s, episodes=12, eval_every=6) for s in range(2)]
    out = aggregate(results, {"n_boot": 200})
    for key in ("experiment", "hypothesis", "conditions", "curves", "tests",
                "conclusion", "viz"):
        assert key in out
    assert out["experiment"] == "exp3_variation"
    assert set(out["conditions"]) == set(CONDITIONS)
    assert len(out["conditions"]["blocked"]["seeds"]) == 2
    s0 = out["conditions"]["interleaved"]["seeds"][0]
    for key in ("seed", "block_order", "retention", "transfer",
                "acquisition_mean", "acquisition_curve", "eval_scores",
                "train_steps", "final_rollouts"):
        assert key in s0
    # three Holm-corrected primaries, each with BOTH MW and Welch p-values
    prim = [t for t in out["tests"].values() if t["family"] == "primary"]
    assert len(prim) == 3
    for t in prim:
        for key in ("u", "p", "t", "p_welch", "p_holm", "p_holm_welch",
                    "significant_mw", "significant_welch", "significant"):
            assert key in t
        assert t["significant"] == (t["significant_mw"]
                                    and t["significant_welch"])
    # order-relative secondaries replace the per-passage ones; blocked is the
    # DESIGN-predicted winner on the just-drilled passage (recency)
    sec = {k: t for k, t in out["tests"].items() if t["family"] == "secondary"}
    assert set(sec) == {"retention_last_blocked_gt_interleaved",
                        "retention_earlier_interleaved_gt_blocked"}
    assert sec["retention_last_blocked_gt_interleaved"][
        "predicted_winner"] == "blocked"
    assert "recency" in sec["retention_last_blocked_gt_interleaved"]["note"]
    for t in sec.values():
        assert "p" in t and "p_welch" in t
    # mechanism-control family: nomotif pair comparisons, prediction = no gap
    mech = {k: t for k, t in out["tests"].items() if t["family"] == "mechanism"}
    assert set(mech) == {"nomotif_acquisition_gap", "nomotif_retention_gap",
                         "nomotif_transfer_gap"}
    for t in mech.values():
        assert set(t["iqm"]) == set(CONTROL_CONDITIONS)
        assert "p" in t and "p_welch" in t and "prediction" in t
    # per-claim verdicts replace the boolean conclusion
    con = out["conclusion"]
    assert "supported" not in con
    assert isinstance(con["summary"], str)
    claims = con["claims"]
    assert [c["name"] for c in claims] == ["acquisition", "retention",
                                           "transfer", "mechanism"]
    for c in claims:
        assert c["verdict"] in {"supported", "refuted", "null", "boundary"}
        assert isinstance(c["claim"], str) and isinstance(c["evidence"], str)
    # curves ride the shared checkpoint grid, now including order-relative
    # retention metrics and the nomotif control conditions
    ck = results[0]["blocked"]["checkpoint_episodes"]
    assert out["curves"]["episodes"] == ck
    assert out["curves"]["steps"] == [e * 12 for e in ck]
    for metric in ("acquisition", "retention_A", "retention_B", "retention_C",
                   "retention_mean", "retention_last", "retention_earlier",
                   "transfer"):
        for cond in CONDITIONS:
            series = out["curves"]["metrics"][metric][cond]
            assert len(series["iqm"]) == len(ck)
            assert len(series["lo"]) == len(series["hi"]) == len(ck)
    # viz has everything the website needs, and the whole payload is
    # pure-python and finite (json refuses NaN/inf and numpy types)
    viz = out["viz"]
    for key in ("example_structure", "retention_bars", "transfer_bars",
                "crossover", "curves", "practice_curves", "example_rollouts",
                "phase_boundaries_episodes", "per_seed_points",
                "block_orders", "order_relative_bars"):
        assert key in viz
    assert viz["conditions"] == list(CONDITIONS)
    assert viz["main_conditions"] == list(MAIN_CONDITIONS)
    assert viz["control_conditions"] == list(CONTROL_CONDITIONS)
    for cond in CONDITIONS:
        assert cond in viz["transfer_bars"]
        assert cond in viz["retention_bars"]
        assert cond in viz["crossover"]
        assert cond in viz["per_seed_points"]
        assert cond in viz["example_rollouts"]
        assert cond in viz["order_relative_bars"]
    assert viz["order_relative_bars"]["labels"] == ["earlier", "last"]
    assert len(viz["block_orders"]) == 2
    assert len(viz["example_structure"]["passages"]) == 4
    assert len(viz["example_structure"]["motif_table"]) == 6
    json.dumps(out, allow_nan=False)
