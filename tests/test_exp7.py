"""EXP7 protocol tests: config pins against the original experiments, the
registered replication decision rule, stat helpers, budget accounting,
machinery equivalence with EXP4 on the original map, and the output
contract of the layout-resampling robustness study."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from devrl.envs.gridhome import GridHome
from devrl.envs.trapgrid import TRAP_MAP, TrapGrid
from devrl.run import save_json

_EXP = Path(__file__).resolve().parents[1] / "experiments"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp7 = _load("exp7_layouts")
exp4 = _load("exp4_generations")
exp1 = _load("exp1_blindfold")

VERDICTS = {"supported", "refuted", "null", "boundary"}

TINY4 = dict(exp7.FULL4, instances=2, seeds=2, life=200, halflife=66,
             n_boot=100)
TINY1 = dict(exp7.FULL1, instances=2, seeds=2, train_steps=600,
             eval_every=200, eval_episodes=4, blind_episodes=4, n_boot=100)


# ------------------------------------------------------------- config pins

def test_part1_config_matches_exp4_machinery_constants():
    c = exp7.FULL4
    assert c["instances"] == 5 and c["seeds"] == 20
    for key in ("gens", "life", "halflife", "lr0", "eps0", "gamma", "cap",
                "memory_k", "advice_cap", "advice_value", "n_boot"):
        assert c[key] == exp4.FULL[key], key
    assert c["gens"] * c["life"] == 75000  # matched total budget per arm


def test_part2_config_matches_exp1_machinery_constants():
    c = exp7.FULL1
    x1 = exp1.make_config(smoke=False)
    assert c["instances"] == 5 and c["seeds"] == 15
    for key in ("slip", "gamma", "lr", "eps", "optimistic_init",
                "planning_steps", "cap", "eval_episodes", "blind_episodes",
                "threshold", "train_steps"):
        assert c[key] == x1[key], key
    # deliberate deviation: coarser curve cadence (10 checkpoints), since the
    # sample-efficiency claim is not re-litigated here
    assert c["eval_every"] == 4000


def test_conditions_and_headlines_registered():
    assert exp7.EXP4_CONDS == ("generational-distill", "no-inheritance")
    assert exp7.EXP1_CONDS == ("sighted-A", "blind-A-touch", "blind-B-touch",
                               "random-B")
    assert exp7.DEFAULT_SEED_OFFSET == 100
    assert "4/5" in exp7.REPLICATION_RULE
    assert "0.05" in exp7.REPLICATION_RULE
    assert "uncorrected" in exp7.REPLICATION_RULE


# --------------------------------------------------- registered decision rule

@pytest.mark.parametrize("n_rep,n_rev,want", [
    (5, 0, "supported"),
    (4, 0, "supported"),
    (4, 1, "supported"),   # the task-registered >=4/5 rule takes precedence
    (3, 0, "boundary"),
    (3, 1, "boundary"),
    (2, 0, "boundary"),
    (2, 1, "boundary"),
    (1, 0, "null"),
    (0, 0, "null"),
    (1, 1, "refuted"),
    (0, 1, "refuted"),
    (0, 5, "refuted"),
])
def test_verdict_from_counts(n_rep, n_rev, want):
    assert exp7.verdict_from_counts(n_rep, n_rev) == want


# ------------------------------------------------------------- stat helpers

def test_holm_adjustment_known_values():
    assert exp7.holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert exp7.holm([0.03, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])


def test_mw_and_welch_degenerate_guards():
    assert exp7.mw_safe([1.0, 1.0], [1.0, 1.0])["p"] == 1.0
    w = exp7.welch_safe([2.0, 2.0], [2.0, 2.0])
    assert w["p"] == 1.0 and w["t"] == 0.0
    w2 = exp7.welch_safe([1.0, 1.0], [2.0, 2.0])
    assert w2["p"] == 0.0 and w2["t"] is None
    a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    got = exp7.mw_safe(a, b)
    ref = mannwhitneyu(a, b, alternative="two-sided")
    assert got["u"] == pytest.approx(float(ref.statistic))
    assert got["p"] == pytest.approx(float(ref.pvalue))


def test_rank_biserial_effect_size():
    assert exp7.rank_biserial([5.0, 6.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert exp7.rank_biserial([1.0, 2.0], [5.0, 6.0]) == pytest.approx(-1.0)
    assert exp7.rank_biserial([3.0, 3.0], [3.0, 3.0]) == pytest.approx(0.0)


def test_instance_entry_flags_replication_and_reversal():
    up = exp7.instance_entry(0, [10.0, 10.0, 10.0, 9.0], [0.3, 0.3, 0.0, 0.3])
    assert up["replicates"] and not up["reversed_significant"]
    down = exp7.instance_entry(0, [0.3, 0.3, 0.0, 0.3],
                               [10.0, 10.0, 10.0, 9.0])
    assert not down["replicates"] and down["reversed_significant"]
    flat = exp7.instance_entry(0, [1.0, 1.0], [1.0, 1.0])
    assert not flat["replicates"] and not flat["reversed_significant"]
    for key in ("instance", "iqm_a", "iqm_b", "mean_a", "mean_b", "u", "p_mw",
                "t", "p_welch", "rank_biserial", "replicates",
                "reversed_significant"):
        assert key in up


# ------------------------------------------------- part 1 machinery (EXP4)

def test_lineage_budget_matched_and_shapes():
    for cond in exp7.EXP4_CONDS:
        out = exp7._lineage(cond, TRAP_MAP, TINY4,
                            np.random.default_rng([0, 0]))
        assert out["steps_trained"] == TINY4["gens"] * TINY4["life"]
        assert len(out["per_gen"]) == TINY4["gens"]
        assert out["final_greedy_return"] == \
            out["per_gen"][-1]["greedy_return"]
        for m in out["per_gen"]:
            assert set(m) == {"greedy_return", "big_goal"}
    d = exp7._lineage("generational-distill", TRAP_MAP, TINY4,
                      np.random.default_rng([0, 0]))
    assert len(d["advice_len_by_gen"]) == TINY4["gens"]
    assert all(0 < n <= TINY4["advice_cap"] for n in d["advice_len_by_gen"])


def test_lineage_reproduces_exp4_machinery_on_original_map():
    # same rng stream + same map -> bit-identical per-generation greedy
    # returns as the original exp4 code path (the replication machinery IS
    # the original machinery, only the map is a parameter)
    cfg4 = dict(exp4.FULL, life=200, halflife=66, slow_halflife=333)
    mine_cfg = dict(TINY4)
    for cond in exp7.EXP4_CONDS:
        ref = exp4._run_condition(cond, np.random.default_rng([7, 3]), cfg4)
        mine = exp7._lineage(cond, TRAP_MAP, mine_cfg,
                             np.random.default_rng([7, 3]))
        assert ([m["greedy_return"] for m in mine["per_gen"]]
                == [m["greedy_return"] for m in ref["per_gen"]])
        assert ([m["big_goal"] for m in mine["per_gen"]]
                == [m["big_goal"] for m in ref["per_gen"]])


def test_unit4_is_deterministic_and_carries_ids():
    maps = [TRAP_MAP, TRAP_MAP]
    a = exp7._unit4(3, maps=maps, cfg=TINY4, offset=100)
    b = exp7._unit4(3, maps=maps, cfg=TINY4, offset=100)
    assert a == b
    assert a["instance"] == 1 and a["seed"] == 101  # instance-major layout
    for cond in exp7.EXP4_CONDS:
        assert a[cond]["steps_trained"] == TINY4["gens"] * TINY4["life"]


# ------------------------------------------------- part 2 machinery (EXP1)

def test_train_dyna_budget_and_eval_schedule():
    from devrl.agents.touchnav import TouchDynaQ
    pair = exp7.gen_instances(0, 1)[1][0]
    agent, ckpts, curve = exp7._train_dyna(pair[0], TINY1, seed=100, inst=0)
    assert isinstance(agent, TouchDynaQ)
    assert agent.age == TINY1["train_steps"]  # env-step budget exact
    assert ckpts == [200, 400, 600]
    assert len(curve) == 3 and all(0.0 <= v <= 1.0 for v in curve)
    assert agent.bump_visits.sum() == TINY1["train_steps"]  # touch recorded


def test_unit1_structure_and_determinism():
    pairs = [exp7.gen_instances(0, 1)[1][0]]
    a = exp7._unit1(1, pairs=pairs, cfg=TINY1, offset=100)
    b = exp7._unit1(1, pairs=pairs, cfg=TINY1, offset=100)
    assert a == b
    assert a["instance"] == 0 and a["seed"] == 101
    assert set(a["blind"]) == set(exp7.EXP1_CONDS)
    for cond in exp7.EXP1_CONDS:
        m = a["blind"][cond]
        assert 0.0 <= m["success_rate"] <= 1.0
        assert 0.0 < m["mean_steps"] <= TINY1["cap"]


def test_gen_instances_is_deterministic_and_valid():
    traps_a, homes_a = exp7.gen_instances(2, 1)
    traps_b, homes_b = exp7.gen_instances(2, 1)
    assert traps_a == traps_b and homes_a == homes_b
    assert len(traps_a) == 2 and len(homes_a) == 1
    assert traps_a[0] != traps_a[1]  # instances differ
    for m in traps_a:
        assert len(TrapGrid(m).candies) == 3
    ma, mb = homes_a[0]
    assert GridHome(ma, rng=np.random.default_rng(0)).shortest_path_len() \
        is not None
    assert ma != mb


# --------------------------------------------------------- output contract

@pytest.fixture(scope="module")
def tiny_out():
    traps, homes = exp7.gen_instances(TINY4["instances"], TINY1["instances"])
    n4 = TINY4["instances"] * TINY4["seeds"]
    n1 = TINY1["instances"] * TINY1["seeds"]
    r4 = [exp7._unit4(u, maps=traps, cfg=TINY4, offset=100)
          for u in range(n4)]
    r1 = [exp7._unit1(u, pairs=homes, cfg=TINY1, offset=100)
          for u in range(n1)]
    return exp7.aggregate(r4, r1, traps, homes, TINY4, TINY1, offset=100,
                          smoke=False, wall=1.0)


def test_output_contract_top_level(tiny_out):
    for key in ("experiment", "hypothesis", "config", "conditions", "curves",
                "tests", "conclusion", "viz"):
        assert key in tiny_out, key
    assert tiny_out["experiment"] == "exp7_layouts"
    cfg = tiny_out["config"]
    assert cfg["gen_seed"] == exp7.GEN_SEED
    assert cfg["seed_offset"] == 100 and cfg["smoke"] is False
    assert cfg["exp4_replication"]["total_budget_per_condition"] == \
        TINY4["gens"] * TINY4["life"]
    assert cfg["exp1_replication"]["train_steps"] == TINY1["train_steps"]


def test_output_contract_tests_block(tiny_out):
    t = tiny_out["tests"]
    assert t["rule"] == exp7.REPLICATION_RULE
    for part, n_inst in (("exp4", TINY4["instances"]),
                         ("exp1", TINY1["instances"])):
        per = t["per_instance"][part]
        assert len(per) == n_inst
        for e in per:
            for key in ("instance", "iqm_a", "iqm_b", "u", "p_mw", "p_welch",
                        "rank_biserial", "replicates", "reversed_significant"):
                assert key in e, key
    pooled = t["pooled"]
    assert [e["name"] for e in pooled] == \
        ["exp4_pooled_distill_gt_noinherit", "exp1_pooled_touchA_gt_touchB"]
    for e in pooled:
        for key in ("iqm_a", "iqm_b", "u", "p_mw", "p_mw_holm", "t",
                    "p_welch", "p_welch_holm", "significant"):
            assert key in e, key


def test_output_contract_conclusion_and_conditions(tiny_out):
    claims = tiny_out["conclusion"]["claims"]
    assert len(claims) == 2
    for c in claims:
        assert c["verdict"] in VERDICTS
        assert c["claim"] and c["evidence"]
    assert isinstance(tiny_out["conclusion"]["summary"], str)

    c4 = tiny_out["conditions"]["exp4_replication"]
    for cond in exp7.EXP4_CONDS:
        inst_blocks = c4[cond]["per_instance"]
        assert len(inst_blocks) == TINY4["instances"]
        for blk in inst_blocks:
            assert len(blk["seeds"]) == TINY4["seeds"]
            for row in blk["seeds"]:
                assert "seed" in row and "final_greedy_return" in row
    c1 = tiny_out["conditions"]["exp1_replication"]
    for cond in exp7.EXP1_CONDS:
        inst_blocks = c1[cond]["per_instance"]
        assert len(inst_blocks) == TINY1["instances"]
        for blk in inst_blocks:
            assert len(blk["seeds"]) == TINY1["seeds"]
            for row in blk["seeds"]:
                assert "seed" in row and "success_rate" in row


def test_output_contract_curves_and_viz(tiny_out):
    c4 = tiny_out["curves"]["exp4_replication"]
    assert c4["steps"] == [TINY4["life"] * (g + 1)
                          for g in range(TINY4["gens"])]
    for cond in exp7.EXP4_CONDS:
        band = c4["conditions"][cond]
        for key in ("iqm", "ci_lo", "ci_hi", "big_goal_mean"):
            assert len(band[key]) == TINY4["gens"]
    c1 = tiny_out["curves"]["exp1_replication"]
    assert c1["checkpoints"] == [200, 400, 600]
    band = c1["conditions"]["dynaq-A"]
    assert len(band["iqm"]) == 3 == len(band["ci_lo"]) == len(band["ci_hi"])

    viz = tiny_out["viz"]
    assert len(viz["trapgrids"]) == TINY4["instances"]
    for tg in viz["trapgrids"]:
        assert len(tg["ascii"]) == 11 and all(len(r) == 15
                                              for r in tg["ascii"])
        assert len(tg["candies"]) == 3
        assert tg["shortest_safe_path"] >= 12
    assert len(viz["home_pairs"]) == TINY1["instances"]
    for hp in viz["home_pairs"]:
        assert len(hp["A"]["ascii"]) == 11 and len(hp["B"]["ascii"]) == 11
        assert 10 <= hp["A"]["shortest_path_len"] <= 25
        assert 10 <= hp["B"]["shortest_path_len"] <= 25
        assert hp["wall_diff"] >= 10
    assert len(viz["per_instance_effects"]["exp4"]) == TINY4["instances"]
    assert len(viz["per_instance_effects"]["exp1"]) == TINY1["instances"]
    for cond in exp7.EXP1_CONDS:
        assert len(viz["blind_summary_pooled"][cond]["success_ci"]) == 2


def test_output_json_serializable(tiny_out, tmp_path):
    save_json(tmp_path / "exp7.json", tiny_out)
    loaded = json.loads((tmp_path / "exp7.json").read_text())
    assert loaded["experiment"] == "exp7_layouts"


def test_smoke_stamp_marks_non_confirmatory(tiny_out):
    traps, homes = exp7.gen_instances(1, 1)
    cfg4 = dict(TINY4, instances=1, seeds=2)
    cfg1 = dict(TINY1, instances=1, seeds=2)
    r4 = [exp7._unit4(u, maps=traps, cfg=cfg4, offset=100) for u in range(2)]
    r1 = [exp7._unit1(u, pairs=homes, cfg=cfg1, offset=100) for u in range(2)]
    out = exp7.aggregate(r4, r1, traps, homes, cfg4, cfg1, offset=100,
                         smoke=True, wall=1.0)
    assert out["config"]["smoke"] is True
    assert "not confirmatory" in out["conclusion"]["note"]
