"""EXP7 — Layout-resampling robustness (critic P7). DESIGN.md
"Amendments (v3) — EXP7".

The two headline results were measured on single fixed layouts, which scopes
their conclusions to those instances (Whiteson et al. 2011; RESEARCH.md
pseudo-replication risk). EXP7 replicates each headline pairing — nothing
else — across freshly generated environment instances:

- Part 1 (EXP4 headline): on 5 generated TrapGrid layouts (standard shell
  and S/G, candy traps resampled in the left third under a random-walk
  absorption calibration guard), run generational-distill vs no-inheritance,
  20 seeds each, the exact 5 x 15k budget and aging/advice machinery of
  EXP4 (the lineage code is verified bit-identical to exp4_generations on
  the original map in tests/test_exp7.py). Metric: final greedy return.
- Part 2 (EXP1 headline): on 5 generated home pairs (original shell and S/G
  coordinates, interior walls from parity-constrained recursive room
  partition, shortest path in [10, 25], pairs differing in >= 10 wall
  cells), train TouchDynaQ 40k env steps in home A, then run the blindfold
  table sighted-A / blind-A-touch / blind-B-touch / random-B on the frozen
  artifacts, 15 seeds each. Metric: blindfold success rate; headline pair
  blind-A-touch vs blind-B-touch.

Registered decision rule (fixed in tests before the confirmatory run):
robustness is supported iff the headline direction holds with per-instance
Mann-Whitney p < 0.05 (uncorrected — these are replications) in >= 4/5
instances of that experiment. Pooled cross-instance stats (both MW and
Welch t, Holm within the m=2 pooled family) are reported alongside but do
not gate the verdict. Budgets are matched to the env step across compared
conditions; eval is budget-free, rng-disjoint from training, and identical
across conditions; confirmatory seeds are offset (default 100).

Usage: python experiments/exp7_layouts.py --out results/exp7_layouts.json \
           [--seeds4 20] [--seeds1 15] [--seed-offset 100] [--smoke] [--jobs 20]
"""

import argparse
import time
from functools import partial

import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind

from devrl.agents.distill import (EpisodicMemory, apply_advice,
                                  extract_advice, halflife_schedule)
from devrl.agents.qlearning import QLearner
from devrl.agents.touchnav import TouchDynaQ, TouchNavigator
from devrl.envs.gridhome import GridHome
from devrl.envs.layouts import (candy_absorption, gen_home_pair,
                                gen_trapgrid, wall_cells)
from devrl.envs.trapgrid import TrapGrid
from devrl.run import run_seeds, save_json
from devrl.stats import bootstrap_ci, iqm, time_to_threshold

GEN_SEED = 202609  # meta-seed for instance generation (disjoint from every
#                    training/eval stream: those use 3-part keys with small
#                    second elements; generation uses [GEN_SEED, 40|10, i])
DEFAULT_SEED_OFFSET = 100

EXP4_CONDS = ("generational-distill", "no-inheritance")
EXP1_CONDS = ("sighted-A", "blind-A-touch", "blind-B-touch", "random-B")
BLIND_TABLE = {"sighted-A": ("A", "state"), "blind-A-touch": ("A", "touch"),
               "blind-B-touch": ("B", "touch"), "random-B": ("B", "random")}

HYPOTHESIS = (
    "EXP7 (robustness): the two headline results — EXP4's "
    "generational-distill > no-inheritance (final greedy return) and EXP1's "
    "blind-A-touch > blind-B-touch (blindfold success) — are properties of "
    "their environment families, not of the single fixed layouts they were "
    "measured on: each replicates on freshly generated instances.")

REPLICATION_RULE = (
    "Registered: robustness for a headline is supported iff its direction "
    "holds with Mann-Whitney p < 0.05 (per-instance, uncorrected — these "
    "are replications, each on its own fresh layout and fresh seeds) in "
    ">= 4/5 generated instances of that experiment. Full verdict map on "
    "(n_rep = instances significant in the headline direction, n_rev = "
    "instances significant in the reversed direction): n_rep >= 4 -> "
    "supported; n_rep in {2, 3} -> boundary; n_rep <= 1 with n_rev >= 1 -> "
    "refuted; else null. Pooled cross-instance MW + Welch t (Holm within "
    "the m=2 pooled family, each statistic separately) are reported "
    "alongside but do not gate the verdict; any significant per-instance "
    "reversal is disclosed in the evidence.")

# Part 1 config: identical machinery constants to exp4_generations.FULL
# (pinned in tests), restricted to the headline pairing at 20 seeds.
FULL4 = dict(instances=5, seeds=20, gens=5, life=15000, halflife=5000,
             lr0=0.3, eps0=0.4, gamma=0.99, cap=120, memory_k=3,
             advice_cap=100, advice_value=5.0, n_boot=10000)
SMOKE4 = dict(FULL4, seeds=2, life=1500, halflife=500, n_boot=2000)

# Part 2 config: identical machinery constants to exp1_blindfold's full
# config (pinned in tests) except eval_every=4000 — a coarser curve cadence,
# since the sample-efficiency claim is not re-litigated here.
FULL1 = dict(instances=5, seeds=15, train_steps=40000, eval_every=4000,
             slip=0.1, gamma=0.97, lr=0.1, eps=0.1, optimistic_init=1.0,
             planning_steps=20, cap=60, eval_episodes=20, blind_episodes=30,
             threshold=0.9, n_boot=10000)
SMOKE1 = dict(FULL1, seeds=2, train_steps=4000, eval_every=400,
              blind_episodes=10, n_boot=2000)

EVAL4_NOTE = (
    "one deterministic greedy rollout per generation boundary (eps=0, argmax "
    "lowest-index tie-break; env and policy deterministic) on the instance's "
    "own map — consumes no training budget, draws no rng, byte-identical "
    "across conditions (exp4's protocol with the map as a parameter)")
EVAL1_NOTE = (
    "training curve: greedy eval every eval_every steps, eval_episodes "
    "episodes with fresh seeded env/tie-break rngs keyed disjointly from "
    "training keys. Blindfold table: frozen artifacts only; env rng streams "
    "keyed by (seed, instance, episode) and SHARED across the four "
    "conditions so each faces the identical slip-noise stream; policy rng "
    "eval-owned; nothing trains")
BUDGET_NOTE_4 = ("both conditions train exactly gens*life env steps per "
                 "(instance, seed); eval rollouts are budget-free")
BUDGET_NOTE_1 = ("one TouchDynaQ trains exactly train_steps env steps per "
                 "(instance, seed); all four blindfold conditions evaluate "
                 "the SAME frozen artifacts, so no cross-condition training "
                 "budget exists; eval and blindfold rollouts are budget-free")

CONDITION_LABELS = {
    "generational-distill": "Generational distill (earliest-tie memory)",
    "no-inheritance": "No inheritance (independent lives)",
    "sighted-A": "Sighted, home A",
    "blind-A-touch": "Blind + touch, home A",
    "blind-B-touch": "Blind + touch, stranger home B",
    "random-B": "Random policy, home B (matched floor)",
    "dynaq-train": "TouchDynaQ training in home A",
}


# ------------------------------------------------------ instance generation

def gen_instances(n_trap, n_home):
    """Deterministic instance sets from the registered meta-seed."""
    traps = [gen_trapgrid(np.random.default_rng([GEN_SEED, 40, i]))
             for i in range(n_trap)]
    homes = [gen_home_pair(np.random.default_rng([GEN_SEED, 10, i]))
             for i in range(n_home)]
    return traps, homes


# ------------------------------------------------- part 1: EXP4 replication

def greedy_rollout_on(Q, map_str, cap):
    """Deterministic greedy rollout on an instance map (exp4's eval with the
    map as a parameter). Returns (undiscounted return, reached-big-goal)."""
    env = TrapGrid(map_str, cap=cap)
    s = env.reset()
    ret = 0.0
    while True:
        s, r, done, info = env.step(int(np.argmax(Q[s])))
        ret += r
        if done or info["truncated"]:
            return float(ret), bool(done and env.pos == env.goal)


def _live(agent, env, memory, n_steps):
    """Exactly n_steps of eps-greedy life (exp4's loop, fresh-reset form):
    completed episodes — terminal or cap-truncated — enter memory."""
    s, sa, ret = env.reset(), [], 0.0
    for _ in range(n_steps):
        a = agent.act(s)
        s2, r, done, info = env.step(a)
        agent.update(s, a, r, s2, done)
        sa.append((s, a))
        ret += r
        if done or info["truncated"]:
            memory.add(ret, sa)
            s, sa, ret = env.reset(), [], 0.0
        else:
            s = s2


def _lineage(cond, map_str, cfg, rng):
    """One 5-generation lineage of the headline pairing on an instance map.

    generational-distill: each fresh student is primed with the previous
    teacher's advice bottleneck; no-inheritance: 5 independent lives. Both
    train exactly gens*life env steps. Machinery constants and call order
    mirror exp4_generations exactly (verified bit-identical on TRAP_MAP)."""
    env = TrapGrid(map_str, cap=cfg["cap"])
    advice, per_gen, advice_lens, steps = None, [], [], 0
    for _ in range(cfg["gens"]):
        agent = QLearner(n_states=env.n_states, n_actions=TrapGrid.n_actions,
                         lr=halflife_schedule(cfg["lr0"], cfg["halflife"]),
                         gamma=cfg["gamma"],
                         eps=halflife_schedule(cfg["eps0"], cfg["halflife"]),
                         optimistic_init=0.0, rng=rng)
        if cond == "generational-distill" and advice:
            apply_advice(agent.Q, advice, cfg["advice_value"])
        memory = EpisodicMemory(cfg["memory_k"])  # earliest-tie (primary)
        _live(agent, env, memory, cfg["life"])
        steps += cfg["life"]
        ret, big = greedy_rollout_on(agent.Q, map_str, cfg["cap"])
        per_gen.append({"greedy_return": ret, "big_goal": int(big)})
        if cond == "generational-distill":
            advice = extract_advice(memory, cfg["advice_cap"])
            advice_lens.append(len(advice))
    out = {"steps_trained": steps, "per_gen": per_gen,
           "final_greedy_return": per_gen[-1]["greedy_return"]}
    if cond == "generational-distill":
        out["advice_len_by_gen"] = advice_lens
    return out


def _unit4(u, maps, cfg, offset):
    """One (instance, seed) work unit of part 1; instance-major order.
    Each condition draws from its own (seed, instance, condition) stream."""
    inst, sidx = divmod(u, cfg["seeds"])
    seed = sidx + offset
    out = {"instance": inst, "seed": seed}
    for ci, cond in enumerate(EXP4_CONDS):
        rng = np.random.default_rng([seed, 800 + inst, ci])
        out[cond] = _lineage(cond, maps[inst], cfg, rng)
    return out


# ------------------------------------------------- part 2: EXP1 replication

def greedy_tiebreak(Q, s, rng):
    q = Q[s]
    best = np.flatnonzero(q == q.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def _eval_greedy(Q, map_str, cfg, seed, inst, ckpt):
    """Greedy success rate on frozen Q; fresh seeded env/rng per episode,
    keys disjoint from every training stream."""
    wins = 0
    for ep in range(cfg["eval_episodes"]):
        env = GridHome(map_str, slip=cfg["slip"],
                       rng=np.random.default_rng([seed, 920 + inst, ckpt, ep]))
        rng = np.random.default_rng([seed, 930 + inst, ckpt, ep])
        s = env.reset()
        for _ in range(cfg["cap"]):
            s, _, done, _ = env.step(greedy_tiebreak(Q, s, rng))
            if done:
                wins += 1
                break
    return wins / cfg["eval_episodes"]


def _train_dyna(map_a, cfg, seed, inst):
    """TouchDynaQ for exactly cfg['train_steps'] env steps in home A
    (exp1's training loop with the map as a parameter)."""
    env = GridHome(map_a, slip=cfg["slip"],
                   rng=np.random.default_rng([seed, 900 + inst]))
    agent = TouchDynaQ(planning_steps=cfg["planning_steps"],
                       n_states=env.n_states, n_actions=env.n_actions,
                       lr=cfg["lr"], gamma=cfg["gamma"], eps=cfg["eps"],
                       optimistic_init=cfg["optimistic_init"],
                       rng=np.random.default_rng([seed, 910 + inst]))
    curve, ckpts = [], []
    s, ep_len = env.reset(), 0
    for t in range(1, cfg["train_steps"] + 1):
        a = agent.act(s)
        s2, r, done, info = env.step(a)
        agent.observe_touch(s, a, info["bump"])
        agent.update(s, a, r, s2, done)
        ep_len += 1
        if done or ep_len >= cfg["cap"]:
            s, ep_len = env.reset(), 0
        else:
            s = s2
        if t % cfg["eval_every"] == 0:
            curve.append(_eval_greedy(agent.Q, map_a, cfg, seed, inst,
                                      len(ckpts)))
            ckpts.append(t)
    return agent, ckpts, curve


def _blind_condition(name, agent, map_a, map_b, cfg, seed, inst):
    """One blindfold-table condition on the frozen home-A artifacts. Env rng
    keys are shared across conditions (identical slip streams); failed
    episodes count cfg['cap'] steps."""
    home, mode = BLIND_TABLE[name]
    map_str = map_a if home == "A" else map_b
    succ, steps_all, steps_succ = [], [], []
    for ep in range(cfg["blind_episodes"]):
        env = GridHome(map_str, slip=cfg["slip"],
                       rng=np.random.default_rng([seed, 940 + inst, ep]))
        policy_rng = np.random.default_rng([seed, 950 + inst, ep])
        s = env.reset()
        nav = (TouchNavigator(agent, agent.Q, s, touch=agent)
               if mode == "touch" else None)
        done, steps = False, cfg["cap"]
        for t in range(cfg["cap"]):
            if mode == "state":
                a = greedy_tiebreak(agent.Q, s, policy_rng)
            elif mode == "random":
                a = int(policy_rng.integers(env.n_actions))
            else:
                a = nav.act()
            s, _, done, info = env.step(a)
            if mode == "touch":
                nav.advance(a, info["bump"])
            if done:
                steps = t + 1
                break
        succ.append(done)
        steps_all.append(steps)
        if done:
            steps_succ.append(steps)
    return {"success_rate": float(np.mean(succ)),
            "mean_steps": float(np.mean(steps_all)),
            "mean_steps_success": (float(np.mean(steps_succ))
                                   if steps_succ else None)}


def _unit1(u, pairs, cfg, offset):
    """One (instance, seed) work unit of part 2; instance-major order."""
    inst, sidx = divmod(u, cfg["seeds"])
    seed = sidx + offset
    map_a, map_b = pairs[inst]
    agent, ckpts, curve = _train_dyna(map_a, cfg, seed, inst)
    blind = {name: _blind_condition(name, agent, map_a, map_b, cfg, seed,
                                    inst)
             for name in EXP1_CONDS}
    return {"instance": inst, "seed": seed, "checkpoints": ckpts,
            "curve": curve,
            "t90": time_to_threshold(ckpts, curve, cfg["threshold"]),
            "final_success": curve[-1], "blind": blind}


# ------------------------------------------------------------------ analysis

def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return [float(v) for v in adj]


def mw_safe(a, b):
    """Two-sided Mann-Whitney; fully tied samples get p=1 instead of NaN."""
    both = np.concatenate([np.asarray(a, float), np.asarray(b, float)])
    if float(np.ptp(both)) == 0.0:
        return {"u": len(a) * len(b) / 2.0, "p": 1.0}
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return {"u": float(u), "p": float(p)}


def welch_safe(a, b):
    """Welch t (scipy ttest_ind, equal_var=False) with degenerate guards:
    both constant and equal -> t=0, p=1; both constant but different ->
    p=0 with t=None (never Infinity, which would break strict JSON)."""
    av, bv = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if av.var() == 0.0 and bv.var() == 0.0:
        if av[0] == bv[0]:
            return {"t": 0.0, "p": 1.0}
        return {"t": None, "p": 0.0}
    t, p = ttest_ind(av, bv, equal_var=False)
    return {"t": float(t), "p": float(p)}


def rank_biserial(a, b):
    """Rank-biserial effect size r = 2*U1/(n1*n2) - 1 in [-1, 1]; positive
    means a stochastically dominates b; 0 for fully tied samples."""
    both = np.concatenate([np.asarray(a, float), np.asarray(b, float)])
    if float(np.ptp(both)) == 0.0:
        return 0.0
    u1 = float(mannwhitneyu(a, b, alternative="two-sided").statistic)
    return 2.0 * u1 / (len(a) * len(b)) - 1.0


def _direction_a_above(ia, ib, ma, mb):
    hi, lo = (ia, ib) if ia != ib else (ma, mb)
    return hi > lo, hi < lo


def instance_entry(inst, a_vals, b_vals):
    """Per-instance replication entry: uncorrected MW (the registered
    replication statistic) + Welch t + rank-biserial effect size. Direction
    is IQM, falling back to means on IQM ties (exp4's convention)."""
    mw, w = mw_safe(a_vals, b_vals), welch_safe(a_vals, b_vals)
    ia, ib = iqm(a_vals), iqm(b_vals)
    ma, mb = float(np.mean(a_vals)), float(np.mean(b_vals))
    above, below = _direction_a_above(ia, ib, ma, mb)
    return {"instance": int(inst), "iqm_a": ia, "iqm_b": ib,
            "mean_a": ma, "mean_b": mb, "u": mw["u"], "p_mw": mw["p"],
            "t": w["t"], "p_welch": w["p"],
            "rank_biserial": rank_biserial(a_vals, b_vals),
            "replicates": bool(mw["p"] < 0.05 and above),
            "reversed_significant": bool(mw["p"] < 0.05 and below)}


def verdict_from_counts(n_rep, n_rev):
    """Registered verdict map (see REPLICATION_RULE)."""
    if n_rep >= 4:
        return "supported"
    if n_rep >= 2:
        return "boundary"
    if n_rev >= 1:
        return "refuted"
    return "null"


def _pooled_entry(name, a_vals, b_vals, a_label, b_label):
    mw, w = mw_safe(a_vals, b_vals), welch_safe(a_vals, b_vals)
    return {"name": name, "a": a_label, "b": b_label,
            "n_a": len(a_vals), "n_b": len(b_vals),
            "iqm_a": iqm(a_vals), "iqm_b": iqm(b_vals),
            "mean_a": float(np.mean(a_vals)),
            "mean_b": float(np.mean(b_vals)),
            "u": mw["u"], "p_mw": mw["p"], "t": w["t"], "p_welch": w["p"]}


def _fmt_inst(entries):
    return "; ".join(
        "inst%d IQM %.2f vs %.2f, MW p=%.3g%s" % (
            e["instance"], e["iqm_a"], e["iqm_b"], e["p_mw"],
            " [REVERSED]" if e["reversed_significant"] else "")
        for e in entries)


def aggregate(results4, results1, trap_maps, home_pairs, cfg4, cfg1, offset,
              smoke, wall):
    boot_rng = np.random.default_rng(202607)
    n_inst4, n_inst1 = len(trap_maps), len(home_pairs)

    def ci(vals, n_boot):
        return list(bootstrap_ci(vals, n_boot=n_boot, rng=boot_rng,
                                 statistic=iqm))

    # ---- part 1 reorganization: [cond][instance] -> per-seed finals ------
    by4 = {cond: [[r[cond] for r in results4 if r["instance"] == i]
                  for i in range(n_inst4)] for cond in EXP4_CONDS}
    finals4 = {cond: [[u["final_greedy_return"] for u in inst]
                      for inst in by4[cond]] for cond in EXP4_CONDS}
    per_inst4 = [instance_entry(i, finals4["generational-distill"][i],
                                finals4["no-inheritance"][i])
                 for i in range(n_inst4)]

    # ---- part 2 reorganization ------------------------------------------
    by1 = [[r for r in results1 if r["instance"] == i]
           for i in range(n_inst1)]
    succ1 = {cond: [[r["blind"][cond]["success_rate"] for r in inst]
                    for inst in by1] for cond in EXP1_CONDS}
    per_inst1 = [instance_entry(i, succ1["blind-A-touch"][i],
                                succ1["blind-B-touch"][i])
                 for i in range(n_inst1)]

    # ---- registered per-instance rule + pooled family --------------------
    counts = {}
    for part, entries in (("exp4", per_inst4), ("exp1", per_inst1)):
        counts[part] = (sum(e["replicates"] for e in entries),
                        sum(e["reversed_significant"] for e in entries))

    pool4_a = [v for inst in finals4["generational-distill"] for v in inst]
    pool4_b = [v for inst in finals4["no-inheritance"] for v in inst]
    pool1_a = [v for inst in succ1["blind-A-touch"] for v in inst]
    pool1_b = [v for inst in succ1["blind-B-touch"] for v in inst]
    pooled = [
        _pooled_entry("exp4_pooled_distill_gt_noinherit", pool4_a, pool4_b,
                      "generational-distill", "no-inheritance"),
        _pooled_entry("exp1_pooled_touchA_gt_touchB", pool1_a, pool1_b,
                      "blind-A-touch", "blind-B-touch"),
    ]
    for e, pmh, pwh in zip(pooled, holm([e["p_mw"] for e in pooled]),
                           holm([e["p_welch"] for e in pooled])):
        e["p_mw_holm"], e["p_welch_holm"] = pmh, pwh
        e["significant"] = bool(pmh < 0.05 and pwh < 0.05)

    tests = {
        "rule": REPLICATION_RULE,
        "alpha": 0.05,
        "metric_exp4": "final greedy return (generation %d)" % cfg4["gens"],
        "metric_exp1": "blindfold success rate",
        "per_instance": {"exp4": per_inst4, "exp1": per_inst1},
        "pooled": pooled,
        "pooled_note": ("pooled entries concatenate per-seed values across "
                        "instances (units = instance x seed); Holm within "
                        "the m=2 pooled family, MW and Welch separately; "
                        "significant requires both Holm'd p < 0.05; "
                        "descriptive — the registered verdict rides on the "
                        "per-instance rule only"),
    }

    # ---- claims ----------------------------------------------------------
    v4 = verdict_from_counts(*counts["exp4"])
    v1 = verdict_from_counts(*counts["exp1"])
    p4, p1 = pooled[0], pooled[1]
    ev4 = ("Replicated in %d/%d instances (significant reversals: %d): %s. "
           "Pooled (n=%d vs %d): IQM %.2f vs %.2f, MW p=%.3g (Holm %.3g), "
           "Welch p=%.3g (Holm %.3g)."
           % (counts["exp4"][0], n_inst4, counts["exp4"][1],
              _fmt_inst(per_inst4), p4["n_a"], p4["n_b"], p4["iqm_a"],
              p4["iqm_b"], p4["p_mw"], p4["p_mw_holm"], p4["p_welch"],
              p4["p_welch_holm"]))
    ev1 = ("Replicated in %d/%d instances (significant reversals: %d): %s. "
           "Pooled (n=%d vs %d): IQM %.2f vs %.2f, MW p=%.3g (Holm %.3g), "
           "Welch p=%.3g (Holm %.3g)."
           % (counts["exp1"][0], n_inst1, counts["exp1"][1],
              _fmt_inst(per_inst1), p1["n_a"], p1["n_b"], p1["iqm_a"],
              p1["iqm_b"], p1["p_mw"], p1["p_mw_holm"], p1["p_welch"],
              p1["p_welch_holm"]))
    claims = [
        {"claim": ("exp4-layout-robustness: generational-distill > "
                   "no-inheritance (final greedy return) is not an artifact "
                   "of the fixed TRAP_MAP — it replicates on >= 4/5 freshly "
                   "generated TrapGrid layouts (per-instance MW p < 0.05, "
                   "uncorrected, direction preserved)"),
         "verdict": v4, "evidence": ev4},
        {"claim": ("exp1-layout-robustness: blind-A-touch > blind-B-touch "
                   "(blindfold success) is not an artifact of the fixed "
                   "HOME_A/HOME_B pair — it replicates on >= 4/5 freshly "
                   "generated home pairs (per-instance MW p < 0.05, "
                   "uncorrected, direction preserved)"),
         "verdict": v1, "evidence": ev1},
    ]
    summary = (
        "EXP4 headline replicated in %d/%d generated TrapGrids (verdict "
        "%s); EXP1 headline replicated in %d/%d generated home pairs "
        "(verdict %s). Pooled: distill IQM %.2f vs no-inheritance %.2f (MW "
        "Holm p=%.3g); blind-A-touch IQM %.2f vs blind-B-touch %.2f (MW "
        "Holm p=%.3g)."
        % (counts["exp4"][0], n_inst4, v4, counts["exp1"][0], n_inst1, v1,
           p4["iqm_a"], p4["iqm_b"], p4["p_mw_holm"],
           p1["iqm_a"], p1["iqm_b"], p1["p_mw_holm"]))
    conclusion = {"claims": claims, "summary": summary}
    if smoke:
        conclusion["note"] = (
            "smoke scale (~1/10 budget, 2 seeds/instance): not confirmatory "
            "— use the full results/exp7_layouts.json for any conclusion")

    # ---- curves ----------------------------------------------------------
    steps4 = [cfg4["life"] * (g + 1) for g in range(cfg4["gens"])]
    curves4 = {}
    for cond in EXP4_CONDS:
        rets = [[r[cond]["per_gen"][g]["greedy_return"] for r in results4]
                for g in range(cfg4["gens"])]
        bigs = [[r[cond]["per_gen"][g]["big_goal"] for r in results4]
                for g in range(cfg4["gens"])]
        cis = [ci(v, cfg4["n_boot"]) for v in rets]
        curves4[cond] = {"iqm": [iqm(v) for v in rets],
                         "mean": [float(np.mean(v)) for v in rets],
                         "ci_lo": [c[0] for c in cis],
                         "ci_hi": [c[1] for c in cis],
                         "big_goal_mean": [float(np.mean(v)) for v in bigs]}
    ckpts1 = results1[0]["checkpoints"]
    per_ck = [[r["curve"][k] for r in results1] for k in range(len(ckpts1))]
    cis1 = [ci(v, cfg1["n_boot"]) for v in per_ck]
    curves1 = {"dynaq-A": {"iqm": [iqm(v) for v in per_ck],
                           "ci_lo": [c[0] for c in cis1],
                           "ci_hi": [c[1] for c in cis1]}}
    curves = {
        "exp4_replication": {
            "steps": steps4, "conditions": curves4,
            "note": ("pooled across all instance x seed units; big_goal is "
                     "a 0/1 variable — plot big_goal_mean (P(goal)), not an "
                     "IQM of it")},
        "exp1_replication": {"checkpoints": ckpts1, "conditions": curves1,
                             "note": "pooled training curve, all units"},
    }

    # ---- conditions (per-seed raw) ---------------------------------------
    cond4 = {cond: {"per_instance": [
        {"instance": i,
         "seeds": [{"seed": r["seed"],
                    "final_greedy_return": r[cond]["final_greedy_return"],
                    "per_gen": r[cond]["per_gen"]}
                   for r in results4 if r["instance"] == i]}
        for i in range(n_inst4)]} for cond in EXP4_CONDS}
    cond1 = {cond: {"per_instance": [
        {"instance": i,
         "seeds": [{"seed": r["seed"], **r["blind"][cond]}
                   for r in by1[i]]}
        for i in range(n_inst1)]} for cond in EXP1_CONDS}
    cond1["dynaq-train"] = {"per_instance": [
        {"instance": i,
         "seeds": [{"seed": r["seed"], "final_success": r["final_success"],
                    "t90": r["t90"], "curve": r["curve"]} for r in by1[i]]}
        for i in range(n_inst1)],
        "t90_censored_frac": float(np.mean(
            [r["t90"] is None for r in results1]))}
    conditions = {"exp4_replication": cond4, "exp1_replication": cond1}

    # ---- viz -------------------------------------------------------------
    trap_viz = []
    for i, m in enumerate(trap_maps):
        env = TrapGrid(m)
        trap_viz.append({
            "instance": i, "ascii": m.strip().splitlines(),
            "start": list(env.start), "goal": list(env.goal),
            "candies": sorted(list(c) for c in env.candies),
            "shortest_safe_path": env.shortest_path_len(),
            "random_walk_candy_absorption": candy_absorption(
                m, np.random.default_rng([GEN_SEED, 99, i]), episodes=2000),
        })
    home_viz = []
    for i, (ma, mb) in enumerate(home_pairs):
        entry = {"instance": i,
                 "wall_diff": len(wall_cells(ma) ^ wall_cells(mb))}
        for key, ms in (("A", ma), ("B", mb)):
            env = GridHome(ms, rng=np.random.default_rng(0))
            entry[key] = {"ascii": ms.strip().splitlines(),
                          "start": [int(env.start[0]), int(env.start[1])],
                          "goal": [int(env.goal[0]), int(env.goal[1])],
                          "shortest_path_len": int(env.shortest_path_len())}
        home_viz.append(entry)

    effects = {part: [{"instance": e["instance"], "iqm_a": e["iqm_a"],
                       "iqm_b": e["iqm_b"],
                       "iqm_diff": e["iqm_a"] - e["iqm_b"],
                       "rank_biserial": e["rank_biserial"],
                       "p_mw": e["p_mw"], "replicates": e["replicates"]}
                      for e in entries]
               for part, entries in (("exp4", per_inst4), ("exp1",
                                                           per_inst1))}

    blind_pooled = {}
    for cond in EXP1_CONDS:
        sv = [r["blind"][cond]["success_rate"] for r in results1]
        st = [r["blind"][cond]["mean_steps"] for r in results1]
        blind_pooled[cond] = {"success_iqm": iqm(sv),
                              "success_ci": ci(sv, cfg1["n_boot"]),
                              "steps_iqm": iqm(st),
                              "steps_ci": ci(st, cfg1["n_boot"])}
    blind_by_inst = [
        {"instance": i,
         **{cond: {"success_iqm": iqm(succ1[cond][i]),
                   "success_ci": ci(succ1[cond][i], cfg1["n_boot"])}
            for cond in EXP1_CONDS}}
        for i in range(n_inst1)]
    gen_curves_by_inst = [
        {"instance": i,
         **{cond: {"iqm": [iqm([u["per_gen"][g]["greedy_return"]
                                for u in by4[cond][i]])
                    for g in range(cfg4["gens"])]} for cond in EXP4_CONDS}}
        for i in range(n_inst4)]

    viz = {
        "headlines": {
            "exp4": "generational-distill vs no-inheritance, final greedy "
                    "return after 5 generations (EXP4 claim 1 pairing)",
            "exp1": "blind-A-touch vs blind-B-touch success rate on frozen "
                    "home-A artifacts (EXP1 A1 primary pairing)"},
        "condition_labels": CONDITION_LABELS,
        "trapgrids": trap_viz,
        "home_pairs": home_viz,
        "per_instance_effects": effects,
        "pooled_curves": curves,
        "blind_summary_pooled": blind_pooled,
        "blind_summary_by_instance": blind_by_inst,
        "exp4_curves_by_instance": gen_curves_by_inst,
        "replication_rule": REPLICATION_RULE,
    }

    config = {
        "gen_seed": GEN_SEED,
        "seed_offset": offset,
        "smoke": bool(smoke),
        "wall_clock_s": round(wall, 2),
        "exp4_replication": {**cfg4, "conditions": list(EXP4_CONDS),
                             "total_budget_per_condition":
                                 cfg4["gens"] * cfg4["life"],
                             "eval_protocol": EVAL4_NOTE,
                             "budget_note": BUDGET_NOTE_4},
        "exp1_replication": {**cfg1, "conditions": list(EXP1_CONDS),
                             "eval_protocol": EVAL1_NOTE,
                             "budget_note": BUDGET_NOTE_1},
    }

    return {"experiment": "exp7_layouts", "hypothesis": HYPOTHESIS,
            "config": config, "conditions": conditions, "curves": curves,
            "tests": tests, "conclusion": conclusion, "viz": viz}


def main():
    ap = argparse.ArgumentParser(description="EXP7 layout-resampling "
                                             "robustness")
    ap.add_argument("--seeds4", type=int, default=FULL4["seeds"],
                    help="seeds per TrapGrid instance (part 1)")
    ap.add_argument("--seeds1", type=int, default=FULL1["seeds"],
                    help="seeds per home pair (part 2)")
    ap.add_argument("--seed-offset", type=int, default=DEFAULT_SEED_OFFSET,
                    dest="seed_offset",
                    help="true seed = index + offset; default 100 keeps "
                         "confirmatory seeds disjoint from all tuning seeds")
    ap.add_argument("--out", default="results/exp7_layouts.json")
    ap.add_argument("--smoke", action="store_true",
                    help="2 seeds/instance, ~10x reduced budgets")
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()
    cfg4 = dict(SMOKE4) if args.smoke else dict(FULL4, seeds=args.seeds4)
    cfg1 = dict(SMOKE1) if args.smoke else dict(FULL1, seeds=args.seeds1)
    t0 = time.time()
    traps, homes = gen_instances(cfg4["instances"], cfg1["instances"])
    n4, n1 = cfg4["instances"] * cfg4["seeds"], cfg1["instances"] * cfg1["seeds"]
    r4 = run_seeds(partial(_unit4, maps=traps, cfg=cfg4,
                           offset=args.seed_offset), n4,
                   n_jobs=min(args.jobs, n4))
    r1 = run_seeds(partial(_unit1, pairs=homes, cfg=cfg1,
                           offset=args.seed_offset), n1,
                   n_jobs=min(args.jobs, n1))
    out = aggregate(r4, r1, traps, homes, cfg4, cfg1,
                    offset=args.seed_offset, smoke=args.smoke,
                    wall=time.time() - t0)
    save_json(args.out, out)
    print("wrote %s (%d+%d units, offset %d, %.1fs)"
          % (args.out, n4, n1, args.seed_offset, time.time() - t0))
    for e in out["tests"]["per_instance"]["exp4"]:
        print("  exp4 inst%d: IQM %.2f vs %.2f, MW p=%.3g, replicates=%s"
              % (e["instance"], e["iqm_a"], e["iqm_b"], e["p_mw"],
                 e["replicates"]))
    for e in out["tests"]["per_instance"]["exp1"]:
        print("  exp1 inst%d: IQM %.2f vs %.2f, MW p=%.3g, replicates=%s"
              % (e["instance"], e["iqm_a"], e["iqm_b"], e["p_mw"],
                 e["replicates"]))
    for c in out["conclusion"]["claims"]:
        print("  [%s] %s" % (c["verdict"], c["claim"].split(":")[0]))
    print("  " + out["conclusion"]["summary"])


if __name__ == "__main__":
    main()
