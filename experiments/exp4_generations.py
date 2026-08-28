"""EXP4 — Generational Teaching (H4: iterated distillation).

Claim: a lineage of short-lived agents, each distilling its best episodes to
a fresh, plastic student through a narrow (s, a) advice bottleneck, ratchets
past (a) weight-copy transfer and (b) one agent living the combined lifetime
with decaying plasticity. TrapGrid's terminal candy cells end episodes a few
steps from home, so the big goal is discovered occasionally but almost never
consolidated within one rigid-by-then lifetime; the bottleneck carries
exactly that rare peak episode into a young brain, which re-earns it.

Conditions (budgets matched EXACTLY at gens * life training steps, 5 x 15k):
- generational-distill: teacher's top-3 episodes -> <=100 dedup (s, a) pairs
  -> fresh student pretrained Q[s, a] = 5.0, fresh schedules.
- weight-copy: student starts from a full copy of teacher's Q, fresh
  schedules (transfer learning control).
- one-long-life: one agent lives all 75k steps, plasticity halflife 5k.
- one-long-life-slow: as above with halflife 25k (tuned fair baseline).
- no-inheritance: 5 independent lives (flat line expected — proves
  inheritance is causal).

Aging: lr(age) = 0.3 * 2^(-age/halflife), eps(age) = 0.4 * 2^(-age/halflife).

Eval at each generation boundary is one greedy rollout with epsilon forced
to 0 and argmax tie-breaking (env and policy both deterministic, so the
20-rollout protocol collapses to its single support point), plus the
Q-derived start value. Eval consumes no training budget, uses no rng, and is
byte-identical across conditions.

Usage: python experiments/exp4_generations.py --seeds 30 --out results/exp4.json [--smoke] [--jobs 20]
"""

import argparse
import time
from functools import partial

import numpy as np

from devrl.agents.distill import (EpisodicMemory, apply_advice, extract_advice,
                                  halflife_schedule)
from devrl.agents.qlearning import QLearner
from devrl.envs.trapgrid import TRAP_MAP, TrapGrid
from devrl.run import run_seeds, save_json
from devrl.stats import bootstrap_ci, iqm, mann_whitney

CONDITIONS = ["generational-distill", "weight-copy", "one-long-life",
              "one-long-life-slow", "no-inheritance"]

FULL = dict(gens=5, life=15000, halflife=5000, slow_halflife=25000,
            lr0=0.3, eps0=0.4, gamma=0.99, cap=120,
            memory_k=3, advice_cap=100, advice_value=5.0, n_boot=10000)
SMOKE = dict(FULL, life=1500, halflife=500, slow_halflife=2500, n_boot=2000)

CURVE_METRICS = ["greedy_return", "big_goal", "gap", "q_start"]
ACTION_NAMES = ["up", "right", "down", "left"]

_GRID = TrapGrid()
_START_STATE = _GRID.state_of(_GRID.start)

HYPOTHESIS = ("H4: a lineage of short-lived agents, each distilling its top "
              "episodes to a fresh plastic student through a narrow advice "
              "bottleneck, ratchets past weight-copy transfer and past single "
              "long lives with decaying plasticity.")


def greedy_rollout(Q):
    """One deterministic greedy rollout (eps=0, argmax ties -> lowest action).

    Returns (undiscounted return, reached-big-goal, [[row, col], ...])."""
    env = TrapGrid()
    s = env.reset()
    traj = [[env.pos[0], env.pos[1]]]
    ret = 0.0
    while True:
        a = int(np.argmax(Q[s]))
        s, r, done, info = env.step(a)
        traj.append([env.pos[0], env.pos[1]])
        ret += r
        if done or info["truncated"]:
            return ret, bool(done and env.pos == env.goal), traj


def _fresh_agent(cfg, halflife, rng):
    return QLearner(n_states=_GRID.n_states, n_actions=TrapGrid.n_actions,
                    lr=halflife_schedule(cfg["lr0"], halflife),
                    gamma=cfg["gamma"],
                    eps=halflife_schedule(cfg["eps0"], halflife),
                    optimistic_init=0.0, rng=rng)


def _live(agent, env, memory, n_steps, carry=None):
    """Exactly n_steps of eps-greedy life. Completed episodes (terminal or
    cap-truncated) enter memory; the mid-episode carry is returned so a long
    life can span checkpoint boundaries without an artificial reset."""
    s, sa, ret = carry if carry is not None else (env.reset(), [], 0.0)
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
    return s, sa, ret


def _gen_metrics(agent, memory):
    """Generation-boundary eval: greedy outcome, Q-derived start value, and
    the best-memory-vs-greedy gap (unconsolidated knowledge)."""
    ret, big, traj = greedy_rollout(agent.Q)
    best = float(memory.best_return())
    return {"greedy_return": float(ret), "big_goal": int(big),
            "greedy_steps": len(traj) - 1,
            "q_start": float(agent.Q[_START_STATE].max()),
            "best_memory_return": best, "gap": best - float(ret)}, traj


def _run_condition(cond, rng, cfg):
    gens, life = cfg["gens"], cfg["life"]
    hl = cfg["slow_halflife"] if cond == "one-long-life-slow" else cfg["halflife"]
    env = TrapGrid(cap=cfg["cap"])
    per_gen, steps_trained, final_traj, out = [], 0, None, {}
    if cond in ("one-long-life", "one-long-life-slow"):
        agent = _fresh_agent(cfg, hl, rng)
        memory = EpisodicMemory(cfg["memory_k"])
        carry = None
        for _ in range(gens):
            carry = _live(agent, env, memory, life, carry)
            steps_trained += life
            m, final_traj = _gen_metrics(agent, memory)
            per_gen.append(m)
    else:
        advice, q_prev, advice_by_gen = None, None, []
        for _ in range(gens):
            agent = _fresh_agent(cfg, hl, rng)
            if cond == "weight-copy" and q_prev is not None:
                agent.Q[:] = q_prev
            if cond == "generational-distill" and advice:
                apply_advice(agent.Q, advice, cfg["advice_value"])
            memory = EpisodicMemory(cfg["memory_k"])
            _live(agent, env, memory, life)
            steps_trained += life
            m, final_traj = _gen_metrics(agent, memory)
            per_gen.append(m)
            if cond == "generational-distill":
                advice = extract_advice(memory, cfg["advice_cap"])
                advice_by_gen.append(advice)
            elif cond == "weight-copy":
                q_prev = agent.Q.copy()
        if cond == "generational-distill":
            out["advice_by_gen"] = advice_by_gen
    out.update(steps_trained=steps_trained, per_gen=per_gen, final_traj=final_traj)
    return out


def _run_seed(seed, cfg):
    """All five conditions for one seed. Each condition draws from its own
    seed-derived rng stream and trains exactly gens*life env steps."""
    return {cond: _run_condition(cond, np.random.default_rng([seed, ci]), cfg)
            for ci, cond in enumerate(CONDITIONS)}


def _holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def _mw_safe(a, b):
    """Mann-Whitney that tolerates a fully degenerate comparison (every value
    in both groups identical -> no evidence either way, p = 1)."""
    if len(set(a) | set(b)) == 1:
        return {"u": len(a) * len(b) / 2.0, "p": 1.0}
    return mann_whitney(a, b)


def _aggregate(results, cfg, seeds, wall_s):
    boot_rng = np.random.default_rng(202404)
    steps = [cfg["life"] * (g + 1) for g in range(cfg["gens"])]

    curves = {}
    for metric in CURVE_METRICS:
        conds = {}
        for cond in CONDITIONS:
            vals = [[r[cond]["per_gen"][g][metric] for r in results]
                    for g in range(cfg["gens"])]
            cis = [bootstrap_ci(v, n_boot=cfg["n_boot"], rng=boot_rng,
                                statistic=iqm) for v in vals]
            conds[cond] = {"iqm": [iqm(v) for v in vals],
                           "mean": [float(np.mean(v)) for v in vals],
                           "ci_lo": [c[0] for c in cis],
                           "ci_hi": [c[1] for c in cis]}
        curves[metric] = {"steps": steps, "conditions": conds}

    finals = {cond: [r[cond]["per_gen"][-1]["greedy_return"] for r in results]
              for cond in CONDITIONS}
    fi = {cond: iqm(finals[cond]) for cond in CONDITIONS}
    comparisons = [("generational-distill", c) for c in CONDITIONS[1:]]
    raw = [_mw_safe(finals[a], finals[b]) for a, b in comparisons]
    adj = _holm([t["p"] for t in raw])
    tests = {"metric": "final greedy return (generation %d)" % cfg["gens"],
             "correction": "holm-bonferroni", "alpha": 0.05,
             "comparisons": [
                 {"a": a, "b": b, "iqm_a": fi[a], "iqm_b": fi[b],
                  "u": t["u"], "p": t["p"], "p_holm": float(pa),
                  "significant": bool(pa < 0.05)}
                 for (a, b), t, pa in zip(comparisons, raw, adj)]}

    sig = {c["b"]: c["significant"] for c in tests["comparisons"]}
    distill_iqms = curves["greedy_return"]["conditions"]["generational-distill"]["iqm"]
    g1, gN = distill_iqms[0], distill_iqms[-1]
    highest = all(fi["generational-distill"] > fi[c] for c in CONDITIONS[1:])
    supported = bool(highest and all(
        sig[b] for b in ("weight-copy", "one-long-life", "one-long-life-slow")))
    summary = ("Final greedy-return IQMs: "
               + ", ".join("%s=%.2f" % (c, fi[c]) for c in CONDITIONS)
               + ". Distill gen-1 IQM %.2f -> gen-%d %.2f%s." % (
                   g1, cfg["gens"], gN,
                   " (ratchets upward)" if gN > g1 else " (no ratchet)")
               + " Holm-adjusted p vs distill: "
               + ", ".join("%s=%.3g" % (c["b"], c["p_holm"])
                           for c in tests["comparisons"]) + ".")

    # representative seed: distill final closest to the condition IQM
    dvals = finals["generational-distill"]
    rep = int(np.argmin([abs(v - fi["generational-distill"]) for v in dvals]))
    advice_viz = [
        {"generation": g + 1, "n_pairs": len(adv),
         "pairs": [{"s": int(s), "rc": [int(s) // _GRID.W, int(s) % _GRID.W],
                    "a": int(a), "action": ACTION_NAMES[a]} for s, a in adv]}
        for g, adv in enumerate(results[rep]["generational-distill"]["advice_by_gen"])]
    viz = {
        "grid": {"map": TRAP_MAP.strip().splitlines(),
                 "height": _GRID.H, "width": _GRID.W,
                 "start": list(_GRID.start), "goal": list(_GRID.goal),
                 "candies": sorted(list(c) for c in _GRID.candies),
                 "candy_reward": TrapGrid.CANDY_REWARD,
                 "goal_reward": TrapGrid.GOAL_REWARD,
                 "cap": cfg["cap"],
                 "shortest_safe_path": _GRID.shortest_path_len()},
        "action_names": ACTION_NAMES,
        "per_generation_curves": curves,
        "representative_seed": rep,
        "advice_by_generation": advice_viz,
        "final_greedy_paths": {cond: results[rep][cond]["final_traj"]
                               for cond in CONDITIONS},
        "per_gen_representative": {cond: results[rep][cond]["per_gen"]
                                   for cond in CONDITIONS},
    }

    return {"experiment": "exp4_generations",
            "hypothesis": HYPOTHESIS,
            "config": {**cfg, "seeds": seeds, "conditions": CONDITIONS,
                       "total_budget": cfg["gens"] * cfg["life"],
                       "wall_clock_s": round(wall_s, 2)},
            "conditions": {cond: {"seeds": [dict(seed=i, **results[i][cond])
                                            for i in range(seeds)]}
                           for cond in CONDITIONS},
            "curves": curves,
            "tests": tests,
            "conclusion": {"supported": supported, "summary": summary},
            "viz": viz}


def main():
    ap = argparse.ArgumentParser(description="EXP4 generational teaching")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default="results/exp4.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()
    cfg = SMOKE if args.smoke else FULL
    seeds = 2 if args.smoke else args.seeds
    t0 = time.time()
    results = run_seeds(partial(_run_seed, cfg=cfg), seeds,
                        n_jobs=min(args.jobs, seeds))
    out = _aggregate(results, cfg, seeds, time.time() - t0)
    save_json(args.out, out)
    print("wrote %s (%d seeds, %.1fs) supported=%s" % (
        args.out, seeds, time.time() - t0, out["conclusion"]["supported"]))


if __name__ == "__main__":
    main()
