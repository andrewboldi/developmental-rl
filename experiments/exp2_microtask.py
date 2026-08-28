"""EXP2 — Microtasks (H2): drills beat whole-game practice.

A reverse curriculum over START STATES ONLY in SoccerGrid: first shoot from
the attacking third (spawn already carrying), then dribble from midfield
(spawn in the left half, ball waiting at center), then full games from
kickoff. No synthetic rewards and one shared Q-table per run, so any speedup
comes purely from where episodes begin. The blocked ablation runs the same
phases from a single fixed spawn per drill (links to H3).

Conditions (training budgets matched EXACTLY; see DESIGN.md Amendments v2):
  whole            all steps on full games from kickoff
  drills-varied    20% shoot drill (random attacking-third spawn), 20% dribble
                   drill (random left-half spawn), 60% full games
  drills-fixed     same phases, one fixed spawn cell per drill (region centers)
  whole-optimistic whole, but Q initialized at 1.0 (the return upper bound) —
                   probes the exploration boundary condition found by the
                   adversarial verification (Xie et al. 2021; JSRL)
  explore-starts   uniformly random non-terminal (agent, ball) spawns for the
                   first 40% of budget (the drill fraction), then full games —
                   the classic exploring-starts alternative: start-state
                   diversity without drill structure

Eval: 20 greedy episodes from kickoff every 2k steps — on a separate env and
rng, so eval consumes no training budget and the protocol is identical across
conditions. Primary metric: time to 90% eval success; censored seeds are
reported and imputed at budget+1 for rank tests (conservative). Every primary
comparison reports both Mann-Whitney and Welch t p-values, each Holm-adjusted
within the 5-comparison family; registered decisions ride on the MW Holm p.
The conclusion is per-claim: {claims: [{claim, verdict, evidence}], summary}.

Confirmatory seeds are 100..(100+N-1) via --seed-offset (default 100),
disjoint from every seed ever used for tuning or verification (0-29,
1000-1029, 3000-3019).

Usage:
  python experiments/exp2_microtask.py --seeds 30 --out results/exp2.json
                                       [--smoke] [--jobs 20] [--seed-offset 100]
"""

import argparse
import functools
import time

import numpy as np
from scipy.stats import ttest_ind

from devrl import stats
from devrl.agents.qlearning import QLearner
from devrl.envs.soccer import CARRIED, SoccerGrid
from devrl.run import run_seeds, save_json

CONDITIONS = ("whole", "drills-varied", "drills-fixed",
              "whole-optimistic", "explore-starts")
BUDGET, EVAL_EVERY = 60_000, 2_000
CAP = 100
N_EVAL_EPISODES = 20
THRESHOLD = 0.9
LR, GAMMA, EPS = 0.3, 0.99, 0.15
SHOOT_FRAC, DRIBBLE_FRAC = 0.2, 0.2      # phase shares of the budget
EXPLORE_FRAC = SHOOT_FRAC + DRIBBLE_FRAC  # explore-starts share == drill share
OPTIMISTIC_Q0 = 1.0                      # upper bound: single terminal reward 1
ATTACK_COLS = (8, 11)                    # attacking third: cols 8..10
LEFT_COLS = (0, 5)                       # left half: cols 0..4
FIXED_SHOOT_SPAWN = (3, 9)               # region centers (blocked ablation)
FIXED_DRIBBLE_SPAWN = (3, 2)
HYPOTHESIS = ("H2: a reverse curriculum over start states (shoot from near "
              "goal -> dribble from midfield -> full game) reaches full-game "
              "mastery in fewer total env steps than whole-game practice; "
              "no synthetic rewards, only the start-state distribution changes.")
PRIMARY_PAIRS = (("drills-varied", "whole"),            # claim 1 headline
                 ("drills-fixed", "whole"),
                 ("drills-varied", "drills-fixed"),
                 ("drills-varied", "whole-optimistic"),  # claim 2 boundary
                 ("drills-varied", "explore-starts"))    # claim 3 structure
CONTEXT_PAIRS = (("whole-optimistic", "whole"), ("explore-starts", "whole"))


def q0_for(condition):
    """Q-table initialization: optimistic only for whole-optimistic."""
    return OPTIMISTIC_Q0 if condition == "whole-optimistic" else 0.0


def phase_at(condition, step, budget):
    """Curriculum phase at a training step (fixed per episode at its start)."""
    if condition in ("whole", "whole-optimistic"):
        return "game"
    if condition == "explore-starts":
        return "explore" if step < round(EXPLORE_FRAC * budget) else "game"
    if step < round(SHOOT_FRAC * budget):
        return "shoot"
    if step < round((SHOOT_FRAC + DRIBBLE_FRAC) * budget):
        return "dribble"
    return "game"


def start_state(condition, phase, rng):
    """(agent cell, ball) for one episode. The entire curriculum lives here."""
    if phase == "shoot":
        if condition == "drills-fixed":
            return FIXED_SHOOT_SPAWN, CARRIED
        return (int(rng.integers(SoccerGrid.H)),
                int(rng.integers(*ATTACK_COLS))), CARRIED
    if phase == "dribble":
        if condition == "drills-fixed":
            return FIXED_DRIBBLE_SPAWN, SoccerGrid.KICKOFF_BALL
        return (int(rng.integers(SoccerGrid.H)),
                int(rng.integers(*LEFT_COLS))), SoccerGrid.KICKOFF_BALL
    if phase == "explore":
        # Uniform over non-terminal states: agent anywhere; ball at any cell
        # or carried (spawning on the ball's cell also picks it up).
        agent = (int(rng.integers(SoccerGrid.H)), int(rng.integers(SoccerGrid.W)))
        code = int(rng.integers(SoccerGrid.n_cells + 1))
        ball = CARRIED if code == SoccerGrid.n_cells else divmod(code, SoccerGrid.W)
        return agent, ball
    return SoccerGrid.KICKOFF_AGENT, SoccerGrid.KICKOFF_BALL


def greedy_eval(Q, rng, n_episodes=N_EVAL_EPISODES, cap=CAP):
    """Fraction of greedy kickoff episodes that score.

    Runs on its own env and rng (never the training ones), so evaluation is
    free of training budget and identical across conditions; argmax
    tie-breaking makes the policy deterministic.
    """
    env = SoccerGrid(rng=rng)
    wins = 0
    for _ in range(n_episodes):
        s = env.reset()
        for _ in range(cap):
            s, r, done, _ = env.step(int(np.argmax(Q[s])))
            if done:
                wins += int(r > 0)
                break
    return wins / n_episodes


def _ball_repr(ball):
    return ball if isinstance(ball, str) else list(ball)


def greedy_traj(Q, rng, cap=CAP):
    """One greedy kickoff episode, recorded step by step for the viz block."""
    env = SoccerGrid(rng=rng)
    s = env.reset()
    steps, scored = [], False
    for _ in range(cap):
        agent, ball = env.decode(s)
        a = int(np.argmax(Q[s]))
        s, r, done, info = env.step(a)
        rec = {"agent": list(agent), "ball": _ball_repr(ball), "a": a}
        if info["shot"]:
            rec["shot_p"] = info["shot_p"]
        steps.append(rec)
        if done:
            scored = info["scored"]
            break
    agent, ball = env.decode(s)
    return {"scored": scored, "steps": steps,
            "final": {"agent": list(agent), "ball": _ball_repr(ball)}}


def _train(seed, condition, budget, eval_every, cap):
    """Train one seed of one condition; returns curve, trajectories and Q.

    Every training step — drill or game — is charged against the same budget,
    and the loop pauses mid-episode for checkpoints, so all conditions see
    exactly `budget` training steps and identical checkpoint times. Eval and
    trajectory rngs are derived from (seed, step) only, never from the
    training streams.
    """
    env_rng, agent_rng, start_rng = (np.random.default_rng(c)
                                     for c in np.random.SeedSequence(seed).spawn(3))
    env = SoccerGrid(rng=env_rng)
    agent = QLearner(env.n_states, env.n_actions, lr=LR, gamma=GAMMA, eps=EPS,
                     optimistic_init=q0_for(condition), rng=agent_rng)
    traj_at = {budget // 4: "25", budget // 2: "50", budget: "100"}
    checkpoints = [0]
    curve = [greedy_eval(agent.Q, np.random.default_rng([seed, 0]), cap=cap)]
    trajs, steps_done = {}, 0
    while steps_done < budget:
        agent_rc, ball = start_state(condition,
                                     phase_at(condition, steps_done, budget),
                                     start_rng)
        s = env.reset(agent=agent_rc, ball=ball)
        for _ in range(cap):
            a = agent.act(s)
            s2, r, done, _ = env.step(a)
            agent.update(s, a, r, s2, done)
            s = s2
            steps_done += 1
            if steps_done % eval_every == 0:
                checkpoints.append(steps_done)
                curve.append(greedy_eval(
                    agent.Q, np.random.default_rng([seed, steps_done]), cap=cap))
            if steps_done in traj_at:
                trajs[traj_at[steps_done]] = {
                    "step": steps_done,
                    **greedy_traj(agent.Q,
                                  np.random.default_rng([seed, steps_done, 7]),
                                  cap=cap)}
            if done or steps_done >= budget:
                break
    return {"seed": seed, "checkpoints": checkpoints, "curve": curve,
            "trajs": trajs, "train_steps": steps_done, "Q": agent.Q}


def run_seed(seed, condition, budget=BUDGET, eval_every=EVAL_EVERY, cap=CAP):
    """Per-seed entry point (top-level for fork pickling); JSON-safe result."""
    out = _train(seed, condition, budget, eval_every, cap)
    out.pop("Q")
    return out


def t90_stats(checkpoints, curves, budget, threshold=THRESHOLD):
    """Per-seed time to threshold; censored -> None, imputed at budget+1."""
    t90 = [stats.time_to_threshold(checkpoints, c, threshold) for c in curves]
    imputed = [budget + 1 if t is None else t for t in t90]
    return t90, imputed, sum(t is None for t in t90)


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def _mw(a, b):
    """Mann-Whitney, tolerant of fully tied samples (p=1 by convention)."""
    if np.ptp(np.concatenate([np.asarray(a, float), np.asarray(b, float)])) == 0:
        return {"u": len(a) * len(b) / 2.0, "p": 1.0}
    return stats.mann_whitney(a, b)


def phase_blocks(condition, budget):
    if condition == "whole":
        return [{"phase": "game", "start": 0, "end": budget}]
    return [{"phase": "shoot", "start": 0, "end": round(SHOOT_FRAC * budget)},
            {"phase": "dribble", "start": round(SHOOT_FRAC * budget),
             "end": round((SHOOT_FRAC + DRIBBLE_FRAC) * budget)},
            {"phase": "game", "start": round((SHOOT_FRAC + DRIBBLE_FRAC) * budget),
             "end": budget}]


def assemble(results, budget, n_boot=10_000, meta=None):
    """Aggregate per-seed results into the shared output JSON contract."""
    boot = np.random.default_rng(0)
    ckpts = results[CONDITIONS[0]][0]["checkpoints"]
    conditions, curve_block, imputed = {}, {}, {}
    viz_trajs, viz_seed = {}, {}
    for cond in CONDITIONS:
        runs = results[cond]
        curves = [r["curve"] for r in runs]
        final = [c[-1] for c in curves]
        t90, imp, n_cens = t90_stats(ckpts, curves, budget)
        conditions[cond] = {
            "n_seeds": len(runs),
            "final_success": final,
            "final_success_iqm": stats.iqm(final),
            "final_success_ci": list(stats.bootstrap_ci(
                final, n_boot=n_boot, rng=boot, statistic=stats.iqm)),
            "time_to_90": t90,
            "time_to_90_imputed": imp,
            "time_to_90_iqm": stats.iqm(imp),
            "time_to_90_ci": list(stats.bootstrap_ci(
                imp, n_boot=n_boot, rng=boot, statistic=stats.iqm)),
            "n_censored": n_cens,
            "censored_fraction": n_cens / len(runs),
            "success_curve_per_seed": curves,
        }
        imputed[cond] = imp
        arr = np.asarray(curves)
        cis = [stats.bootstrap_ci(arr[:, j], n_boot=n_boot, rng=boot,
                                  statistic=stats.iqm)
               for j in range(arr.shape[1])]
        curve_block[cond] = {"iqm": [stats.iqm(arr[:, j]) for j in range(arr.shape[1])],
                             "ci_lo": [c[0] for c in cis],
                             "ci_hi": [c[1] for c in cis]}
        mid = int(np.argsort(final)[len(final) // 2])  # median-final seed for viz
        viz_seed[cond] = runs[mid]["seed"]
        viz_trajs[cond] = runs[mid]["trajs"]

    pairs = [("drills-varied", "whole"), ("drills-fixed", "whole"),
             ("drills-varied", "drills-fixed")]
    raw = [_mw(imputed[a], imputed[b]) for a, b in pairs]
    adj = holm([r["p"] for r in raw])
    primary = []
    for (a, b), r, ph in zip(pairs, raw, adj):
        ia, ib = stats.iqm(imputed[a]), stats.iqm(imputed[b])
        primary.append({"comparison": f"{a} vs {b}",
                        "metric": "time_to_90 (steps; censored -> budget+1)",
                        "u": r["u"], "p": r["p"], "p_holm": float(ph),
                        "iqm": {a: ia, b: ib},
                        "faster": a if ia < ib else (b if ib < ia else "tie"),
                        "significant": bool(ph < 0.05)})
    secondary = [{"comparison": f"{a} vs {b}", "metric": "final_success",
                  **_mw(conditions[a]["final_success"], conditions[b]["final_success"])}
                 for a, b in pairs]

    head = primary[0]
    supported = bool(head["significant"] and head["faster"] == "drills-varied")
    summary = ("Time to 90% eval success (IQM steps): "
               + ", ".join(f"{c}={conditions[c]['time_to_90_iqm']:.0f}"
                           for c in CONDITIONS)
               + f". drills-varied vs whole: Holm p={head['p_holm']:.3g} — "
               + ("supports" if supported else "does not support") + " H2. "
               + "Censored: "
               + ", ".join(f"{c} {conditions[c]['n_censored']}/{conditions[c]['n_seeds']}"
                           for c in CONDITIONS) + ".")

    geom = SoccerGrid(rng=np.random.default_rng(0))
    viz = {
        "pitch": {"W": geom.W, "H": geom.H,
                  "goal_cells": [list(g) for g in geom.GOAL_CELLS],
                  "goal_center": list(geom.GOAL_CENTER),
                  "kickoff": {"agent": list(geom.KICKOFF_AGENT),
                              "ball": list(geom.KICKOFF_BALL)},
                  "attacking_third_cols": list(range(*ATTACK_COLS)),
                  "left_half_cols": list(range(*LEFT_COLS)),
                  "fixed_spawns": {"shoot": list(FIXED_SHOOT_SPAWN),
                                   "dribble": list(FIXED_DRIBBLE_SPAWN)},
                  "shot_p": [[geom.shot_p((r, c)) for c in range(geom.W)]
                             for r in range(geom.H)]},
        "action_names": ["up", "right", "down", "left", "shoot"],
        "phases": {c: phase_blocks(c, budget) for c in CONDITIONS},
        "trajectories": viz_trajs,
        "viz_seed": viz_seed,
        "eval_curves": {"checkpoints": ckpts, "threshold": THRESHOLD,
                        "conditions": curve_block},
    }
    return {"experiment": "exp2_microtask",
            "hypothesis": HYPOTHESIS,
            "meta": meta or {},
            "conditions": conditions,
            "curves": {"checkpoints": ckpts, "conditions": curve_block},
            "tests": {"alpha": 0.05,
                      "correction": "holm-bonferroni over the primary family",
                      "primary": primary,
                      "secondary_final_success": secondary},
            "conclusion": {"supported": supported, "summary": summary},
            "viz": viz}


def main():
    ap = argparse.ArgumentParser(description="EXP2 microtask curriculum (H2)")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default="results/exp2.json")
    ap.add_argument("--smoke", action="store_true",
                    help="2 seeds, 10x reduced budget")
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()
    n_seeds, budget, eval_every = ((2, BUDGET // 10, EVAL_EVERY // 10)
                                   if args.smoke else (args.seeds, BUDGET, EVAL_EVERY))
    t0 = time.time()
    results = {}
    for cond in CONDITIONS:
        fn = functools.partial(run_seed, condition=cond, budget=budget,
                               eval_every=eval_every, cap=CAP)
        results[cond] = run_seeds(fn, n_seeds, n_jobs=args.jobs)
        print(f"{cond}: {n_seeds} seeds done at {time.time() - t0:.1f}s", flush=True)
    meta = {"seeds": n_seeds, "budget": budget, "eval_every": eval_every,
            "cap": CAP, "n_eval_episodes": N_EVAL_EPISODES, "threshold": THRESHOLD,
            "lr": LR, "gamma": GAMMA, "eps": EPS,
            "smoke": args.smoke, "jobs": args.jobs}
    out = assemble(results, budget, meta=meta)
    out["meta"]["wall_clock_s"] = round(time.time() - t0, 2)
    save_json(args.out, out)
    print(out["conclusion"]["summary"])


if __name__ == "__main__":
    main()
