"""EXP6 — The Self-Coach (H6): an agent devises its own microtasks.

The founding brief's untested idea: with no teacher to design drills, the
agent coaches itself ("ok, let's work on learning 1 measure"). It keeps an
online episodic memory of its top-10 episodes by return (ties -> earlier;
with SoccerGrid's binary returns the memory converges on the first ten
scoring episodes) and, during the practice phase, restarts episodes from
moments of those remembered best games — a start-state drill distribution
devised entirely from its own experience. Until the first goal is ever
scored, practice draws fall back to a uniform sample over ALL states
visited so far (exploration restarts).

Conditions (every arm trains exactly 60k env steps; DESIGN.md v3):
  whole            full games from kickoff (bit-identical rerun of EXP2 whole)
  teacher-drills   EXP2's drills-varied protocol, imported: 20% shoot drill,
                   20% dribble drill, 60% kickoff games (bit-identical rerun)
  self-drills      first 40% of budget: with p=0.75 restart from a uniformly
                   sampled state snapshot (agent cell + ball status) drawn
                   from the union of states visited in the top-10 remembered
                   episodes, else kickoff; final 60% kickoff only
  self-drills-late the same mechanism, but p anneals 0.75 -> 0 linearly over
                   the WHOLE budget (no hard phase boundary)

Eval (identical everywhere, budget-free, rng-disjoint from training): 20
greedy kickoff episodes every 2k steps, via EXP2's greedy_eval. Primary
metric: time to 90% eval success, censored seeds imputed at budget+1.
Primary family (Holm, m=2, MW and Welch adjusted separately): self-drills
vs whole (claim 1) and self-drills vs teacher-drills (claim 2, an
equivalence-style claim with a pre-registered +20% margin that never
accepts a bare null). Conclusions are per-claim verdicts.

Confirmatory seeds are 100..(100+N-1) via --seed-offset (default 100); dev
and test probes use seeds 0-50 only.

Usage:
  python experiments/exp6_selfcoach.py --seeds 30 --out results/exp6.json
                                       [--smoke] [--jobs 20] [--seed-offset 100]
"""

import argparse
import functools
import time

import numpy as np

from devrl import stats
from devrl.agents.qlearning import QLearner
from devrl.envs.soccer import CARRIED, SoccerGrid
from devrl.run import run_seeds, save_json

# EXP2 is the frozen protocol source: the whole/teacher arms, all shared
# hyperparameters, the eval functions, and the stats helpers come from it,
# so those arms replicate EXP2's shipped runs bit for bit (enforced by test).
from exp2_microtask import (BUDGET, CAP, DRIBBLE_FRAC, EPS, EVAL_EVERY, GAMMA,
                            LR, N_EVAL_EPISODES, SHOOT_FRAC, THRESHOLD, _mw,
                            _welch, greedy_eval, greedy_traj, holm, phase_at,
                            phase_blocks, start_state, t90_stats)

CONDITIONS = ("whole", "teacher-drills", "self-drills", "self-drills-late")
SELF_CONDITIONS = ("self-drills", "self-drills-late")
EXP2_ALIAS = {"whole": "whole", "teacher-drills": "drills-varied"}
SELF_FRAC = SHOOT_FRAC + DRIBBLE_FRAC  # practice share == teacher drill share
MEMORY_P = 0.75                        # memory-start probability (peak)
MEMORY_K = 10                          # top-k episodes remembered
MATCH_MARGIN = 0.2                     # claim-2 margin: +20% of teacher t90
DEV_DEMO_SEED = 3                      # dev-range seed for the end-to-end test
HYPOTHESIS = ("H6: when no teacher exists, an agent can devise its own "
              "microtasks by restarting practice from moments of its "
              "remembered best episodes — an online top-10 episodic memory "
              "turns 'my best games' into a start-state drill distribution. "
              "Self-drills should reach full-game mastery faster than "
              "whole-game practice (claim 1) and are compared head-to-head "
              "with EXP2's teacher-designed drills (claim 2: match, or "
              "quantify the value of expert curriculum design); the annealed "
              "variant probes schedule sensitivity.")
PRIMARY_PAIRS = (("self-drills", "whole"),            # claim 1
                 ("self-drills", "teacher-drills"))   # claim 2
CONTEXT_PAIRS = (("teacher-drills", "whole"),         # EXP2 replication check
                 ("self-drills-late", "whole"),
                 ("self-drills", "self-drills-late"))  # schedule sensitivity

_GEOM = SoccerGrid(rng=np.random.default_rng(0))  # geometry only, never stepped


class SelfCoach:
    """Online self-curriculum: remember your best games, practice their moments.

    Keeps the top-`k` episodes by undiscounted return (ties -> the EARLIER
    episode wins) as (return, arrival, [(s, a), ...]) records, plus the set of
    every state the agent ever acted from. Practice candidates are the union
    of states in the remembered episodes once at least one of them scored;
    before that, all visited states (exploration restarts). Candidates are
    deduplicated and sorted, so sampling is uniform over DISTINCT states and
    deterministic given the rng. Bookkeeping consumes no rng.
    """

    def __init__(self, k=MEMORY_K):
        self.k = k
        self._eps = []   # (-return, arrival index, [(s, a), ...])
        self._n = 0
        self.visited = set()

    def observe(self, s):
        """Record a state the agent is about to act from."""
        self.visited.add(int(s))

    def end_episode(self, ep_return, sa_pairs):
        """File a finished episode; keep only the top-k (ties -> earlier)."""
        self._eps.append((-float(ep_return), self._n, list(sa_pairs)))
        self._n += 1
        self._eps.sort(key=lambda e: e[:2])
        del self._eps[self.k:]

    def has_scoring_episode(self):
        return bool(self._eps) and -self._eps[0][0] > 0

    def practice_states(self):
        """Sorted candidate restart states for the current practice draw."""
        if self.has_scoring_episode():
            return sorted({s for _, _, sa in self._eps for s, _ in sa})
        return sorted(self.visited)

    def sample_start(self, rng):
        """One uniform draw over the candidates; None if nothing visited yet."""
        cands = self.practice_states()
        if not cands:
            return None
        return cands[int(rng.integers(len(cands)))]


def memory_p(condition, step, budget):
    """Memory-start probability at a training step (fixed per episode start)."""
    if condition == "self-drills":
        return MEMORY_P if step < round(SELF_FRAC * budget) else 0.0
    if condition == "self-drills-late":
        return MEMORY_P * max(0.0, 1.0 - step / budget)
    return 0.0


def self_start(coach, p, rng):
    """(agent, ball, kind) for one self-coached episode.

    With probability p, restart from a remembered moment ("memory") or —
    before the first scoring episode — from any visited state ("fallback");
    otherwise kickoff. Draws from rng only while p > 0, so the kickoff-only
    game phase leaves the start-state stream untouched.
    """
    if p > 0 and rng.random() < p:
        s = coach.sample_start(rng)
        if s is not None:
            agent, ball = _GEOM.decode(s)
            kind = "memory" if coach.has_scoring_episode() else "fallback"
            return agent, ball, kind
    return SoccerGrid.KICKOFF_AGENT, SoccerGrid.KICKOFF_BALL, "kickoff"


def episode_start(condition, coach, step, budget, rng):
    """Start state for one episode under any condition's protocol."""
    if condition in EXP2_ALIAS:
        c2 = EXP2_ALIAS[condition]
        agent, ball = start_state(c2, phase_at(c2, step, budget), rng)
        return agent, ball, "protocol"
    return self_start(coach, memory_p(condition, step, budget), rng)


def heat_windows(condition, budget):
    """Three equal (start, end, label) windows over the practice phase."""
    if condition == "self-drills":
        span = round(SELF_FRAC * budget)
    elif condition == "self-drills-late":
        span = budget
    else:
        return None
    e1, e2 = round(span / 3), round(2 * span / 3)
    return ((0, e1, "early"), (e1, e2, "mid"), (e2, span, "end"))


def _empty_window(w):
    start, end, label = w
    zeros = [[0] * SoccerGrid.W for _ in range(SoccerGrid.H)]
    return {"start": start, "end": end, "label": label,
            "agent_counts": zeros,
            "ball_counts": [[0] * SoccerGrid.W for _ in range(SoccerGrid.H)],
            "ball_carried": 0, "n_memory": 0, "n_fallback": 0}


def _record_start(window, agent_rc, ball, kind):
    window["agent_counts"][agent_rc[0]][agent_rc[1]] += 1
    if ball == CARRIED:
        window["ball_carried"] += 1
    else:
        window["ball_counts"][ball[0]][ball[1]] += 1
    window["n_memory" if kind == "memory" else "n_fallback"] += 1


def _train(seed, condition, budget, eval_every, cap):
    """Train one seed of one condition; returns curve, trajs, Q (+ coach data).

    The loop is EXP2's exactly — every training step charged to the same
    budget, checkpoints paused mid-episode, eval and trajectory rngs derived
    from (seed, step) only — so the whole/teacher arms consume their rng
    streams identically to EXP2 and replicate its runs bit for bit. Self
    arms add rng-free coach bookkeeping plus start draws from the same
    start-state stream.
    """
    env_rng, agent_rng, start_rng = (np.random.default_rng(c)
                                     for c in np.random.SeedSequence(seed).spawn(3))
    env = SoccerGrid(rng=env_rng)
    agent = QLearner(env.n_states, env.n_actions, lr=LR, gamma=GAMMA, eps=EPS,
                     optimistic_init=0.0, rng=agent_rng)
    is_self = condition in SELF_CONDITIONS
    coach = SelfCoach() if is_self else None
    windows = heat_windows(condition, budget)
    heat = [_empty_window(w) for w in windows] if is_self else None
    self_stats = ({"first_score_step": None, "n_memory_starts": 0,
                   "n_fallback_starts": 0, "n_kickoff_starts": 0}
                  if is_self else None)
    traj_at = {budget // 4: "25", budget // 2: "50", budget: "100"}
    checkpoints = [0]
    curve = [greedy_eval(agent.Q, np.random.default_rng([seed, 0]), cap=cap)]
    trajs, steps_done = {}, 0
    while steps_done < budget:
        agent_rc, ball, kind = episode_start(condition, coach, steps_done,
                                             budget, start_rng)
        if is_self:
            if kind == "kickoff":
                self_stats["n_kickoff_starts"] += 1
            else:
                self_stats[f"n_{kind}_starts"] += 1
                w = 0 if steps_done < windows[0][1] else \
                    (1 if steps_done < windows[1][1] else 2)
                _record_start(heat[w], agent_rc, ball, kind)
        s = env.reset(agent=agent_rc, ball=ball)
        ep_sa, ep_return, done = [], 0.0, False
        for _ in range(cap):
            a = agent.act(s)
            if is_self:
                coach.observe(s)
                ep_sa.append((int(s), int(a)))
            s2, r, done, _info = env.step(a)
            agent.update(s, a, r, s2, done)
            ep_return += r
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
        if is_self and (done or len(ep_sa) == cap):
            # budget-truncated episodes are incomplete games: not filed
            coach.end_episode(ep_return, ep_sa)
            if ep_return > 0 and self_stats["first_score_step"] is None:
                self_stats["first_score_step"] = steps_done
    out = {"seed": seed, "checkpoints": checkpoints, "curve": curve,
           "trajs": trajs, "train_steps": steps_done, "Q": agent.Q}
    if is_self:
        out["self_stats"] = self_stats
        out["heat"] = {"windows": heat}
        out["coach"] = coach
    return out


def run_seed(seed, condition, budget=BUDGET, eval_every=EVAL_EVERY, cap=CAP):
    """Per-seed entry point (top-level for fork pickling); JSON-safe result."""
    out = _train(seed, condition, budget, eval_every, cap)
    out.pop("Q")
    out.pop("coach", None)
    return out


def run_seed_at(i, offset, condition, budget=BUDGET, eval_every=EVAL_EVERY,
                cap=CAP):
    """Runs seed offset+i (top-level for fork pickling). Confirmatory runs use
    offset 100 so seeds are disjoint from all dev/test seeds (0-50)."""
    return run_seed(offset + i, condition=condition, budget=budget,
                    eval_every=eval_every, cap=cap)


def phase_blocks6(condition, budget):
    """Phase blocks for the viz, annotated with the memory-start schedule."""
    if condition in EXP2_ALIAS:
        return phase_blocks(EXP2_ALIAS[condition], budget)
    if condition == "self-drills":
        e = round(SELF_FRAC * budget)
        return [{"phase": "self-practice", "start": 0, "end": e,
                 "p_memory": MEMORY_P},
                {"phase": "game", "start": e, "end": budget}]
    return [{"phase": "self-practice-annealed", "start": 0, "end": budget,
             "p_memory_start": MEMORY_P, "p_memory_end": 0.0}]


def match_analysis(self_imp, teacher_imp, margin_frac=MATCH_MARGIN,
                   n_boot=10_000, rng=None):
    """Claim-2 equivalence view: bootstrap CI of the t90 IQM gap vs a margin.

    diff = IQM(self) - IQM(teacher) on imputed t90 (positive = self slower);
    the pre-registered margin is +20% of the teacher IQM. Registered in
    DESIGN.md v3; the claim never accepts a bare null as equivalence.
    """
    rng = rng if rng is not None else np.random.default_rng(1)
    a = np.asarray(self_imp, dtype=float)
    b = np.asarray(teacher_imp, dtype=float)
    diffs = np.array([stats.iqm(rng.choice(a, size=len(a), replace=True))
                      - stats.iqm(rng.choice(b, size=len(b), replace=True))
                      for _ in range(n_boot)])
    return {"metric": "time_to_90 IQM difference (self - teacher; steps)",
            "diff_iqm": stats.iqm(a) - stats.iqm(b),
            "ci_lo": float(np.percentile(diffs, 2.5)),
            "ci_hi": float(np.percentile(diffs, 97.5)),
            "margin_frac": margin_frac,
            "margin": margin_frac * stats.iqm(b),
            "n_boot": n_boot}


def build_claims(primary, match, conditions):
    """Per-claim verdicts per the decision rules registered in DESIGN.md v3."""
    by = {t["comparison"]: t for t in primary}

    def n_sig(t):
        return int(t["significant"]) + int(t["welch_significant"])

    def ev(t, extra=""):
        a, b = t["comparison"].split(" vs ")
        return (f"t90 IQM {a}={t['iqm'][a]:.0f} vs {b}={t['iqm'][b]:.0f} steps; "
                f"MW p_holm={t['p_holm']:.3g}, Welch p_holm={t['welch_p_holm']:.3g}; "
                f"censored {a} {conditions[a]['n_censored']}/{conditions[a]['n_seeds']}, "
                f"{b} {conditions[b]['n_censored']}/{conditions[b]['n_seeds']}"
                + (f". {extra}" if extra else "."))

    c1 = by["self-drills vs whole"]
    k1 = n_sig(c1)
    if k1 == 0:
        v1 = "null"
    elif k1 == 1:
        v1 = "boundary"
    elif c1["faster"] == "self-drills":
        v1 = "supported"
    elif c1["faster"] == "whole":
        v1 = "refuted"
    else:
        v1 = "boundary"
    claims = [{"claim": ("an agent can devise useful drills from its own "
                         "episodic memory: self-drills (restarts from "
                         "remembered best-episode moments) reaches 90% "
                         "full-game eval success in fewer env steps than "
                         "whole-game practice"),
               "verdict": v1, "evidence": ev(c1)}]

    c2 = by["self-drills vs teacher-drills"]
    k2 = n_sig(c2)
    within = match["ci_hi"] <= match["margin"]
    ci_txt = (f"gap CI95 [{match['ci_lo']:.0f}, {match['ci_hi']:.0f}] steps vs "
              f"margin +{match['margin']:.0f} (+{match['margin_frac']:.0%} of "
              f"teacher t90)")
    if k2 == 0:
        if within:
            v2, how = "supported", (
                "No detectable difference and the gap is bounded within the "
                f"margin ({ci_txt}): self-coaching matches the teacher's "
                "curriculum — agent self-sufficiency.")
        else:
            v2, how = "null", (
                f"No detectable difference but the gap CI exceeds the margin "
                f"({ci_txt}): underpowered, not equivalence.")
    elif c2["faster"] == "teacher-drills":
        gap = c2["iqm"]["self-drills"] - c2["iqm"]["teacher-drills"]
        frac = gap / c2["iqm"]["teacher-drills"]
        if within:
            v2, how = "boundary", (
                f"The teacher is detectably faster by {gap:.0f} steps "
                f"({frac:.0%} of teacher t90), but the gap stays within the "
                f"margin ({ci_txt}).")
        else:
            v2, how = "refuted", (
                f"Expert curriculum design is worth {gap:.0f} steps "
                f"({frac:.0%} of teacher t90): the teacher is detectably "
                f"faster and the gap exceeds the margin ({ci_txt}).")
    elif c2["faster"] == "self-drills" and k2 == 2:
        v2, how = "supported", (
            "Self-devised drills are significantly FASTER than the teacher's "
            f"({ci_txt}) — self-coaching matches and even beats expert "
            "curriculum design here.")
    else:
        v2, how = "boundary", (
            f"One-legged detection with self-drills faster ({ci_txt}).")
    claims.append({"claim": ("self-devised drills match teacher-designed "
                             "drills (t90 within a pre-registered +20% "
                             "margin; direction reported either way)"),
                   "verdict": v2, "evidence": ev(c2, how)})
    return claims


def _mean_grids(runs, w):
    """Cross-seed mean of window w's agent-count grid (floats)."""
    grids = np.array([r["heat"]["windows"][w]["agent_counts"] for r in runs],
                     dtype=float)
    return [[float(x) for x in row] for row in grids.mean(axis=0)]


def assemble(results, budget, n_boot=10_000, config=None):
    """Aggregate per-seed results into the shared output JSON contract."""
    boot = np.random.default_rng(0)
    ckpts = results[CONDITIONS[0]][0]["checkpoints"]
    conditions, curve_block, imputed = {}, {}, {}
    viz_trajs, viz_seed, heatmaps = {}, {}, {}
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
        if runs[0].get("self_stats") is not None:
            conditions[cond]["self_stats"] = [r["self_stats"] for r in runs]
        imputed[cond] = imp
        arr = np.asarray(curves)
        cis = [stats.bootstrap_ci(arr[:, j], n_boot=n_boot, rng=boot,
                                  statistic=stats.iqm)
               for j in range(arr.shape[1])]
        curve_block[cond] = {"iqm": [stats.iqm(arr[:, j])
                                     for j in range(arr.shape[1])],
                             "ci_lo": [c[0] for c in cis],
                             "ci_hi": [c[1] for c in cis]}
        mid = int(np.argsort(final)[len(final) // 2])  # median-final seed
        viz_seed[cond] = runs[mid]["seed"]
        viz_trajs[cond] = runs[mid]["trajs"]
        if runs[0].get("heat") is not None:
            heatmaps[cond] = {"seed": runs[mid]["seed"],
                              "windows": runs[mid]["heat"]["windows"],
                              "mean_agent_counts": [_mean_grids(runs, w)
                                                    for w in range(3)]}

    raw = [_mw(imputed[a], imputed[b]) for a, b in PRIMARY_PAIRS]
    welch = [_welch(imputed[a], imputed[b]) for a, b in PRIMARY_PAIRS]
    adj = holm([r["p"] for r in raw])
    wadj = holm([w["p"] for w in welch])
    primary = []
    for (a, b), r, ph, w, wph in zip(PRIMARY_PAIRS, raw, adj, welch, wadj):
        ia, ib = stats.iqm(imputed[a]), stats.iqm(imputed[b])
        entry = {"comparison": f"{a} vs {b}",
                 "metric": "time_to_90 (steps; censored -> budget+1)",
                 "u": r["u"], "p": r["p"], "p_holm": float(ph),
                 "welch_t": w["t"], "welch_p": w["p"],
                 "welch_p_holm": float(wph),
                 "iqm": {a: ia, b: ib},
                 "faster": a if ia < ib else (b if ib < ia else "tie"),
                 "significant": bool(ph < 0.05),
                 "welch_significant": bool(wph < 0.05)}
        if r.get("degenerate") or w.get("degenerate"):
            entry["degenerate"] = {"mw": bool(r.get("degenerate", False)),
                                   "welch": bool(w.get("degenerate", False))}
        primary.append(entry)
    match = match_analysis(imputed["self-drills"], imputed["teacher-drills"],
                           n_boot=n_boot)
    secondary = [{"comparison": f"{a} vs {b}", "metric": "final_success",
                  **_mw(conditions[a]["final_success"],
                        conditions[b]["final_success"])}
                 for a, b in PRIMARY_PAIRS]
    context = []
    for a, b in CONTEXT_PAIRS:  # descriptive, no claim rides on them (no Holm)
        mw, w = _mw(imputed[a], imputed[b]), _welch(imputed[a], imputed[b])
        entry = {"comparison": f"{a} vs {b}",
                 "metric": "time_to_90 (steps; censored -> budget+1)",
                 "u": mw["u"], "p": mw["p"],
                 "welch_t": w["t"], "welch_p": w["p"],
                 "iqm": {a: stats.iqm(imputed[a]), b: stats.iqm(imputed[b])}}
        if mw.get("degenerate") or w.get("degenerate"):
            entry["degenerate"] = {"mw": bool(mw.get("degenerate", False)),
                                   "welch": bool(w.get("degenerate", False))}
        context.append(entry)

    claims = build_claims(primary, match, conditions)
    first_scores = {c: [s["first_score_step"]
                        for s in conditions[c]["self_stats"]]
                    for c in SELF_CONDITIONS}
    kinds = {c: {k: int(sum(s[f"n_{k}_starts"]
                            for s in conditions[c]["self_stats"]))
                 for k in ("memory", "fallback", "kickoff")}
             for c in SELF_CONDITIONS}
    summary = ("Time to 90% eval success (IQM steps): "
               + ", ".join(f"{c}={conditions[c]['time_to_90_iqm']:.0f}"
                           for c in CONDITIONS)
               + f". self-drills vs whole: MW p_holm={primary[0]['p_holm']:.3g}"
               + f", Welch p_holm={primary[0]['welch_p_holm']:.3g}; "
               + f"self-drills vs teacher-drills: MW p_holm="
               + f"{primary[1]['p_holm']:.3g}, Welch p_holm="
               + f"{primary[1]['welch_p_holm']:.3g}, gap CI95 "
               + f"[{match['ci_lo']:.0f}, {match['ci_hi']:.0f}] vs margin "
               + f"+{match['margin']:.0f}. Verdicts: "
               + "; ".join(f"claim {i} ({label}) — {c['verdict']}"
                           for i, (label, c) in enumerate(
                               zip(("self-drills beat whole",
                                    "self matches teacher"), claims), 1))
               + ". Censored: "
               + ", ".join(f"{c} {conditions[c]['n_censored']}"
                           f"/{conditions[c]['n_seeds']}" for c in CONDITIONS)
               + ".")

    viz = {
        "pitch": {"W": _GEOM.W, "H": _GEOM.H,
                  "goal_cells": [list(g) for g in _GEOM.GOAL_CELLS],
                  "goal_center": list(_GEOM.GOAL_CENTER),
                  "kickoff": {"agent": list(_GEOM.KICKOFF_AGENT),
                              "ball": list(_GEOM.KICKOFF_BALL)},
                  "shot_p": [[_GEOM.shot_p((r, c)) for c in range(_GEOM.W)]
                             for r in range(_GEOM.H)]},
        "action_names": ["up", "right", "down", "left", "shoot"],
        "phases": {c: phase_blocks6(c, budget) for c in CONDITIONS},
        "trajectories": viz_trajs,
        "viz_seed": viz_seed,
        "eval_curves": {"checkpoints": ckpts, "threshold": THRESHOLD,
                        "conditions": curve_block},
        "t90_table": [{"condition": c,
                       "t90_iqm": conditions[c]["time_to_90_iqm"],
                       "ci": conditions[c]["time_to_90_ci"],
                       "n_censored": conditions[c]["n_censored"],
                       "n_seeds": conditions[c]["n_seeds"],
                       "final_success_iqm": conditions[c]["final_success_iqm"]}
                      for c in CONDITIONS],
        "practice_heatmaps": heatmaps,
        "self_coach": {"memory_k": MEMORY_K, "memory_p": MEMORY_P,
                       "phase_frac": SELF_FRAC,
                       "first_score_steps": first_scores,
                       "start_kind_totals": kinds},
    }
    return {"experiment": "exp6_selfcoach",
            "hypothesis": HYPOTHESIS,
            "config": config or {},
            "conditions": conditions,
            "curves": {"checkpoints": ckpts, "conditions": curve_block},
            "tests": {"alpha": 0.05,
                      "correction": ("holm-bonferroni over the 2-comparison "
                                     "primary family (MW and Welch families "
                                     "adjusted separately); verdicts count "
                                     "both statistics per DESIGN.md v3"),
                      "primary": primary,
                      "match_analysis": match,
                      "secondary_final_success": secondary,
                      "secondary_t90_context": context},
            "conclusion": {"claims": claims, "summary": summary},
            "viz": viz}


def main():
    ap = argparse.ArgumentParser(description="EXP6 self-coach (H6)")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default="results/exp6.json")
    ap.add_argument("--smoke", action="store_true",
                    help="2 seeds, 10x reduced budget")
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--seed-offset", type=int, default=100,
                    help="first seed; confirmatory runs use 100..(100+N-1), "
                         "disjoint from all dev/test seeds (0-50)")
    args = ap.parse_args()
    n_seeds, budget, eval_every = ((2, BUDGET // 10, EVAL_EVERY // 10)
                                   if args.smoke
                                   else (args.seeds, BUDGET, EVAL_EVERY))
    t0 = time.time()
    results = {}
    for cond in CONDITIONS:
        fn = functools.partial(run_seed_at, offset=args.seed_offset,
                               condition=cond, budget=budget,
                               eval_every=eval_every, cap=CAP)
        results[cond] = run_seeds(fn, n_seeds, n_jobs=args.jobs)
        print(f"{cond}: {n_seeds} seeds done at {time.time() - t0:.1f}s",
              flush=True)
    config = {"seeds": n_seeds, "seed_offset": args.seed_offset,
              "seed_list": list(range(args.seed_offset,
                                      args.seed_offset + n_seeds)),
              "budget": budget, "eval_every": eval_every, "cap": CAP,
              "n_eval_episodes": N_EVAL_EPISODES, "threshold": THRESHOLD,
              "lr": LR, "gamma": GAMMA, "eps": EPS, "q0": 0.0,
              "memory_k": MEMORY_K, "memory_p": MEMORY_P,
              "self_frac": SELF_FRAC, "match_margin_frac": MATCH_MARGIN,
              "teacher_protocol": ("EXP2 drills-varied and whole, imported "
                                   "from exp2_microtask.py (bit-identical "
                                   "reruns of EXP2's shipped arms)"),
              "eval_protocol": ("20 greedy kickoff episodes every "
                                f"{eval_every} steps on a separate env/rng "
                                "(default_rng([seed, step])); excluded from "
                                "all budgets"),
              "smoke": args.smoke, "jobs": args.jobs}
    out = assemble(results, budget, config=config)
    if args.smoke:
        out["conclusion"]["note"] = ("SMOKE RUN (2 seeds, 1/10 budget): "
                                     "non-confirmatory; verdicts here do not "
                                     "supersede the registered full run")
    out["config"]["wall_clock_s"] = round(time.time() - t0, 2)
    save_json(args.out, out)
    print(out["conclusion"]["summary"])


if __name__ == "__main__":
    main()
