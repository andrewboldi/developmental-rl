"""EXP4 — Generational Teaching (H4: iterated distillation), v3.

Claim: a lineage of short-lived agents, each distilling its best episodes to
a fresh, plastic student through a narrow (s, a) advice bottleneck, ratchets
past (a) weight-copy transfer and (b) one agent living the combined lifetime
with decaying plasticity. TrapGrid's terminal candy cells end episodes a few
steps from home, so the big goal is discovered occasionally but almost never
consolidated within one rigid-by-then lifetime; the bottleneck carries
exactly that rare peak episode into a young brain, which re-earns it.

v2 (post adversarial verification — see DESIGN.md "Amendments (v2) — EXP4"):
original five conditions unchanged (same rng stream indices), four arms added:

- generational-distill-shortest: distill under the rejected episodic-memory
  tie-break (ties keep the SHORTEST episode) — the v1 headline was found
  sensitive to this DESIGN-silent choice, so both rules are reported.
- random-advice: optimism-matched control — each generation's fresh student
  is primed Q[s, a] = 5.0 on 100 RANDOM non-terminal (s, a) pairs. Separates
  advice CONTENT from optimism scatter.
- optimistic-init: one long life, Q0 = 5.0 everywhere, standard decay —
  maximal content-free optimism in the long-life format.
- constant-eps-life: one long life, eps fixed at 0.4 forever, lr decaying as
  usual — does undying exploration rescue the long life? (claim 4).

v3 (see DESIGN.md "Amendments (v3) — EXP4 reset-with-replay identifiability
control"): two arms answer RESEARCH.md's "isn't this just resets?" objection
(Nikishin et al. 2022 reset networks but keep the FULL replay buffer; Ash &
Adams 2020 justify fresh students) — the distill advantage could be fresh
plasticity plus ANY inherited signal, not curated content:

- reset-replay-full: each generation a fresh student (fresh Q, fresh
  schedules) first replays the teacher's ENTIRE 15k-transition lifetime log,
  shuffled, as one pass of standard Q-updates at the student's initial lr
  (age untouched — full plasticity), then lives its own 15k steps. Raw
  experience inheritance at full dose.
- reset-replay-100: identical, but replaying 100 uniformly-sampled (without
  replacement) transitions — dose-matched to the 100-pair advice bottleneck.

Budgets matched EXACTLY: every condition trains gens * life = 75k env steps
(replay pretraining, like advice priming, consumes no env steps — the
inheritance channel is budget-free in every arm; see config.free_inheritance).
Conclusion is per-claim ({claim, verdict, evidence}); every registered
comparison reports BOTH Mann-Whitney and Welch t, Holm within its family.
Confirmatory seeds are offset (default 100 -> seeds 100..159), disjoint from
every seed ever used to tune this experiment (0-29).

Aging: lr(age) = 0.3 * 2^(-age/halflife), eps(age) = 0.4 * 2^(-age/halflife).

Eval at each generation boundary is one greedy rollout with epsilon forced
to 0 and argmax tie-breaking (env and policy both deterministic, so the
20-rollout protocol collapses to its single support point), plus the
Q-derived start value. Eval consumes no training budget, uses no rng, and is
byte-identical across conditions.

Usage: python experiments/exp4_generations.py --seeds 60 --seed-offset 100 \
           --out results/exp4.json [--smoke] [--jobs 20]
"""

import argparse
import time
from collections import Counter
from functools import partial

import numpy as np
from scipy.stats import ttest_ind

from devrl.agents.distill import (EpisodicMemory, TransitionLog, apply_advice,
                                  extract_advice, halflife_schedule,
                                  replay_pretrain)
from devrl.agents.qlearning import QLearner
from devrl.envs.trapgrid import TRAP_MAP, TrapGrid
from devrl.run import run_seeds, save_json
from devrl.stats import bootstrap_ci, iqm, mann_whitney

# Original five first (their per-condition rng stream indices are unchanged
# from v1), v2 arms next, v3 reset-replay arms appended — every pre-existing
# condition keeps its stream index, so v1/v2 arms are bit-identical reruns.
CONDITIONS = ["generational-distill", "weight-copy", "one-long-life",
              "one-long-life-slow", "no-inheritance",
              "generational-distill-shortest", "random-advice",
              "optimistic-init", "constant-eps-life",
              "reset-replay-full", "reset-replay-100"]

LONG_LIFE_CONDITIONS = ("one-long-life", "one-long-life-slow",
                        "optimistic-init", "constant-eps-life")
DISTILL_CONDITIONS = ("generational-distill", "generational-distill-shortest")
ADVICE_CONDITIONS = DISTILL_CONDITIONS + ("random-advice",)
REPLAY_CONDITIONS = ("reset-replay-full", "reset-replay-100")

CONDITION_LABELS = {
    "generational-distill": "Generational distill (earliest-tie memory)",
    "weight-copy": "Weight copy",
    "one-long-life": "One long life",
    "one-long-life-slow": "One long life (slow decay)",
    "no-inheritance": "No inheritance",
    "generational-distill-shortest": "Generational distill (shortest-tie memory)",
    "random-advice": "Random advice (optimism-matched)",
    "optimistic-init": "Optimistic init (Q0=5 long life)",
    "constant-eps-life": "Constant-eps long life",
    "reset-replay-full": "Reset + full-log replay (raw inheritance)",
    "reset-replay-100": "Reset + 100-transition replay (dose-matched)",
}

# replay_dose == advice_cap by construction: the raw-experience control's
# sampled dose is matched to the advice bottleneck's 100 primed cells.
FULL = dict(gens=5, life=15000, halflife=5000, slow_halflife=25000,
            lr0=0.3, eps0=0.4, gamma=0.99, cap=120,
            memory_k=3, advice_cap=100, advice_value=5.0, replay_dose=100,
            n_boot=10000)
SMOKE = dict(FULL, life=1500, halflife=500, slow_halflife=2500, n_boot=2000)

CURVE_METRICS = ["greedy_return", "big_goal", "gap", "q_start"]
ACTION_NAMES = ["up", "right", "down", "left"]

_GRID = TrapGrid()
_START_STATE = _GRID.state_of(_GRID.start)

HYPOTHESIS = ("H4: a lineage of short-lived agents, each distilling its top "
              "episodes to a fresh plastic student through a narrow advice "
              "bottleneck, ratchets past weight-copy transfer and past single "
              "long lives with decaying plasticity.")

EVAL_PROTOCOL = (
    "One deterministic greedy rollout per generation boundary (eps=0, "
    "np.argmax with lowest-index tie-break; env deterministic, so repeated "
    "rollouts collapse to a single support point). This is a fixed-tie-break "
    "proxy for the agent's own randomized-tie greedy policy: per-seed 0.0 "
    "finals mean the argmax policy loops, not necessarily every greedy "
    "realization. Eval consumes no training budget, draws no agent rng, and "
    "is byte-identical across conditions.")

FREE_INHERITANCE = (
    "budget symmetry: replay pretraining (reset-replay-*) performs its "
    "Q-updates from the dead teacher's transition log at the student's "
    "initial lr with the student's age untouched — it consumes no env steps "
    "and no plasticity, free exactly the way advice priming's 100 Q-cell "
    "writes are free. Every condition trains exactly gens*life env steps; "
    "the inheritance channel (100 curated pairs, 100 raw transitions, the "
    "full 15k-transition log, or a full Q copy) is the manipulated variable "
    "and is budget-free in every arm.")

# Registered primary family (Holm within family, m=9): the four v1
# comparisons unchanged, the two v2 content-vs-optimism controls, the two v3
# reset-with-replay identifiability controls, and the claim-4 rescue test.
PRIMARY_COMPARISONS = [
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

# Tie-break robustness family (Holm within family, m=7): the shortest-tie
# distill arm against every baseline, plus the two distill variants against
# each other.
ROBUSTNESS_COMPARISONS = [
    ("generational-distill-shortest", "weight-copy"),
    ("generational-distill-shortest", "one-long-life"),
    ("generational-distill-shortest", "one-long-life-slow"),
    ("generational-distill-shortest", "no-inheritance"),
    ("generational-distill-shortest", "random-advice"),
    ("generational-distill-shortest", "optimistic-init"),
    ("generational-distill", "generational-distill-shortest"),
]

CLAIM_NAMES = (
    "peak-experience distillation ratchets across generations",
    "the bottleneck beats weight copying",
    "advice content matters beyond optimism scatter",
    "plasticity decay is what strands the long life",
    "curated advice beats raw experience inheritance at matched plasticity",
)


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


def _agent_for(cond, cfg, rng):
    """Fresh QLearner configured for a condition.

    - one-long-life-slow: both schedules decay with the slow halflife.
    - constant-eps-life: eps fixed at eps0 forever; lr decays as usual.
    - optimistic-init: Q0 = advice_value everywhere (5.0), standard decay.
    - everything else: standard decaying schedules, Q0 = 0.
    """
    hl = cfg["slow_halflife"] if cond == "one-long-life-slow" else cfg["halflife"]
    eps = (cfg["eps0"] if cond == "constant-eps-life"
           else halflife_schedule(cfg["eps0"], hl))
    q0 = cfg["advice_value"] if cond == "optimistic-init" else 0.0
    return QLearner(n_states=_GRID.n_states, n_actions=TrapGrid.n_actions,
                    lr=halflife_schedule(cfg["lr0"], hl),
                    gamma=cfg["gamma"], eps=eps,
                    optimistic_init=q0, rng=rng)


def _random_advice(rng, cap):
    """`cap` distinct random (s, a) pairs over actable non-terminal cells.

    Optimism-matched control for the distill bottleneck: same pair count,
    same Q=5.0 blessing, but the content is scatter, not a remembered path.
    Walls, candy cells, and the goal are excluded — the agent never acts
    from those states, so priming them would waste the optimism dose and
    undermine the match.
    """
    eligible = []
    for row in range(_GRID.H):
        for col in range(_GRID.W):
            rc = (row, col)
            if (_GRID.grid[rc] != "#" and rc not in _GRID.candies
                    and rc != _GRID.goal):
                s = _GRID.state_of(rc)
                eligible.extend((s, a) for a in range(TrapGrid.n_actions))
    idx = rng.choice(len(eligible), size=cap, replace=False)
    return [eligible[int(i)] for i in idx]


def _live(agent, env, memory, n_steps, carry=None, log=None):
    """Exactly n_steps of eps-greedy life. Completed episodes (terminal or
    cap-truncated) enter memory; the mid-episode carry is returned so a long
    life can span checkpoint boundaries without an artificial reset. A
    TransitionLog `log` records every (s, a, r, s2, done) exactly as fed to
    the Q-update (the raw-inheritance channel for the reset-replay arms)."""
    s, sa, ret = carry if carry is not None else (env.reset(), [], 0.0)
    for _ in range(n_steps):
        a = agent.act(s)
        s2, r, done, info = env.step(a)
        agent.update(s, a, r, s2, done)
        if log is not None:
            log.add(s, a, r, s2, done)
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
    env = TrapGrid(cap=cfg["cap"])
    per_gen, steps_trained, final_traj, out = [], 0, None, {}
    if cond in LONG_LIFE_CONDITIONS:
        agent = _agent_for(cond, cfg, rng)
        memory = EpisodicMemory(cfg["memory_k"])
        carry = None
        for _ in range(gens):
            carry = _live(agent, env, memory, life, carry)
            steps_trained += life
            m, final_traj = _gen_metrics(agent, memory)
            per_gen.append(m)
    else:
        tie = "shortest" if cond == "generational-distill-shortest" else "earliest"
        advice, q_prev, advice_by_gen = None, None, []
        prev_log, replayed_by_gen = None, []
        for _ in range(gens):
            agent = _agent_for(cond, cfg, rng)
            if cond == "weight-copy" and q_prev is not None:
                agent.Q[:] = q_prev
            if cond == "random-advice":
                # fresh random scatter EVERY generation (gen 1 included):
                # the control always gets its full optimism dose, which can
                # only flatter it relative to distill's advice-free gen 1.
                advice = _random_advice(rng, cfg["advice_cap"])
                advice_by_gen.append(advice)
                apply_advice(agent.Q, advice, cfg["advice_value"])
            elif cond in DISTILL_CONDITIONS and advice:
                apply_advice(agent.Q, advice, cfg["advice_value"])
            elif cond in REPLAY_CONDITIONS:
                # raw experience inheritance (the reset-with-replay control):
                # the fresh student first replays the dead teacher's log —
                # the WHOLE 15k-transition lifetime shuffled, or replay_dose
                # uniform samples — as one pass of standard Q-updates at its
                # initial lr, with its age (hence schedules) untouched.
                # Replay, like advice priming, consumes no env steps.
                # Generation 1 has no teacher and replays nothing, matching
                # distill's advice-free generation 1.
                batch = []
                if prev_log is not None:
                    batch = (prev_log.shuffled(rng)
                             if cond == "reset-replay-full"
                             else prev_log.sample(rng, cfg["replay_dose"]))
                    replay_pretrain(agent.Q, batch,
                                    lr=cfg["lr0"], gamma=cfg["gamma"])
                replayed_by_gen.append(len(batch))
            memory = EpisodicMemory(cfg["memory_k"], tie_break=tie)
            log = TransitionLog() if cond in REPLAY_CONDITIONS else None
            _live(agent, env, memory, life, log=log)
            steps_trained += life
            m, final_traj = _gen_metrics(agent, memory)
            per_gen.append(m)
            if cond in DISTILL_CONDITIONS:
                # advice EMITTED at this generation's death (applied to the
                # next student); for random-advice the entries above are the
                # scatter APPLIED to each generation instead.
                advice = extract_advice(memory, cfg["advice_cap"])
                advice_by_gen.append(advice)
            elif cond == "weight-copy":
                q_prev = agent.Q.copy()
            elif cond in REPLAY_CONDITIONS:
                prev_log = log
        if cond in ADVICE_CONDITIONS:
            out["advice_by_gen"] = advice_by_gen
        if cond in REPLAY_CONDITIONS:
            out["replayed_by_gen"] = replayed_by_gen
    out.update(steps_trained=steps_trained, per_gen=per_gen, final_traj=final_traj)
    return out


def _run_seed(seed, cfg, offset=0):
    """All conditions for one seed. Each condition draws from its own
    seed-derived rng stream and trains exactly gens*life env steps. `offset`
    shifts the whole stream family: offset=100, seed=0 is exactly the stream
    a plain seed-100 run would use (fresh confirmatory seeds 100..159)."""
    return {cond: _run_condition(cond, np.random.default_rng([seed + offset, ci]), cfg)
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


def _welch_safe(a, b):
    """Welch's t (scipy ttest_ind, equal_var=False) with degenerate guards:
    both groups constant and equal -> t=0, p=1 (no evidence either way);
    both constant but different -> deterministic separation, p=0 with t=None
    (never Infinity, which would break strict JSON)."""
    av, bv = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if av.var() == 0.0 and bv.var() == 0.0:
        if av[0] == bv[0]:
            return {"t": 0.0, "p": 1.0}
        return {"t": None, "p": 0.0}
    t, p = ttest_ind(av, bv, equal_var=False)
    return {"t": float(t), "p": float(p)}


def _tests_for(pairs, finals, fi, fm):
    """Dual-test comparison entries for one registered family: Mann-Whitney
    and Welch t, each Holm-adjusted within the family separately.
    `significant` (the registered decision level) requires BOTH."""
    raw_mw = [_mw_safe(finals[a], finals[b]) for a, b in pairs]
    raw_w = [_welch_safe(finals[a], finals[b]) for a, b in pairs]
    mw_holm = _holm([t["p"] for t in raw_mw])
    w_holm = _holm([t["p"] for t in raw_w])
    comps = []
    for (a, b), mw, w, pmh, pwh in zip(pairs, raw_mw, raw_w, mw_holm, w_holm):
        smw, sw = bool(pmh < 0.05), bool(pwh < 0.05)
        comps.append({
            "a": a, "b": b, "iqm_a": fi[a], "iqm_b": fi[b],
            "mean_a": fm[a], "mean_b": fm[b],
            "u": mw["u"], "p_mw": mw["p"], "p_mw_holm": float(pmh),
            "p": mw["p"], "p_holm": float(pmh),  # v1 aliases (MW is primary)
            "t": w["t"], "p_welch": w["p"], "p_welch_holm": float(pwh),
            "significant_mw": smw, "significant_welch": sw,
            "significant": bool(smw and sw)})
    return comps


def _comp_verdict(comp):
    """Registered verdict for one comparison a-vs-b on final greedy return:
    supported iff BOTH tests Holm-significant with a above b; refuted iff
    both significant with a below; boundary iff exactly one test is
    significant (the two tests disagree); null iff neither. Direction is
    IQM, falling back to means when IQMs tie."""
    hi, lo = comp["iqm_a"], comp["iqm_b"]
    if hi == lo:
        hi, lo = comp["mean_a"], comp["mean_b"]
    if comp["significant_mw"] and comp["significant_welch"]:
        if hi > lo:
            return "supported"
        return "refuted" if hi < lo else "boundary"
    if comp["significant_mw"] or comp["significant_welch"]:
        return "boundary"
    return "null"


def _combine_verdicts(v1, v2):
    """Conjunctive claim over two comparisons: any refuted -> refuted; both
    supported -> supported; both null -> null; mixed -> boundary."""
    if "refuted" in (v1, v2):
        return "refuted"
    if v1 == v2 and v1 in ("supported", "null"):
        return v1
    return "boundary"


# Claim 4 is a rescue test: constant-eps-life keeps exploration plasticity
# alive forever, so FAILURE to beat one-long-life is itself informative —
# a null comparison refutes "plasticity decay strands the long life", it
# does not merely fail to support it (registered in DESIGN.md v2).
_CLAIM4_MAP = {"supported": "supported", "boundary": "boundary",
               "null": "refuted", "refuted": "refuted"}


def _replay_story(v_full, v_100):
    """Registered honesty clause for claim 5 (DESIGN v3): a fixed map from
    the two comparison verdicts (distill vs reset-replay-full, distill vs
    reset-replay-100) to the interpretation sentence — in particular, if the
    full raw log matches curated advice, the bottleneck story must be
    narrowed to dose efficiency (or dropped), and this says so plainly."""
    if v_full == "supported" and v_100 == "supported":
        return ("Curation beats raw experience at full dose and at matched "
                "dose: the bottleneck's advantage is content selection, not "
                "fresh plasticity plus any inherited signal.")
    if v_full == "refuted":
        return ("Raw full-log replay BEATS curated advice: the bottleneck "
                "story fails — fresh plasticity plus unfiltered experience "
                "is the better inheritance channel here.")
    if v_full in ("null", "boundary"):
        if v_100 == "supported":
            return ("Full-log replay matches distill, so the bottleneck "
                    "story NARROWS to dose efficiency: curation shows its "
                    "value only at the matched 100-item dose, where curated "
                    "pairs beat raw uniform transitions.")
        return ("Full-log replay matches distill and the dose-matched raw "
                "sample is not reliably beaten either: the distill advantage "
                "is not identifiable as curated content on this evidence.")
    # v_full == "supported", v_100 in {null, boundary, refuted}
    if v_100 == "refuted":
        return ("Curation beats the unfiltered full log but LOSES to 100 raw "
                "uniform transitions at matched dose: content selection "
                "helps against bulk inheritance, yet the curated content "
                "itself is worse than random experience at matched dose.")
    return ("Curation beats the unfiltered full log, but at matched dose "
            "the curated-vs-raw difference is not established.")


def _fmt_comp(c):
    return ("IQM %.2f vs %.2f (means %.2f vs %.2f), MW p=%.3g (Holm %.3g), "
            "Welch p=%.3g (Holm %.3g)"
            % (c["iqm_a"], c["iqm_b"], c["mean_a"], c["mean_b"],
               c["p_mw"], c["p_mw_holm"], c["p_welch"], c["p_welch_holm"]))


def _distill_facts(results, cfg, finals):
    """Descriptive facts for the evidence strings: final-outcome histogram,
    per-generation goal rate, and peak transmission/retention counts for the
    primary distill arm."""
    gens = cfg["gens"]
    hist = Counter(finals["generational-distill"])
    goal_rate = [float(np.mean([r["generational-distill"]["per_gen"][g]["big_goal"]
                                for r in results])) for g in range(gens)]
    handoffs = consolidated = losses = 0
    for r in results:
        pg = r["generational-distill"]["per_gen"]
        for g in range(gens - 1):
            if pg[g]["best_memory_return"] == TrapGrid.GOAL_REWARD:
                handoffs += 1
                if pg[g + 1]["greedy_return"] == TrapGrid.GOAL_REWARD:
                    consolidated += 1
                if pg[g + 1]["best_memory_return"] < TrapGrid.GOAL_REWARD:
                    losses += 1
    return {"final_hist": hist, "goal_rate": goal_rate, "handoffs": handoffs,
            "consolidated": consolidated, "losses": losses}


def _claims(results, cfg, finals, comp_by, seeds):
    """The four registered claims, each decided by its mandated comparisons."""
    facts = _distill_facts(results, cfg, finals)
    n_goal = {cond: int(sum(r[cond]["per_gen"][-1]["big_goal"] for r in results))
              for cond in CONDITIONS}
    hist_txt = ", ".join("%dx%.1f" % (n, v)
                         for v, n in sorted(facts["final_hist"].items()))

    c1 = comp_by[("generational-distill", "no-inheritance")]
    ev1 = ("Distill vs no-inheritance: %s. Goal consolidated in %d/%d distill "
           "lineages vs %d/%d; per-generation P(goal) %s. The ratchet is peak "
           "RETENTION, not monotone aggregate improvement: %d goal-bearing "
           "advice hand-offs, %d consolidated by the next student, %d peak-loss "
           "events. Final distill outcomes are bimodal (%s): failed lineages "
           "end at 0.0, BELOW the 0.3 trap floor every baseline sits on."
           % (_fmt_comp(c1), n_goal["generational-distill"], seeds,
              n_goal["no-inheritance"], seeds,
              " -> ".join("%.2f" % g for g in facts["goal_rate"]),
              facts["handoffs"], facts["consolidated"], facts["losses"],
              hist_txt))

    c2 = comp_by[("generational-distill", "weight-copy")]
    ev2 = ("Distill vs weight-copy: %s. Goal consolidated in %d/%d vs %d/%d "
           "lineages: full Q inheritance hands the student trap-shaped values "
           "with fresh schedules and never consolidates the goal."
           % (_fmt_comp(c2), n_goal["generational-distill"], seeds,
              n_goal["weight-copy"], seeds))

    c3a = comp_by[("generational-distill", "random-advice")]
    c3b = comp_by[("generational-distill", "optimistic-init")]
    v3a, v3b = _comp_verdict(c3a), _comp_verdict(c3b)
    ev3 = ("Conjunctive: distill must beat BOTH optimism controls. Vs "
           "random-advice (100 random pairs primed to %.1f each generation): "
           "%s [%s]. Vs optimistic-init (Q0=%.1f everywhere, one long life): "
           "%s [%s]. Final goal counts: distill %d/%d, random-advice %d/%d, "
           "optimistic-init %d/%d."
           % (cfg["advice_value"], _fmt_comp(c3a), v3a, cfg["advice_value"],
              _fmt_comp(c3b), v3b, n_goal["generational-distill"], seeds,
              n_goal["random-advice"], seeds, n_goal["optimistic-init"], seeds))

    c4 = comp_by[("constant-eps-life", "one-long-life")]
    v4c = _comp_verdict(c4)
    ev4 = ("Rescue test: constant-eps-life (eps fixed %.1f forever, lr "
           "decaying as usual) vs one-long-life: %s [comparison verdict %s; "
           "a null RESCUE is registered as refuting the claim — undying "
           "exploration failing to un-strand the long life means exploration "
           "decay was not the binding constraint]. Final goal counts: "
           "constant-eps-life %d/%d, one-long-life %d/%d, one-long-life-slow "
           "%d/%d. Caveat: lr still decays in this arm; but weight-copy "
           "(fully fresh schedules, inherited values) and one-long-life-slow "
           "(5x slower decay of both) locate the strand in consolidated "
           "trap values rather than lost plasticity."
           % (cfg["eps0"], _fmt_comp(c4), v4c,
              n_goal["constant-eps-life"], seeds, n_goal["one-long-life"],
              seeds, n_goal["one-long-life-slow"], seeds))

    c5a = comp_by[("generational-distill", "reset-replay-full")]
    c5b = comp_by[("generational-distill", "reset-replay-100")]
    v5a, v5b = _comp_verdict(c5a), _comp_verdict(c5b)
    # descriptive accounting (no verdict rides on it): how often a replay
    # teacher's memory held a 10.0 episode — so its log CONTAINED goal
    # transitions handed to the next student — and how often the successor
    # consolidated; guards against the "replay arms never found the goal"
    # misreading either way.
    rh = {}
    for cond in REPLAY_CONDITIONS:
        h = consolidated = 0
        for r in results:
            pg = r[cond]["per_gen"]
            for g in range(cfg["gens"] - 1):
                if pg[g]["best_memory_return"] == TrapGrid.GOAL_REWARD:
                    h += 1
                    if pg[g + 1]["greedy_return"] == TrapGrid.GOAL_REWARD:
                        consolidated += 1
        rh[cond] = (h, consolidated)
    ev5 = ("Reset-with-replay identifiability control (the 'isn't this just "
           "resets?' objection): both replay students get exactly distill's "
           "fresh plasticity (fresh Q, fresh schedules; replay runs at the "
           "initial lr %.1f with age untouched and costs no env steps) plus "
           "RAW experience instead of curated advice — the teacher's entire "
           "%d-transition lifetime log shuffled (reset-replay-full) or %d "
           "uniform transitions (reset-replay-100, dose-matched to the "
           "%d-pair advice cap). Vs reset-replay-full: %s [%s]. Vs "
           "reset-replay-100: %s [%s]. Final goal counts: distill %d/%d, "
           "reset-replay-full %d/%d, reset-replay-100 %d/%d. "
           "Discovery-vs-consolidation (descriptive): goal-bearing hand-offs "
           "(teacher memory held a 10.0 episode, so its log contained goal "
           "transitions; the full arm replays that entire log) "
           "reset-replay-full %d (consolidated by the successor %d), "
           "reset-replay-100 %d (consolidated %d), distill %d "
           "(consolidated %d). %s"
           % (cfg["lr0"], cfg["life"], cfg["replay_dose"], cfg["advice_cap"],
              _fmt_comp(c5a), v5a, _fmt_comp(c5b), v5b,
              n_goal["generational-distill"], seeds,
              n_goal["reset-replay-full"], seeds,
              n_goal["reset-replay-100"], seeds,
              rh["reset-replay-full"][0], rh["reset-replay-full"][1],
              rh["reset-replay-100"][0], rh["reset-replay-100"][1],
              facts["handoffs"], facts["consolidated"],
              _replay_story(v5a, v5b)))

    return [
        {"claim": CLAIM_NAMES[0], "verdict": _comp_verdict(c1), "evidence": ev1},
        {"claim": CLAIM_NAMES[1], "verdict": _comp_verdict(c2), "evidence": ev2},
        {"claim": CLAIM_NAMES[2], "verdict": _combine_verdicts(v3a, v3b),
         "evidence": ev3},
        {"claim": CLAIM_NAMES[3], "verdict": _CLAIM4_MAP[v4c], "evidence": ev4},
        {"claim": CLAIM_NAMES[4], "verdict": _combine_verdicts(v5a, v5b),
         "evidence": ev5},
    ]


def _aggregate(results, cfg, seeds, wall_s, offset=0, smoke=False):
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
    curves["big_goal"]["stat_note"] = (
        "binary metric: plot the mean field (P(big goal)); the iqm field is "
        "a trimmed proportion of a 0/1 variable and overstates success rate")

    finals = {cond: [r[cond]["per_gen"][-1]["greedy_return"] for r in results]
              for cond in CONDITIONS}
    fi = {cond: iqm(finals[cond]) for cond in CONDITIONS}
    fm = {cond: float(np.mean(finals[cond])) for cond in CONDITIONS}
    primary = _tests_for(PRIMARY_COMPARISONS, finals, fi, fm)
    robustness = _tests_for(ROBUSTNESS_COMPARISONS, finals, fi, fm)
    tests = {"metric": "final greedy return (generation %d)" % cfg["gens"],
             "correction": ("holm-bonferroni within each family (primary m=%d, "
                            "robustness m=%d), applied to MW and Welch "
                            "separately" % (len(primary), len(robustness))),
             "alpha": 0.05,
             "decision_rule": ("registered significance requires BOTH the "
                               "Holm-adjusted Mann-Whitney AND Holm-adjusted "
                               "Welch p below alpha"),
             "comparisons": primary,
             "robustness_comparisons": robustness}

    comp_by = {(c["a"], c["b"]): c for c in primary + robustness}
    claims = _claims(results, cfg, finals, comp_by, seeds)
    distill_iqms = curves["greedy_return"]["conditions"]["generational-distill"]["iqm"]
    facts = _distill_facts(results, cfg, finals)
    hist_txt = ", ".join("%dx%.1f" % (n, v)
                         for v, n in sorted(facts["final_hist"].items()))
    short = comp_by[("generational-distill", "generational-distill-shortest")]
    summary = ("Final greedy-return IQMs: "
               + ", ".join("%s=%.2f" % (c, fi[c]) for c in CONDITIONS)
               + ". Distill per-generation IQM " +
               " -> ".join("%.2f" % v for v in distill_iqms)
               + "; final outcomes bimodal (%s)." % hist_txt
               + " Tie-break robustness: earliest vs shortest %s." % _fmt_comp(short)
               + " Claims: "
               + "; ".join("[%s] %s" % (c["verdict"], c["claim"])
                           for c in claims) + ".")
    conclusion = {"claims": claims, "summary": summary}
    if smoke:
        conclusion["note"] = (
            "smoke scale (1/10 budget): not confirmatory — per-life goal "
            "discovery collapses at this budget, so verdicts here can invert "
            "the full run's; use results/exp4.json for any conclusion")

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
        "condition_labels": CONDITION_LABELS,
        "per_generation_curves": curves,
        "representative_seed": rep + offset,
        "advice_by_generation": advice_viz,
        "final_greedy_paths": {cond: results[rep][cond]["final_traj"]
                               for cond in CONDITIONS},
        "per_gen_representative": {cond: results[rep][cond]["per_gen"]
                                   for cond in CONDITIONS},
    }

    return {"experiment": "exp4_generations",
            "hypothesis": HYPOTHESIS,
            "config": {**cfg, "seeds": seeds, "seed_offset": offset,
                       "smoke": bool(smoke), "conditions": CONDITIONS,
                       "total_budget": cfg["gens"] * cfg["life"],
                       "eval_protocol": EVAL_PROTOCOL,
                       "free_inheritance": FREE_INHERITANCE,
                       "wall_clock_s": round(wall_s, 2)},
            "conditions": {cond: {"seeds": [dict(seed=i + offset,
                                                 **results[i][cond])
                                            for i in range(seeds)]}
                           for cond in CONDITIONS},
            "curves": curves,
            "tests": tests,
            "conclusion": conclusion,
            "viz": viz}


def main():
    ap = argparse.ArgumentParser(description="EXP4 generational teaching (v2)")
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--seed-offset", type=int, default=100, dest="seed_offset",
                    help="first seed = offset (confirmatory range 100..159, "
                         "disjoint from all tuning seeds 0-29)")
    ap.add_argument("--out", default="results/exp4.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()
    cfg = SMOKE if args.smoke else FULL
    seeds = 2 if args.smoke else args.seeds
    t0 = time.time()
    results = run_seeds(partial(_run_seed, cfg=cfg, offset=args.seed_offset),
                        seeds, n_jobs=min(args.jobs, seeds))
    out = _aggregate(results, cfg, seeds, time.time() - t0,
                     offset=args.seed_offset, smoke=args.smoke)
    save_json(args.out, out)
    verdicts = ", ".join("claim%d=%s" % (i + 1, c["verdict"])
                         for i, c in enumerate(out["conclusion"]["claims"]))
    print("wrote %s (%d seeds, offset %d, %.1fs) %s" % (
        args.out, seeds, args.seed_offset, time.time() - t0, verdicts))


if __name__ == "__main__":
    main()
