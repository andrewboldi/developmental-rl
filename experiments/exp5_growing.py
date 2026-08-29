"""EXP5 — Growing Bodies (H5: morphological curriculum), v2.

v1 claim: training a small body first (falls are cheap, relative strength is
high — square-cube law) and growing toward the adult body reaches adult
competence with far less cumulative damage; balance-first should beat
walking-from-the-start. Adversarial verification (3 lenses) invalidated the
v1 *claim structure* (a single supported boolean over a conjunction whose
parts came out differently), not the damage effect. v2 (see DESIGN.md
"Amendments (v2) — EXP5"):

- Seven conditions: the missing factorial cell `grow-adaptive-walk` completes
  the growth x task-staging factorial.
- PRIMARY family = the confound-free walk-direct pairing (grow-linear-walk &
  grow-adaptive-walk vs adult-walk, on damage AND steps); the v1 family is
  retained as SECONDARY.
- Per-claim verdicts replace the boolean: damage_at_competence, steps_parity
  (equivalence view with a +20% margin — never "accept the null"),
  gradualism, balance_first.
- Both Mann-Whitney AND Welch t reported for every test, Holm within family.
- Physics-robustness arms (reduced seed count, labeled, non-confirmatory):
  tau3 (tau_max = 40 s^3, muscle-torque law) and damp2 (b = 1.0 s^2,
  size-scaled damping) — is the damage ordering stable?
- Fresh confirmatory seeds 100..(100+N-1) via --seed-offset, disjoint from
  the v1 tuning seeds (0-9) and every seed used to probe v1.
- Standing-only guard: eval also records time-on-target fractions (overall
  and on lean targets, which a pure stander cannot earn).

Damage = sum of (s/1)^4 over training falls, recorded separately from reward.
Eval is ALWAYS the adult body (s=1.0), greedy, walking targets, every
EVAL_EVERY steps, identical across conditions and excluded from the training
budget. Competence = eval return >= 500. Budgets matched exactly at `BUDGET`
training steps for every condition.

    python experiments/exp5_growing.py --seeds 40 --out results/exp5.json
    python experiments/exp5_growing.py --smoke --out results/exp5_smoke.json
"""

import argparse
import functools
import time
from collections import deque

import numpy as np
from scipy.stats import ttest_ind

from devrl.agents.qlearning import QLearner
from devrl.envs.balance import BalanceBot, CAP, FALL_ANGLE, TARGETS
from devrl.run import run_seeds, save_json
from devrl.stats import bootstrap_ci, iqm, mann_whitney, time_to_threshold

CONDITIONS = ("adult-walk", "adult-balance-first", "grow-linear",
              "grow-adaptive", "grow-jump", "grow-linear-walk",
              "grow-adaptive-walk")
DESCRIPTIONS = {
    "adult-walk": "s=1.0 throughout, walking task from the start",
    "adult-balance-first": "s=1.0; target fixed at 0 for first 30%, then walking",
    "grow-linear": "s 0.5->1.0 over first 60% of budget; balance-first",
    "grow-adaptive": "grow s by 0.05 when rolling no-fall rate > 70%; balance-first",
    "grow-jump": "s=0.5 for 60% of budget then s=1.0; balance-first",
    "grow-linear-walk": "grow-linear schedule but walking task from the start",
    "grow-adaptive-walk": "grow-adaptive gate but walking task from the start",
}
HYPOTHESIS = ("H5: a morphological curriculum — train a small body first and "
              "grow toward the adult — reaches adult competence with far less "
              "cumulative damage (and no slower), and balance-first beats "
              "walking-from-the-start. v2 scores these as four independent "
              "claims: damage_at_competence and steps_parity on the "
              "confound-free walk-direct pairing (growth vs adult, both "
              "walking from the start), plus gradualism and balance_first.")

BUDGET = 120000
SMOKE_BUDGET = 12000
THRESHOLD = 500.0
N_EVAL = 5
TRACE_LEN = 200
MARGIN_FRAC = 0.20  # steps-parity equivalence margin: +20% of adult IQM
# One agent config shared by every condition (tuned once on v1 seeds 0-9,
# applied uniformly; confirmatory seeds are offset to stay disjoint):
LR = 0.25
GAMMA = 0.99
EPS0, EPS_FLOOR, EPS_HALFLIFE_FRAC = 0.5, 0.08, 0.25
# grow-adaptive*: +0.05 size whenever the last GROW_WINDOW episodes had a
# no-fall rate above GROW_RATE ("grow when ready"); window resets on growth.
GROW_WINDOW, GROW_RATE, GROW_STEP = 20, 0.7, 0.05
ADAPTIVE = ("grow-adaptive", "grow-adaptive-walk")

# Physics-robustness variants: name -> (tau_exp, damp_exp).
#   default: tau_max = 40 s^2, b = 1.0        (DESIGN v1 laws)
#   tau3:    tau_max = 40 s^3, b = 1.0        (muscle torque ~ L^3)
#   damp2:   tau_max = 40 s^2, b = 1.0 s^2    (size-scaled damping)
# All variants coincide at s=1.0, so adult-only arms are variant-independent.
VARIANTS = {"default": (2, 0), "tau3": (3, 0), "damp2": (2, 2)}
ROBUST_VARIANTS = ("tau3", "damp2")
ROBUST_CONDITIONS = ("grow-linear", "grow-adaptive", "grow-linear-walk",
                     "grow-adaptive-walk")


def _eps(age, budget):
    return max(EPS_FLOOR, EPS0 * 2 ** (-age / (EPS_HALFLIFE_FRAC * budget)))


def size_schedule(cond, t, budget):
    """Body size at training step t for the schedule-driven conditions.

    grow-adaptive* are event-driven (see run_one), not scheduled.
    """
    if cond in ADAPTIVE:
        raise ValueError(f"{cond} size is event-driven, not scheduled")
    if cond in ("grow-linear", "grow-linear-walk"):
        return 0.5 + 0.5 * min(t / (0.6 * budget), 1.0)
    if cond == "grow-jump":
        return 0.5 if t < 0.6 * budget else 1.0
    return 1.0  # adult-walk, adult-balance-first


def walk_start(cond, budget):
    """Training step at which the walking task begins (0 = from the start)."""
    if cond in ("adult-walk", "grow-linear-walk", "grow-adaptive-walk"):
        return 0
    return int(0.3 * budget)


def evaluate(Q, seed, k, n_eval):
    """Greedy eval on the ADULT body with walking targets.

    Seeded by (seed, checkpoint) only — never by condition or variant — so
    every condition faces the identical eval protocol; runs in a separate env
    and consumes no training budget. Greedy = argmax (no agent rng touched).
    Returns mean return plus time-on-target fractions: `ontarget` = fraction
    of eval steps within 0.05 rad of the current target, `lean_ontarget` =
    the same restricted to steps whose target is nonzero — the standing-only
    guard: a pure stander cannot earn lean-target time (RESEARCH.md flags the
    alive-bonus local optimum).
    """
    env = BalanceBot(s=1.0, mode="walk",
                     rng=np.random.default_rng([777, seed, k]))
    rets = []
    steps = hits = lean_steps = lean_hits = 0
    for _ in range(n_eval):
        s, ret, done = env.reset(), 0.0, False
        while not done:
            s, r, done, info = env.step(int(np.argmax(Q[s])))
            ret += r
            steps += 1
            if not info["fall"]:
                hit = abs(info["theta"] - info["target"]) < 0.05
                hits += hit
                if info["target"] != 0.0:
                    lean_steps += 1
                    lean_hits += hit
        rets.append(ret)
    return {"ret": float(np.mean(rets)),
            "ontarget": hits / steps if steps else 0.0,
            "lean_ontarget": lean_hits / lean_steps if lean_steps else 0.0}


def greedy_trace(Q, seed, trace_len):
    """theta/target/action trace of the final policy on the adult walking task."""
    env = BalanceBot(s=1.0, mode="walk", rng=np.random.default_rng([888, seed]))
    s = env.reset()
    thetas, targets, actions, fell = [], [], [], False
    for _ in range(trace_len):
        a = int(np.argmax(Q[s]))
        s, _, done, info = env.step(a)
        thetas.append(float(info["theta"]))
        targets.append(float(info["target"]))
        actions.append(a)
        if done:
            fell = bool(info["fall"])
            break
    return {"theta": thetas, "target": targets, "action": actions, "fell": fell}


def time_to_durable(steps, vals, threshold):
    """First step of TWO consecutive checkpoints at/above threshold.

    Durable-competence variant (verification: first-crossing competence is
    transient under the eps floor); None if never sustained.
    """
    for i in range(len(vals) - 1):
        if vals[i] >= threshold and vals[i + 1] >= threshold:
            return steps[i]
    return None


def run_one(seed, cond, budget, eval_every, n_eval, trace_len=TRACE_LEN,
            variant="default"):
    """Train one seed of one condition for exactly `budget` env steps.

    `variant` selects the physics-robustness laws (VARIANTS); rng streams do
    not depend on it, so the default variant reproduces v1 runs exactly and
    adult-only conditions are bit-identical across variants (the laws
    coincide at s=1.0).
    """
    tau_exp, damp_exp = VARIANTS[variant]
    ci = CONDITIONS.index(cond)
    agent_rng = np.random.default_rng([ci, seed, 0])
    env_rng = np.random.default_rng([ci, seed, 1])
    ws = walk_start(cond, budget)
    adaptive = cond in ADAPTIVE
    s_adapt = 0.5
    size0 = s_adapt if adaptive else size_schedule(cond, 0, budget)
    env = BalanceBot(s=size0, mode="walk" if ws == 0 else "balance",
                     rng=env_rng, tau_exp=tau_exp, damp_exp=damp_exp)
    agent = QLearner(env.n_states, env.n_actions, lr=LR, gamma=GAMMA,
                     eps=functools.partial(_eps, budget=budget),
                     rng=agent_rng)

    s = env.reset()
    cum_damage, falls = 0.0, []
    eval_steps, eval_returns, dmg_ck, size_ck = [], [], [], []
    eval_ontgt, eval_lean = [], []
    dense_every = max(1, budget // 300)
    dense_steps, dense_s = [], []
    window = deque(maxlen=GROW_WINDOW)
    for t in range(1, budget + 1):
        env.set_size(s_adapt if adaptive else
                     size_schedule(cond, t - 1, budget))
        env.set_mode("walk" if t - 1 >= ws else "balance")
        a = agent.act(s)
        s2, r, done, info = env.step(a)
        agent.update(s, a, r, s2, done=info["fall"])  # bootstrap through cap
        if info["fall"]:
            cum_damage += info["damage"]
            falls.append([t, float(info["s"]), float(info["damage"])])
        if done:
            window.append(info["fall"])
            if adaptive and len(window) == GROW_WINDOW and s_adapt < 1.0 \
                    and 1.0 - np.mean(window) > GROW_RATE:
                s_adapt = min(1.0, s_adapt + GROW_STEP)
                window.clear()
            s = env.reset()
        else:
            s = s2
        if t % dense_every == 0:
            dense_steps.append(t)
            dense_s.append(env.s)
        if t % eval_every == 0:
            ev = evaluate(agent.Q, seed, t // eval_every, n_eval)
            eval_steps.append(t)
            eval_returns.append(ev["ret"])
            eval_ontgt.append(ev["ontarget"])
            eval_lean.append(ev["lean_ontarget"])
            dmg_ck.append(cum_damage)
            size_ck.append(env.s)

    stc = time_to_threshold(eval_steps, eval_returns, THRESHOLD)
    censored = stc is None
    ck = len(eval_steps) - 1 if censored else eval_steps.index(stc)
    return {
        "seed": seed, "cond": cond, "train_steps": budget, "variant": variant,
        "eval_steps": eval_steps, "eval_returns": eval_returns,
        "eval_ontarget": eval_ontgt, "eval_lean_ontarget": eval_lean,
        "damage_at_checkpoint": dmg_ck, "size_at_checkpoint": size_ck,
        "steps_to_competence": stc, "censored": censored,
        "steps_to_durable": time_to_durable(eval_steps, eval_returns,
                                            THRESHOLD),
        "damage_at_competence": float(cum_damage if censored else dmg_ck[ck]),
        "ontarget_at_competence": eval_ontgt[ck],
        "lean_ontarget_at_competence": eval_lean[ck],
        "final_perf": eval_returns[-1], "total_damage": float(cum_damage),
        "final_ontarget": eval_ontgt[-1], "final_lean_ontarget": eval_lean[-1],
        "n_falls": len(falls), "falls": falls,
        "size_dense": {"steps": dense_steps, "s": dense_s},
        "trace": greedy_trace(agent.Q, seed, trace_len),
    }


def _run_indexed(i, cond, budget, eval_every, n_eval, seed_offset=0,
                 variant="default"):
    """run_seeds adapter: worker index i -> seed seed_offset + i.

    Confirmatory protocol: seeds 100..(100+N-1) via --seed-offset, disjoint
    from the v1 tuning seeds (0-9) and every seed used to probe v1.
    """
    return run_one(seed_offset + i, cond=cond, budget=budget,
                   eval_every=eval_every, n_eval=n_eval, variant=variant)


# ------------------------------------------------------------------ statistics

def holm(pvals):
    """Holm-Bonferroni adjusted p-values (step-down, capped at 1)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def welch(a, b):
    """Welch's t (scipy ttest_ind, equal_var=False), degenerate-input guarded:
    fully tied -> p=1; two distinct constants -> p=0."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.var(a) == 0.0 and np.var(b) == 0.0:
        same = float(np.mean(a)) == float(np.mean(b))
        return {"t": 0.0 if same else float("inf"),
                "p": 1.0 if same else 0.0}
    t, p = ttest_ind(a, b, equal_var=False)
    if not np.isfinite(p):
        return {"t": float(t), "p": 1.0}
    return {"t": float(t), "p": float(p)}


def _mw(a, b):
    """Mann-Whitney with the all-tied guard (p -> 1)."""
    try:
        mw = mann_whitney(a, b)
    except ValueError:  # degenerate input (legacy scipy raises on all-tied)
        return {"u": float("nan"), "p": 1.0}
    if not np.isfinite(mw["p"]):
        mw["p"] = 1.0
    return mw


def parity_verdict(diff_iqm, ci_hi, margin, significant_slower):
    """Equivalence-style verdict for a no-slower claim — never accepts a null.

    diff_iqm = IQM(growth) - IQM(adult) steps (positive = slower; reported in
    evidence); ci_hi = upper 95% bootstrap bound on that difference; margin =
    pre-registered slowdown margin (MARGIN_FRAC x adult IQM);
    significant_slower = a slowdown was detected at the decision level.

    - detected slowdown, CI upper bound beyond the margin  -> refuted
    - detected slowdown, but bounded within the margin     -> boundary
    - no detected slowdown, CI bound within the margin     -> supported
      (equivalence shown, not a null accepted)
    - no detected slowdown, CI cannot exclude a slowdown
      beyond the margin                                    -> null
    """
    if significant_slower:
        return "refuted" if ci_hi > margin else "boundary"
    return "supported" if ci_hi <= margin else "null"


def directional_verdict(sig_mw, sig_welch, direction_ok):
    """Verdict for a directional claim from both test statistics.

    Both significant -> supported/refuted by direction; exactly one
    significant -> boundary (the tests disagree; Colas et al. 2019 show MW
    miscalibrates under unequal shapes); neither -> null.
    """
    if sig_mw and sig_welch:
        return "supported" if direction_ok else "refuted"
    if sig_mw or sig_welch:
        return "boundary"
    return "null"


def combine_verdicts(verdicts):
    """Combine per-comparison verdicts into one claim verdict."""
    s = set(verdicts)
    if "supported" in s and "refuted" in s:
        return "boundary"
    if "refuted" in s:
        return "refuted"
    if "boundary" in s:
        return "boundary"
    if s == {"supported"}:
        return "supported"
    if "supported" in s:  # supported mixed with null
        return "boundary"
    return "null"


# ------------------------------------------------------------- test families

# PRIMARY (v2): the confound-free walk-direct pairing — growth vs adult with
# the task staging matched (both walk from the start), isolating morphology.
# Each entry predicts metric(a) < metric(b); steps entries are no-slower
# checks scored by parity_verdict. Holm within this family (m=4).
PRIMARY = (
    ("damage: grow-linear-walk < adult-walk", "damage_at_competence",
     "grow-linear-walk", "adult-walk"),
    ("damage: grow-adaptive-walk < adult-walk", "damage_at_competence",
     "grow-adaptive-walk", "adult-walk"),
    ("steps: grow-linear-walk <= adult-walk", "steps_to_competence",
     "grow-linear-walk", "adult-walk"),
    ("steps: grow-adaptive-walk <= adult-walk", "steps_to_competence",
     "grow-adaptive-walk", "adult-walk"),
)

# SECONDARY: the v1 family, retained unchanged (its growth arms carry the
# independently-scored balance-first staging factor). Holm within (m=8).
SECONDARY = (
    ("damage: grow-linear < adult-walk", "damage_at_competence",
     "grow-linear", "adult-walk"),
    ("damage: grow-adaptive < adult-walk", "damage_at_competence",
     "grow-adaptive", "adult-walk"),
    ("damage: grow-linear < adult-balance-first", "damage_at_competence",
     "grow-linear", "adult-balance-first"),
    ("damage: grow-linear < grow-jump (gradualism)", "damage_at_competence",
     "grow-linear", "grow-jump"),
    ("steps: grow-linear <= adult-walk", "steps_to_competence",
     "grow-linear", "adult-walk"),
    ("steps: grow-adaptive <= adult-walk", "steps_to_competence",
     "grow-adaptive", "adult-walk"),
    ("steps: adult-balance-first < adult-walk (balance-first)",
     "steps_to_competence", "adult-balance-first", "adult-walk"),
    ("steps: grow-linear < grow-linear-walk (balance-first)",
     "steps_to_competence", "grow-linear", "grow-linear-walk"),
)


def _metric(results, cond, metric, budget):
    if metric == "steps_to_competence":
        # protocol: censored seeds get budget+1 (conservative for rank tests)
        return [budget + 1 if r["steps_to_competence"] is None
                else r["steps_to_competence"] for r in results[cond]]
    return [r[metric] for r in results[cond]]


def _diff_ci(va, vb, n_boot, rng):
    """95% percentile bootstrap CI on IQM(a) - IQM(b) (independent resamples)."""
    va = np.asarray(va, dtype=float)
    vb = np.asarray(vb, dtype=float)
    diffs = np.array([iqm(rng.choice(va, size=len(va), replace=True))
                      - iqm(rng.choice(vb, size=len(vb), replace=True))
                      for _ in range(n_boot)])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _censoring_sensitivity(results, a, b):
    """Damage-test robustness to the censored->budget-end-damage policy.

    drop_censored_p: MW p with censored seeds removed from both sides.
    worst_case_p: MW p with a's censored seeds ranked worst (+inf damage) and
    b's censored seeds ranked best (below every real value) — maximally
    against the prediction a < b.
    """
    da = [(r["damage_at_competence"], r["censored"]) for r in results[a]]
    db = [(r["damage_at_competence"], r["censored"]) for r in results[b]]
    kept_a = [v for v, c in da if not c]
    kept_b = [v for v, c in db if not c]
    if len(kept_a) >= 2 and len(kept_b) >= 2:
        drop_p = _mw(kept_a, kept_b)["p"]
    else:
        drop_p = 1.0  # not estimable after dropping
    worst_a = [float("inf") if c else v for v, c in da]
    worst_b = [-1.0 if c else v for v, c in db]
    return {"drop_censored_p": drop_p,
            "worst_case_p": _mw(worst_a, worst_b)["p"],
            "n_censored_a": int(sum(c for _, c in da)),
            "n_censored_b": int(sum(c for _, c in db))}


def _run_tests(results, family, entries, budget, n_boot, rng):
    """Build test entries with MW + Welch, Holm within the family, plus the
    steps-difference equivalence CI or the damage censoring sensitivity."""
    tests = []
    for name, metric, a, b in entries:
        va = _metric(results, a, metric, budget)
        vb = _metric(results, b, metric, budget)
        mw = _mw(va, vb)
        w = welch(va, vb)
        t = {"name": name, "family": family, "metric": metric, "a": a, "b": b,
             "iqm_a": iqm(va), "iqm_b": iqm(vb),
             "u": mw["u"], "p": mw["p"],
             "t_welch": w["t"], "p_welch": w["p"],
             "direction_ok": bool(iqm(va) < iqm(vb))}
        if metric == "steps_to_competence":
            lo, hi = _diff_ci(va, vb, n_boot, rng)
            t["steps_diff"] = {"iqm_diff": iqm(va) - iqm(vb),
                               "lo": lo, "hi": hi,
                               "margin": MARGIN_FRAC * iqm(vb),
                               "margin_frac": MARGIN_FRAC}
        else:
            t["censoring_sensitivity"] = _censoring_sensitivity(results, a, b)
        tests.append(t)
    for t, ph in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = ph
        t["significant"] = bool(ph < 0.05)
    for t, ph in zip(tests, holm([t["p_welch"] for t in tests])):
        t["p_welch_holm"] = ph
        t["significant_welch"] = bool(ph < 0.05)
    return tests


# ------------------------------------------------------------------- verdicts

def _steps_parity(t):
    """Per-pairing steps-parity verdict. A slowdown detected by EITHER test
    (MW or Welch, Holm-adjusted) blocks parity — the stricter reading."""
    sd = t["steps_diff"]
    slower_detected = ((t["significant"] or t["significant_welch"])
                       and not t["direction_ok"])
    return parity_verdict(sd["iqm_diff"], sd["hi"], sd["margin"],
                          slower_detected)


def _dirw(t):
    return directional_verdict(t["significant"], t["significant_welch"],
                               t["direction_ok"])


def _pp(t):
    return (f"MW Holm p={t['p_holm']:.2g}, Welch Holm p={t['p_welch_holm']:.2g}")


def _claims(tests, conditions):
    by = {t["name"]: t for t in tests}
    dmg_lw = by["damage: grow-linear-walk < adult-walk"]
    dmg_aw = by["damage: grow-adaptive-walk < adult-walk"]
    stp_lw = by["steps: grow-linear-walk <= adult-walk"]
    stp_aw = by["steps: grow-adaptive-walk <= adult-walk"]
    grad = by["damage: grow-linear < grow-jump (gradualism)"]
    bf_adult = by["steps: adult-balance-first < adult-walk (balance-first)"]
    bf_grow = by["steps: grow-linear < grow-linear-walk (balance-first)"]

    def ontgt(cond):
        return iqm(conditions[cond]["lean_ontarget_at_competence"])

    claims = []

    # 1. damage at competence (headline, walk-direct pairing)
    v = combine_verdicts([_dirw(dmg_lw), _dirw(dmg_aw)])
    cs_l, cs_a = dmg_lw["censoring_sensitivity"], dmg_aw["censoring_sensitivity"]
    claims.append({
        "key": "damage_at_competence",
        "claim": ("Growth curriculum (walk-direct) reaches adult competence "
                  "with far less cumulative damage than adult training "
                  "(grow-linear-walk and grow-adaptive-walk vs adult-walk; "
                  "confound-free morphology contrast, task staging matched)."),
        "verdict": v,
        "evidence": (
            f"Damage IQM: grow-linear-walk {dmg_lw['iqm_a']:.1f} vs adult-walk "
            f"{dmg_lw['iqm_b']:.1f} ({_pp(dmg_lw)}); grow-adaptive-walk "
            f"{dmg_aw['iqm_a']:.1f} vs {dmg_aw['iqm_b']:.1f} ({_pp(dmg_aw)}). "
            f"Censoring sensitivity (drop / worst-case MW p): grow-linear-walk "
            f"{cs_l['drop_censored_p']:.2g} / {cs_l['worst_case_p']:.2g}; "
            f"grow-adaptive-walk {cs_a['drop_censored_p']:.2g} / "
            f"{cs_a['worst_case_p']:.2g}. Standing-only guard: lean-target "
            f"on-target IQM at competence grow-linear-walk "
            f"{ontgt('grow-linear-walk'):.2f}, grow-adaptive-walk "
            f"{ontgt('grow-adaptive-walk'):.2f}, adult-walk "
            f"{ontgt('adult-walk'):.2f} (nonzero = tracking, not standing). "
            f"Balance-first-staged growth arms agree (secondary family): "
            f"grow-linear vs adult-walk damage "
            f"MW Holm p={by['damage: grow-linear < adult-walk']['p_holm']:.2g}, "
            f"grow-adaptive "
            f"p={by['damage: grow-adaptive < adult-walk']['p_holm']:.2g}."),
    })

    # 2. steps parity (equivalence view; never accept the null)
    pv_l, pv_a = _steps_parity(stp_lw), _steps_parity(stp_aw)
    v = combine_verdicts([pv_l, pv_a])

    def sd_str(t):
        sd = t["steps_diff"]
        return (f"diff IQM {sd['iqm_diff']:+.0f} steps, 95% CI "
                f"[{sd['lo']:.0f}, {sd['hi']:.0f}], margin +{sd['margin']:.0f} "
                f"({sd['margin_frac']:.0%} of adult IQM)")

    claims.append({
        "key": "steps_parity",
        "claim": ("Growth curriculum (walk-direct) reaches adult competence "
                  "no slower than adult training, within a pre-registered "
                  f"+{MARGIN_FRAC:.0%} equivalence margin (never scored by "
                  "accepting a null)."),
        "verdict": v,
        "evidence": (
            f"Steps to competence (censored=budget+1): grow-linear-walk "
            f"{stp_lw['iqm_a']:.0f} vs adult-walk {stp_lw['iqm_b']:.0f} "
            f"({sd_str(stp_lw)}; {_pp(stp_lw)}) -> {pv_l}; grow-adaptive-walk "
            f"{stp_aw['iqm_a']:.0f} vs {stp_aw['iqm_b']:.0f} "
            f"({sd_str(stp_aw)}; {_pp(stp_aw)}) -> {pv_a}. "
            f"Balance-first-staged growth arms (secondary, context): "
            f"grow-linear {by['steps: grow-linear <= adult-walk']['iqm_a']:.0f} "
            f"vs adult-walk "
            f"{by['steps: grow-linear <= adult-walk']['iqm_b']:.0f} "
            f"({_pp(by['steps: grow-linear <= adult-walk'])})."),
    })

    # 3. gradualism (grow-linear < grow-jump on damage)
    v = _dirw(grad)
    direction = ("predicted direction (gradual cheaper)" if grad["direction_ok"]
                 else "OPPOSITE direction — abrupt growth was cheaper")
    claims.append({
        "key": "gradualism",
        "claim": ("Gradual growth beats abrupt growth on damage "
                  "(grow-linear < grow-jump, matched balance-first staging)."),
        "verdict": v,
        "evidence": (
            f"Damage IQM: grow-linear {grad['iqm_a']:.1f} vs grow-jump "
            f"{grad['iqm_b']:.1f}; point estimate in the {direction}; "
            f"{_pp(grad)} (secondary family, m=8)."),
    })

    # 4. balance-first (matched-morphology staging comparisons)
    va, vg = _dirw(bf_adult), _dirw(bf_grow)
    v = combine_verdicts([va, vg])
    claims.append({
        "key": "balance_first",
        "claim": ("Balance-first task staging beats walking-from-the-start "
                  "within matched morphology (adult-balance-first < adult-walk "
                  "and grow-linear < grow-linear-walk on steps)."),
        "verdict": v,
        "evidence": (
            f"Steps IQM: adult-balance-first {bf_adult['iqm_a']:.0f} vs "
            f"adult-walk {bf_adult['iqm_b']:.0f} ({_pp(bf_adult)}) -> {va}; "
            f"grow-linear {bf_grow['iqm_a']:.0f} vs grow-linear-walk "
            f"{bf_grow['iqm_b']:.0f} ({_pp(bf_grow)}) -> {vg}."
            + (" REVERSED: walking-from-the-start was FASTER on both "
               "matched-morphology comparisons." if v == "refuted" else "")),
    })
    return claims


def _verdict_phrase(claim):
    v = claim["verdict"].upper()
    if claim["key"] == "balance_first" and claim["verdict"] == "refuted":
        return "REFUTED (reversed: walk-from-the-start was faster)"
    if claim["key"] == "gradualism" and claim["verdict"] == "refuted":
        return "REFUTED (reversed: abrupt growth was cheaper)"
    return v


# ---------------------------------------------------------------- aggregation

def _curve_ci(mat, n_boot, rng):
    iqms, los, his = [], [], []
    for col in mat.T:
        iqms.append(iqm(col))
        lo, hi = bootstrap_ci(col, n_boot=n_boot, rng=rng, statistic=iqm)
        los.append(lo)
        his.append(hi)
    return iqms, los, his


def _condition_block(rs):
    censored = [r["censored"] for r in rs]
    durable = [r["steps_to_durable"] for r in rs]
    return {
        "n_seeds": len(rs),
        "seeds": [r["seed"] for r in rs],
        "steps_to_competence": [r["steps_to_competence"] for r in rs],
        "censored": censored,
        "censored_frac": float(np.mean(censored)),
        "steps_to_durable": durable,
        "durable_frac": float(np.mean([d is not None for d in durable])),
        "damage_at_competence": [r["damage_at_competence"] for r in rs],
        "ontarget_at_competence": [r["ontarget_at_competence"] for r in rs],
        "lean_ontarget_at_competence":
            [r["lean_ontarget_at_competence"] for r in rs],
        "final_perf": [r["final_perf"] for r in rs],
        "final_ontarget": [r["final_ontarget"] for r in rs],
        "final_lean_ontarget": [r["final_lean_ontarget"] for r in rs],
        "total_damage": [r["total_damage"] for r in rs],
        "n_falls": [r["n_falls"] for r in rs],
    }


def _robustness_block(variant, var_results, main_results, budget):
    """Reduced-seed physics-robustness analysis (labeled, non-confirmatory).

    adult-walk comparator is reused from the main arm restricted to the
    variant's seeds: the variant laws coincide with the default at s=1.0 and
    rng streams are variant-independent, so an adult-walk rerun under the
    variant would be bit-identical.
    """
    rob_seeds = [r["seed"] for r in var_results[ROBUST_CONDITIONS[0]]]
    adult = [r for r in main_results["adult-walk"] if r["seed"] in rob_seeds]
    results = dict(var_results)
    results["adult-walk"] = adult

    conds = {c: {"damage_at_competence":
                 [r["damage_at_competence"] for r in results[c]],
                 "steps_to_competence":
                 [r["steps_to_competence"] for r in results[c]],
                 "censored_frac":
                 float(np.mean([r["censored"] for r in results[c]])),
                 "total_damage": [r["total_damage"] for r in results[c]]}
             for c in (*ROBUST_CONDITIONS, "adult-walk")}

    tests = []
    for cond in ROBUST_CONDITIONS:
        va = _metric(results, cond, "damage_at_competence", budget)
        vb = _metric(results, "adult-walk", "damage_at_competence", budget)
        mw = _mw(va, vb)
        w = welch(va, vb)
        tests.append({"name": f"damage: {cond} < adult-walk [{variant}]",
                      "metric": "damage_at_competence",
                      "a": cond, "b": "adult-walk",
                      "iqm_a": iqm(va), "iqm_b": iqm(vb),
                      "u": mw["u"], "p": mw["p"],
                      "t_welch": w["t"], "p_welch": w["p"],
                      "direction_ok": bool(iqm(va) < iqm(vb))})
    for t, ph in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = ph
        t["significant"] = bool(ph < 0.05)
    for t, ph in zip(tests, holm([t["p_welch"] for t in tests])):
        t["p_welch_holm"] = ph
        t["significant_welch"] = bool(ph < 0.05)

    ordering = sorted(({"cond": c,
                        "damage_iqm": iqm(conds[c]["damage_at_competence"])}
                       for c in conds), key=lambda d: d["damage_iqm"])
    tau_exp, damp_exp = VARIANTS[variant]
    return {
        "variant": variant,
        "laws": {"tau_max": f"40 * s^{tau_exp}", "damping": f"1.0 * s^{damp_exp}"},
        "n_seeds": len(rob_seeds),
        "seeds": rob_seeds,
        "conditions": conds,
        "tests": tests,
        "damage_ordering": ordering,
        "growth_all_cheaper_than_adult": bool(all(
            iqm(conds[c]["damage_at_competence"])
            < iqm(conds["adult-walk"]["damage_at_competence"])
            for c in ROBUST_CONDITIONS)),
        "note": ("ROBUSTNESS ARM (reduced seed count, not confirmatory): "
                 "training physics variant "
                 f"tau_max=40 s^{tau_exp}, damping b=1.0 s^{damp_exp}. "
                 "adult-walk comparator reused from the main arm restricted "
                 "to these seeds — variant laws coincide with the default at "
                 "s=1.0 and rng streams are variant-independent, so a rerun "
                 "would be bit-identical. Eval unchanged (adult body)."),
    }


def aggregate(results, budget, n_boot, robustness=None):
    """Assemble the shared-contract JSON from per-seed results."""
    rng = np.random.default_rng(0)
    steps_ck = results[CONDITIONS[0]][0]["eval_steps"]
    conditions, curves = {}, {}
    viz_sched, viz_falls, viz_traces, viz_dvc, viz_ontgt = {}, {}, {}, {}, {}
    for cond in CONDITIONS:
        rs = results[cond]
        ret = np.array([r["eval_returns"] for r in rs])
        dmg = np.array([r["damage_at_checkpoint"] for r in rs])
        size = np.array([r["size_at_checkpoint"] for r in rs])
        ontgt = np.array([r["eval_ontarget"] for r in rs])
        lean = np.array([r["eval_lean_ontarget"] for r in rs])
        r_iqm, r_lo, r_hi = _curve_ci(ret, n_boot, rng)
        d_iqm, d_lo, d_hi = _curve_ci(dmg, n_boot, rng)
        curves[cond] = {"steps": steps_ck, "iqm": r_iqm, "lo": r_lo,
                        "hi": r_hi}
        conditions[cond] = _condition_block(rs)
        # representative seed: median final performance
        rep = int(np.argsort([r["final_perf"] for r in rs])[len(rs) // 2])
        viz_traces[cond] = {"seed": rs[rep]["seed"], **rs[rep]["trace"]}
        viz_sched[cond] = {
            "steps": steps_ck,
            "iqm": [iqm(col) for col in size.T],
            "min": size.min(axis=0).tolist(), "max": size.max(axis=0).tolist(),
            "example_seed": rs[rep]["seed"],
            "example_steps": rs[rep]["size_dense"]["steps"],
            "example_s": rs[rep]["size_dense"]["s"],
        }
        events = sorted((f[0], f[1], f[2], r["seed"])
                        for r in rs for f in r["falls"])
        if len(events) > 500:
            keep = np.linspace(0, len(events) - 1, 500).astype(int)
            events = [events[i] for i in keep]
        viz_falls[cond] = [{"step": e[0], "size": e[1], "damage": e[2],
                            "seed": e[3]} for e in events]
        viz_dvc[cond] = {"steps": steps_ck,
                         "damage_iqm": d_iqm, "damage_lo": d_lo,
                         "damage_hi": d_hi,
                         "return_iqm": r_iqm, "return_lo": r_lo,
                         "return_hi": r_hi}
        viz_ontgt[cond] = {"steps": steps_ck,
                           "ontarget_iqm": [iqm(col) for col in ontgt.T],
                           "lean_iqm": [iqm(col) for col in lean.T]}

    tests = (_run_tests(results, "primary", PRIMARY, budget, n_boot, rng)
             + _run_tests(results, "secondary", SECONDARY, budget, n_boot,
                          rng))

    rob_blocks = {}
    for variant, var_results in (robustness or {}).items():
        rob_blocks[variant] = _robustness_block(variant, var_results,
                                                results, budget)

    claims = _claims(tests, conditions)
    cens = ", ".join(f"{c} {conditions[c]['censored_frac']:.2f}"
                     for c in CONDITIONS)
    rob_line = "; ".join(
        f"{v}: growth arms all cheaper than adult-walk = "
        f"{blk['growth_all_cheaper_than_adult']}"
        for v, blk in rob_blocks.items()) or "not run"
    summary = (
        "EXP5 v2 per-claim verdicts (primary = walk-direct pairing, "
        "morphology isolated): "
        + " | ".join(f"{c['key']}: {_verdict_phrase(c)}" for c in claims)
        + f". Censored fractions: {cens}. Physics robustness ({rob_line})."
        + " Both Mann-Whitney and Welch reported, Holm within family;"
        " steps parity scored by equivalence CI against a "
        f"+{MARGIN_FRAC:.0%} margin, never by accepting a null.")

    out = {
        "experiment": "exp5_growing",
        "hypothesis": HYPOTHESIS,
        "config": {
            "budget": budget, "eval_every": steps_ck[1] - steps_ck[0]
            if len(steps_ck) > 1 else steps_ck[0],
            "n_eval_episodes": N_EVAL, "threshold": THRESHOLD,
            "cap": CAP, "n_boot": n_boot,
            "agent": {"algo": "tabular Q-learning", "lr": LR, "gamma": GAMMA,
                      "eps": f"max({EPS_FLOOR}, {EPS0} * 2^(-age / "
                             f"({EPS_HALFLIFE_FRAC} * budget)))"},
            "grow_adaptive": {"window": GROW_WINDOW, "no_fall_rate": GROW_RATE,
                              "step": GROW_STEP},
            "equivalence_margin_frac": MARGIN_FRAC,
            "stats": ("two-sided Mann-Whitney AND Welch t "
                      "(scipy.stats.ttest_ind equal_var=False), each "
                      "Holm-adjusted within its family (primary m=4, "
                      "secondary m=8; robustness m=4 per variant)"),
            "tuning": ("hyperparameters tuned once on v1 seeds 0-9; "
                       "confirmatory seeds are offset to stay disjoint"),
            "variants": {v: {"tau_exp": te, "damp_exp": de}
                         for v, (te, de) in VARIANTS.items()},
        },
        "conditions": conditions,
        "curves": curves,
        "tests": tests,
        "robustness": rob_blocks,
        "conclusion": {"claims": claims, "summary": summary},
        "viz": {
            "size_schedules": viz_sched,
            "fall_events": viz_falls,
            "theta_traces": viz_traces,
            "damage_vs_competence": viz_dvc,
            "ontarget": viz_ontgt,
            "robustness": {v: {"n_seeds": blk["n_seeds"],
                               "laws": blk["laws"],
                               "damage_ordering": blk["damage_ordering"],
                               "growth_all_cheaper_than_adult":
                                   blk["growth_all_cheaper_than_adult"]}
                           for v, blk in rob_blocks.items()},
            "meta": {"threshold": THRESHOLD, "fall_angle": float(FALL_ANGLE),
                     "targets": list(TARGETS), "cap": CAP,
                     "max_return_per_step": 3.0,
                     "conditions": dict(DESCRIPTIONS),
                     "damage_law": "(s / 1.0)^4 per fall",
                     "reward": ("+1/step upright, +2/step within 0.05 rad of "
                                "target, -5 on a fall (fixed across sizes; "
                                "physical damage recorded separately)"),
                     "damping": ("default b = 1.0 constant across sizes -> "
                                 "relative damping b/(m l^2) = 1/(15 s^5), "
                                 "~32x stronger for s=0.5 than adult (v1 "
                                 "deviation, disclosed); damp2 variant "
                                 "scales b = 1.0 s^2"),
                     "eval": "adult body s=1.0, greedy, walking targets"},
        },
    }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=40)
    p.add_argument("--seed-offset", type=int, default=100,
                   help="confirmatory seeds are seed_offset..seed_offset+N-1, "
                        "disjoint from v1 tuning seeds 0-9")
    p.add_argument("--robust-seeds", type=int, default=20,
                   help="seeds per condition for the physics-robustness arms")
    p.add_argument("--out", default="results/exp5.json")
    p.add_argument("--smoke", action="store_true",
                   help="2 seeds, 10x reduced budget, fewer bootstrap draws")
    p.add_argument("--jobs", type=int, default=20)
    args = p.parse_args(argv)

    seeds = 2 if args.smoke else args.seeds
    rob_seeds = 2 if args.smoke else min(args.robust_seeds, seeds)
    budget = SMOKE_BUDGET if args.smoke else BUDGET
    eval_every = budget // 30
    n_boot = 1000 if args.smoke else 10000

    t0 = time.time()
    results = {}
    for cond in CONDITIONS:
        t1 = time.time()
        fn = functools.partial(_run_indexed, cond=cond, budget=budget,
                               eval_every=eval_every, n_eval=N_EVAL,
                               seed_offset=args.seed_offset)
        results[cond] = run_seeds(fn, seeds, n_jobs=args.jobs)
        print(f"{cond:20s} {seeds} seeds in {time.time() - t1:5.1f}s")
    robustness = {}
    for variant in ROBUST_VARIANTS:
        robustness[variant] = {}
        for cond in ROBUST_CONDITIONS:
            t1 = time.time()
            fn = functools.partial(_run_indexed, cond=cond, budget=budget,
                                   eval_every=eval_every, n_eval=N_EVAL,
                                   seed_offset=args.seed_offset,
                                   variant=variant)
            robustness[variant][cond] = run_seeds(fn, rob_seeds,
                                                  n_jobs=args.jobs)
            print(f"[{variant}] {cond:20s} {rob_seeds} seeds "
                  f"in {time.time() - t1:5.1f}s")
    out = aggregate(results, budget=budget, n_boot=n_boot,
                    robustness=robustness)
    out["config"]["seeds"] = seeds
    out["config"]["seed_offset"] = args.seed_offset
    out["config"]["seed_range"] = [args.seed_offset,
                                   args.seed_offset + seeds - 1]
    out["config"]["robust_seeds"] = rob_seeds
    out["config"]["smoke"] = args.smoke
    save_json(args.out, out)
    print(f"wrote {args.out} in {time.time() - t0:.1f}s total")
    print(out["conclusion"]["summary"])
    for c in out["conclusion"]["claims"]:
        print(f"  [{c['verdict']:9s}] {c['key']}: {c['evidence']}")


if __name__ == "__main__":
    main()
