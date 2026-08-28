"""EXP5 — Growing Bodies (H5: morphological curriculum).

Claim: training a small body first (falls are cheap, relative strength is
high — square-cube law) and growing toward the adult body reaches adult
competence with far less cumulative damage; balance-first should beat
walking-from-the-start. Six conditions on `BalanceBot`, one shared tabular
Q-learner config, budgets matched exactly at `BUDGET` training steps.
Damage = sum of (s/1)^4 over training falls, recorded separately from reward
(the reward landscape is identical at every size — only the body wears
differently). Eval is ALWAYS the adult body (s=1.0), greedy, walking
targets, every EVAL_EVERY steps, identical across conditions and excluded
from the training budget. Competence = eval return >= 500 (survive the
400-step cap while tracking targets a decent fraction of the time).

    python experiments/exp5_growing.py --seeds 20 --out results/exp5.json
    python experiments/exp5_growing.py --smoke --out results/exp5_smoke.json
"""

import argparse
import functools
import time
from collections import deque

import numpy as np

from devrl.agents.qlearning import QLearner
from devrl.envs.balance import BalanceBot, CAP, FALL_ANGLE, TARGETS
from devrl.run import run_seeds, save_json
from devrl.stats import bootstrap_ci, iqm, mann_whitney, time_to_threshold

CONDITIONS = ("adult-walk", "adult-balance-first", "grow-linear",
              "grow-adaptive", "grow-jump", "grow-linear-walk")
DESCRIPTIONS = {
    "adult-walk": "s=1.0 throughout, walking task from the start",
    "adult-balance-first": "s=1.0; target fixed at 0 for first 30%, then walking",
    "grow-linear": "s 0.5->1.0 over first 60% of budget; balance-first",
    "grow-adaptive": "grow s by 0.05 when rolling no-fall rate > 70%; balance-first",
    "grow-jump": "s=0.5 for 60% of budget then s=1.0; balance-first",
    "grow-linear-walk": "grow-linear schedule but walking task from the start",
}
HYPOTHESIS = ("H5: a morphological curriculum — train a small body first and "
              "grow toward the adult — reaches adult competence with far less "
              "cumulative damage (and no slower), and balance-first beats "
              "walking-from-the-start.")

BUDGET = 120000
SMOKE_BUDGET = 12000
THRESHOLD = 500.0
N_EVAL = 5
TRACE_LEN = 200
# One agent config shared by every condition (tuned once, applied uniformly):
LR = 0.25
GAMMA = 0.99
EPS0, EPS_FLOOR, EPS_HALFLIFE_FRAC = 0.5, 0.08, 0.25
# grow-adaptive: +0.05 size whenever the last GROW_WINDOW episodes had a
# no-fall rate above GROW_RATE ("grow when ready"); window resets on growth.
GROW_WINDOW, GROW_RATE, GROW_STEP = 20, 0.7, 0.05


def _eps(age, budget):
    return max(EPS_FLOOR, EPS0 * 2 ** (-age / (EPS_HALFLIFE_FRAC * budget)))


def size_schedule(cond, t, budget):
    """Body size at training step t for the schedule-driven conditions.

    grow-adaptive is event-driven (see run_one), not scheduled.
    """
    if cond == "grow-adaptive":
        raise ValueError("grow-adaptive size is event-driven, not scheduled")
    if cond in ("grow-linear", "grow-linear-walk"):
        return 0.5 + 0.5 * min(t / (0.6 * budget), 1.0)
    if cond == "grow-jump":
        return 0.5 if t < 0.6 * budget else 1.0
    return 1.0  # adult-walk, adult-balance-first


def walk_start(cond, budget):
    """Training step at which the walking task begins (0 = from the start)."""
    return 0 if cond in ("adult-walk", "grow-linear-walk") else int(0.3 * budget)


def evaluate(Q, seed, k, n_eval):
    """Mean greedy return on the ADULT body with walking targets.

    Seeded by (seed, checkpoint) only — never by condition — so every
    condition faces the identical eval protocol; runs in a separate env and
    consumes no training budget. Greedy = argmax (no agent rng touched).
    """
    env = BalanceBot(s=1.0, mode="walk",
                     rng=np.random.default_rng([777, seed, k]))
    rets = []
    for _ in range(n_eval):
        s, ret, done = env.reset(), 0.0, False
        while not done:
            s, r, done, _ = env.step(int(np.argmax(Q[s])))
            ret += r
        rets.append(ret)
    return float(np.mean(rets))


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


def run_one(seed, cond, budget, eval_every, n_eval, trace_len=TRACE_LEN):
    """Train one seed of one condition for exactly `budget` env steps."""
    ci = CONDITIONS.index(cond)
    agent_rng = np.random.default_rng([ci, seed, 0])
    env_rng = np.random.default_rng([ci, seed, 1])
    ws = walk_start(cond, budget)
    adaptive = cond == "grow-adaptive"
    s_adapt = 0.5
    size0 = s_adapt if adaptive else size_schedule(cond, 0, budget)
    env = BalanceBot(s=size0, mode="walk" if ws == 0 else "balance",
                     rng=env_rng)
    agent = QLearner(env.n_states, env.n_actions, lr=LR, gamma=GAMMA,
                     eps=functools.partial(_eps, budget=budget),
                     rng=agent_rng)

    s = env.reset()
    cum_damage, falls = 0.0, []
    eval_steps, eval_returns, dmg_ck, size_ck = [], [], [], []
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
            eval_steps.append(t)
            eval_returns.append(evaluate(agent.Q, seed, t // eval_every,
                                         n_eval))
            dmg_ck.append(cum_damage)
            size_ck.append(env.s)

    stc = time_to_threshold(eval_steps, eval_returns, THRESHOLD)
    censored = stc is None
    dmg_at = cum_damage if censored else dmg_ck[eval_steps.index(stc)]
    return {
        "seed": seed, "cond": cond, "train_steps": budget,
        "eval_steps": eval_steps, "eval_returns": eval_returns,
        "damage_at_checkpoint": dmg_ck, "size_at_checkpoint": size_ck,
        "steps_to_competence": stc, "censored": censored,
        "damage_at_competence": float(dmg_at),
        "final_perf": eval_returns[-1], "total_damage": float(cum_damage),
        "n_falls": len(falls), "falls": falls,
        "size_dense": {"steps": dense_steps, "s": dense_s},
        "trace": greedy_trace(agent.Q, seed, trace_len),
    }


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


# Primary family (Holm-Bonferroni corrected together). Each entry predicts
# metric(a) < metric(b); the two "<=" steps entries are no-slower checks.
PRIMARY = (
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


def _curve_ci(mat, n_boot, rng):
    iqms, los, his = [], [], []
    for col in mat.T:
        iqms.append(iqm(col))
        lo, hi = bootstrap_ci(col, n_boot=n_boot, rng=rng, statistic=iqm)
        los.append(lo)
        his.append(hi)
    return iqms, los, his


def aggregate(results, budget, n_boot):
    """Assemble the shared-contract JSON from per-seed results."""
    rng = np.random.default_rng(0)
    steps_ck = results[CONDITIONS[0]][0]["eval_steps"]
    conditions, curves = {}, {}
    viz_sched, viz_falls, viz_traces, viz_dvc = {}, {}, {}, {}
    for cond in CONDITIONS:
        rs = results[cond]
        ret = np.array([r["eval_returns"] for r in rs])
        dmg = np.array([r["damage_at_checkpoint"] for r in rs])
        size = np.array([r["size_at_checkpoint"] for r in rs])
        r_iqm, r_lo, r_hi = _curve_ci(ret, n_boot, rng)
        d_iqm, d_lo, d_hi = _curve_ci(dmg, n_boot, rng)
        curves[cond] = {"steps": steps_ck, "iqm": r_iqm, "lo": r_lo,
                        "hi": r_hi}
        censored = [r["censored"] for r in rs]
        conditions[cond] = {
            "n_seeds": len(rs),
            "seeds": [r["seed"] for r in rs],
            "steps_to_competence": [r["steps_to_competence"] for r in rs],
            "censored": censored,
            "censored_frac": float(np.mean(censored)),
            "damage_at_competence": [r["damage_at_competence"] for r in rs],
            "final_perf": [r["final_perf"] for r in rs],
            "total_damage": [r["total_damage"] for r in rs],
            "n_falls": [r["n_falls"] for r in rs],
        }
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

    pvals, tests = [], []
    for name, metric, a, b in PRIMARY:
        va = _metric(results, a, metric, budget)
        vb = _metric(results, b, metric, budget)
        mw = mann_whitney(va, vb)
        if not np.isfinite(mw["p"]):  # all-tied inputs (e.g. all censored)
            mw["p"] = 1.0
        pvals.append(mw["p"])
        tests.append({"name": name, "metric": metric, "a": a, "b": b,
                      "iqm_a": iqm(va), "iqm_b": iqm(vb),
                      "u": mw["u"], "p": mw["p"],
                      "direction_ok": bool(iqm(va) < iqm(vb))})
    for t, ph in zip(tests, holm(pvals)):
        t["p_holm"] = ph
        t["significant"] = bool(ph < 0.05)

    byname = {t["name"]: t for t in tests}
    dmg_lin = byname["damage: grow-linear < adult-walk"]
    dmg_ada = byname["damage: grow-adaptive < adult-walk"]
    stp_lin = byname["steps: grow-linear <= adult-walk"]
    stp_ada = byname["steps: grow-adaptive <= adult-walk"]
    bf_adult = byname["steps: adult-balance-first < adult-walk (balance-first)"]
    bf_grow = byname["steps: grow-linear < grow-linear-walk (balance-first)"]
    damage_ok = all(t["significant"] and t["direction_ok"]
                    for t in (dmg_lin, dmg_ada))
    no_slower = all(not (t["significant"] and not t["direction_ok"])
                    for t in (stp_lin, stp_ada))
    supported = bool(damage_ok and no_slower)
    bf_wins = [t["significant"] and t["direction_ok"]
               for t in (bf_adult, bf_grow)]
    bf_loses = [t["significant"] and not t["direction_ok"]
                for t in (bf_adult, bf_grow)]
    summary = (
        f"Morphological curriculum: {'SUPPORTED' if supported else 'NOT SUPPORTED'}. "
        f"Damage at adult competence (IQM): grow-linear {dmg_lin['iqm_a']:.0f} "
        f"vs adult-walk {dmg_lin['iqm_b']:.0f} (Holm p={dmg_lin['p_holm']:.2g}); "
        f"grow-adaptive {dmg_ada['iqm_a']:.0f} (Holm p={dmg_ada['p_holm']:.2g}). "
        f"Steps to competence (IQM, censored=budget+1): grow-linear "
        f"{stp_lin['iqm_a']:.0f} vs adult-walk {stp_lin['iqm_b']:.0f} "
        f"(Holm p={stp_lin['p_holm']:.2g}). Gradualism (grow-linear < "
        f"grow-jump damage): "
        f"{'significant' if byname['damage: grow-linear < grow-jump (gradualism)']['significant'] else 'not significant'}. "
        f"Balance-first sub-claim: "
        f"{'supported' if all(bf_wins) else ('REFUTED (walking-from-the-start was faster)' if any(bf_loses) else 'not significant')} "
        f"on both matched-morphology comparisons. Censored fractions: "
        + ", ".join(f"{c} {conditions[c]['censored_frac']:.2f}"
                    for c in CONDITIONS) + ".")

    return {
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
        },
        "conditions": conditions,
        "curves": curves,
        "tests": tests,
        "conclusion": {"supported": supported, "summary": summary},
        "viz": {
            "size_schedules": viz_sched,
            "fall_events": viz_falls,
            "theta_traces": viz_traces,
            "damage_vs_competence": viz_dvc,
            "meta": {"threshold": THRESHOLD, "fall_angle": float(FALL_ANGLE),
                     "targets": list(TARGETS), "cap": CAP,
                     "max_return_per_step": 3.0,
                     "conditions": dict(DESCRIPTIONS),
                     "damage_law": "(s / 1.0)^4 per fall",
                     "eval": "adult body s=1.0, greedy, walking targets"},
        },
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--out", default="results/exp5.json")
    p.add_argument("--smoke", action="store_true",
                   help="2 seeds, 10x reduced budget, fewer bootstrap draws")
    p.add_argument("--jobs", type=int, default=20)
    args = p.parse_args(argv)

    seeds = 2 if args.smoke else args.seeds
    budget = SMOKE_BUDGET if args.smoke else BUDGET
    eval_every = budget // 30
    n_boot = 1000 if args.smoke else 10000

    t0 = time.time()
    results = {}
    for cond in CONDITIONS:
        t1 = time.time()
        fn = functools.partial(run_one, cond=cond, budget=budget,
                               eval_every=eval_every, n_eval=N_EVAL)
        results[cond] = run_seeds(fn, seeds, n_jobs=args.jobs)
        print(f"{cond:20s} {seeds} seeds in {time.time() - t1:5.1f}s")
    out = aggregate(results, budget=budget, n_boot=n_boot)
    out["config"]["seeds"] = seeds
    out["config"]["smoke"] = args.smoke
    save_json(args.out, out)
    print(f"wrote {args.out} in {time.time() - t0:.1f}s total")
    print(out["conclusion"]["summary"])


if __name__ == "__main__":
    main()
