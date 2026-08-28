"""EXP1 — The Blindfold Test (H1: world models).

Train Q-learning vs DynaQ(planning=20)+touch in HOME_A for the same number
of env steps, with greedy eval every EVAL_EVERY steps (sample-efficiency
curves). Then freeze the DynaQ artifacts (Q, transition model, bump model)
and run the blindfold table: {home A, home B} x {full state, no observation,
bump-only} plus a random-policy chance floor. Blind conditions choose
actions greedily on expected Q under a belief dead-reckoned through the
learned model; touch conditions additionally filter the belief with the
learned bump likelihood. Budgets are matched exactly (both agents take
TRAIN_STEPS env steps; Dyna's planning is imagination, not env steps); eval
and blindfold rollouts use their own seed-derived rngs, never touch agent
state, and follow an identical protocol in every condition.

python experiments/exp1_blindfold.py --seeds 30 --out results/exp1.json [--smoke] [--jobs 20]
"""

import argparse
import time
from functools import partial

import numpy as np

from devrl.agents.dyna import BlindNavigator
from devrl.agents.qlearning import QLearner
from devrl.agents.touchnav import TouchDynaQ, TouchNavigator
from devrl.envs.gridhome import HOME_A, HOME_B, GridHome
from devrl.run import run_seeds, save_json
from devrl.stats import bootstrap_ci, iqm, mann_whitney, time_to_threshold

HYPOTHESIS = (
    "H1 (world models): an agent that learned a world model of its home can "
    "keep acting when observation is removed — dead reckoning through the "
    "model, sharpened by a learned touch (bump) likelihood — but the same "
    "machinery fails in a stranger's home; and model-based Dyna dominates "
    "model-free Q-learning on sample efficiency."
)

# blindfold table: condition -> (home, observation mode)
CONDITIONS = {
    "sighted-A": ("A", "state"),
    "blind-A": ("A", "dead"),
    "blind-A-touch": ("A", "touch"),
    "blind-B": ("B", "dead"),
    "blind-B-touch": ("B", "touch"),
    "sighted-B-transfer": ("B", "state"),
    "random-A": ("A", "random"),
}
COND_ORDER = list(CONDITIONS)
TRAJ_CONDS = ["blind-A", "blind-A-touch", "blind-B", "blind-B-touch",
              "random-A"]
MAPS = {"A": HOME_A, "B": HOME_B}
TRAIN_CONDS = ("qlearning-A", "dynaq-A")
KIND_ID = {"qlearning": 0, "dynaq": 1}


def make_config(smoke=False):
    # optimistic_init=1.0 (the max possible return) drives systematic
    # exploration for BOTH agents: pure eps-greedy random walk reaches the
    # fridge in <0.2% of 60-step episodes, so neither learner would ever
    # see reward inside the budget without it.
    cfg = dict(slip=0.1, gamma=0.97, lr=0.1, eps=0.1, optimistic_init=1.0,
               planning_steps=20, cap=60, eval_episodes=20, threshold=0.9,
               n_traj=3, n_boot=10_000)
    if smoke:
        cfg.update(train_steps=4000, eval_every=400, blind_episodes=10)
    else:
        cfg.update(train_steps=40_000, eval_every=1000, blind_episodes=30)
    return cfg


def greedy_tiebreak(Q, s, rng):
    """Greedy action from Q with random tie-break from an eval-owned rng."""
    q = Q[s]
    best = np.flatnonzero(q == q.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def belief_entropy_bits(belief):
    p = belief[belief > 1e-15]
    return float(-(p * np.log2(p)).sum() + 0.0)  # +0.0 avoids JSON -0.0


def eval_greedy(Q, map_str, cfg, seed, ckpt_idx):
    """Greedy success rate; fresh seeded env/rng per episode, so evaluation
    consumes no training budget and is identical across conditions."""
    wins = 0
    for ep in range(cfg["eval_episodes"]):
        env = GridHome(map_str, slip=cfg["slip"],
                       rng=np.random.default_rng((seed, 30, ckpt_idx, ep)))
        rng = np.random.default_rng((seed, 31, ckpt_idx, ep))
        s = env.reset()
        for _ in range(cfg["cap"]):
            s, _, done, _ = env.step(greedy_tiebreak(Q, s, rng))
            if done:
                wins += 1
                break
    return wins / cfg["eval_episodes"]


def train_agent(kind, cfg, seed):
    """Train in HOME_A for exactly cfg['train_steps'] env steps."""
    env = GridHome(HOME_A, slip=cfg["slip"],
                   rng=np.random.default_rng((seed, 10, KIND_ID[kind])))
    common = dict(n_states=env.n_states, n_actions=env.n_actions,
                  lr=cfg["lr"], gamma=cfg["gamma"], eps=cfg["eps"],
                  optimistic_init=cfg["optimistic_init"],
                  rng=np.random.default_rng((seed, 11, KIND_ID[kind])))
    agent = (TouchDynaQ(planning_steps=cfg["planning_steps"], **common)
             if kind == "dynaq" else QLearner(**common))
    curve, ckpts = [], []
    s, ep_len = env.reset(), 0
    for t in range(1, cfg["train_steps"] + 1):
        a = agent.act(s)
        s2, r, done, info = env.step(a)
        if kind == "dynaq":
            agent.observe_touch(s, a, info["bump"])
        agent.update(s, a, r, s2, done)
        ep_len += 1
        if done or ep_len >= cfg["cap"]:
            s, ep_len = env.reset(), 0
        else:
            s = s2
        if t % cfg["eval_every"] == 0:
            curve.append(eval_greedy(agent.Q, HOME_A, cfg, seed, len(ckpts)))
            ckpts.append(t)
    return agent, ckpts, curve


def _traj_row(t, env, nav):
    row = {"t": int(t), "true": [int(env.pos[0]), int(env.pos[1])],
           "believed": None, "entropy_bits": None}
    if nav is not None:
        row["believed"] = list(env.pos_of(int(np.argmax(nav.belief))))
        row["entropy_bits"] = belief_entropy_bits(nav.belief)
    return row


def blind_episode(env, agent, mode, cfg, policy_rng, record):
    """One rollout of a blindfold-table condition; trains nothing."""
    s = env.reset()
    nav = None
    if mode == "dead":
        nav = BlindNavigator(agent, agent.Q, s)
    elif mode == "touch":
        nav = TouchNavigator(agent, agent.Q, s, touch=agent)
    path, done, steps = [], False, cfg["cap"]
    for t in range(cfg["cap"]):
        row = _traj_row(t, env, nav) if record else None
        if mode == "state":
            a = greedy_tiebreak(agent.Q, s, policy_rng)
        elif mode == "random":
            a = int(policy_rng.integers(env.n_actions))
        else:
            a = nav.act()
        s, _, done, info = env.step(a)
        if mode == "dead":
            nav.advance(a)
        elif mode == "touch":
            nav.advance(a, info["bump"])
        if record:
            row["action"], row["bump"] = int(a), bool(info["bump"])
            path.append(row)
        if done:
            steps = t + 1
            break
    if record:
        end = _traj_row(steps, env, nav)
        end["action"] = end["bump"] = None
        path.append(end)
    return done, steps, path


def run_blind_condition(name, agent, cfg, seed):
    """All episodes of one condition, on the frozen HOME_A artifacts.

    Env rngs are keyed by (seed, episode) only, so every condition faces the
    same slip-noise stream; steps for failed episodes count as the cap."""
    home, mode = CONDITIONS[name]
    succ, steps_all, steps_succ, trajs = [], [], [], []
    for ep in range(cfg["blind_episodes"]):
        env = GridHome(MAPS[home], slip=cfg["slip"],
                       rng=np.random.default_rng((seed, 40, ep)))
        policy_rng = np.random.default_rng((seed, 41, ep))
        record = name in TRAJ_CONDS and ep < cfg["n_traj"]
        done, steps, path = blind_episode(env, agent, mode, cfg, policy_rng,
                                          record)
        succ.append(done)
        steps_all.append(steps)
        if done:
            steps_succ.append(steps)
        if record:
            trajs.append({"episode": ep, "success": bool(done),
                          "steps": int(steps), "path": path})
    metrics = {
        "success_rate": float(np.mean(succ)),
        "mean_steps": float(np.mean(steps_all)),
        "mean_steps_success": float(np.mean(steps_succ)) if steps_succ
        else None,
    }
    return metrics, trajs


def run_seed(seed, cfg):
    """One seed: train both agents in HOME_A, then the blindfold table on
    the frozen DynaQ artifacts. Module-top-level for fork pickling."""
    _, ckpts, q_curve = train_agent("qlearning", cfg, seed)
    d_agent, _, d_curve = train_agent("dynaq", cfg, seed)
    curves = {"qlearning-A": q_curve, "dynaq-A": d_curve}
    blind, trajs = {}, {}
    for name in COND_ORDER:
        metrics, t = run_blind_condition(name, d_agent, cfg, seed)
        blind[name] = metrics
        if name in TRAJ_CONDS:
            trajs[name] = t
    return {
        "seed": seed,
        "checkpoints": ckpts,
        "curves": curves,
        "t90": {k: time_to_threshold(ckpts, v, cfg["threshold"])
                for k, v in curves.items()},
        "final_success": {k: v[-1] for k, v in curves.items()},
        "blind": blind,
        "trajectories": trajs,
    }


# ------------------------------------------------------------------ analysis

def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj, running = [0.0] * m, 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, float(running))
    return adj


def censor_times(times, budget):
    """None (threshold never reached) -> budget+1: conservative for ranks."""
    vals = [budget + 1 if t is None else t for t in times]
    return vals, float(np.mean([t is None for t in times]))


def safe_mann_whitney(a, b):
    """mann_whitney, but fully tied samples get p=1 instead of NaN."""
    both = np.concatenate([np.asarray(a, float), np.asarray(b, float)])
    if float(np.ptp(both)) == 0.0:
        return {"u": len(a) * len(b) / 2.0, "p": 1.0}
    return mann_whitney(a, b)


def _home_viz(map_str):
    env = GridHome(map_str, rng=np.random.default_rng(0))
    return {
        "ascii": map_str.strip().splitlines(),
        "shape": [env.H, env.W],
        "walls": sorted([int(r), int(c)] for r, c in env.walls_rc()),
        "start": [int(env.start[0]), int(env.start[1])],
        "goal": [int(env.goal[0]), int(env.goal[1])],
        "shortest_path_len": int(env.shortest_path_len()),
    }


def aggregate(results, cfg, smoke, wall_clock):
    rng = np.random.default_rng(0)
    n_seeds = len(results)
    ckpts = results[0]["checkpoints"]

    def ci(vals):
        return list(bootstrap_ci(vals, n_boot=cfg["n_boot"], rng=rng,
                                 statistic=iqm))

    conditions = {}
    for cond in TRAIN_CONDS:
        t90 = [r["t90"][cond] for r in results]
        _, cfrac = censor_times(t90, cfg["train_steps"])
        conditions[cond] = {
            "phase": "train",
            "per_seed": {
                "final_success": [r["final_success"][cond] for r in results],
                "t90_step": t90,
            },
            "t90_censored_frac": cfrac,
        }
    for name in COND_ORDER:
        conditions[name] = {
            "phase": "blindfold",
            "per_seed": {k: [r["blind"][name][k] for r in results]
                         for k in ("success_rate", "mean_steps",
                                   "mean_steps_success")},
        }

    curves = {"checkpoints": ckpts, "conditions": {}}
    for cond in TRAIN_CONDS:
        per = np.array([r["curves"][cond] for r in results])  # seeds x ckpts
        band = {"iqm": [], "ci_lo": [], "ci_hi": []}
        for i in range(len(ckpts)):
            band["iqm"].append(iqm(per[:, i]))
            lo, hi = ci(per[:, i])
            band["ci_lo"].append(lo)
            band["ci_hi"].append(hi)
        curves["conditions"][cond] = band

    succ = {n: conditions[n]["per_seed"]["success_rate"] for n in COND_ORDER}
    steps = {n: conditions[n]["per_seed"]["mean_steps"] for n in COND_ORDER}
    d_t90, d_frac = censor_times([r["t90"]["dynaq-A"] for r in results],
                                 cfg["train_steps"])
    q_t90, q_frac = censor_times([r["t90"]["qlearning-A"] for r in results],
                                 cfg["train_steps"])

    def entry(name, family, metric, a_name, b_name, a, b, predicted):
        mw = safe_mann_whitney(a, b)
        return {"name": name, "family": family, "metric": metric,
                "a": a_name, "b": b_name, "iqm_a": iqm(a), "iqm_b": iqm(b),
                "u": mw["u"], "p": mw["p"], "predicted": predicted}

    primary = [
        entry("dyna_faster_than_q_t90", "primary",
              "env steps to 90% eval success (censored -> budget+1)",
              "dynaq-A", "qlearning-A", d_t90, q_t90, "a < b"),
        entry("blindA_beats_blindB", "primary", "blindfold success rate",
              "blind-A", "blind-B", succ["blind-A"], succ["blind-B"],
              "a > b"),
        entry("touch_beats_pure_deadreckoning", "primary",
              "blindfold success rate", "blind-A-touch", "blind-A",
              succ["blind-A-touch"], succ["blind-A"], "a > b"),
    ]
    for t, p_h in zip(primary, holm([t["p"] for t in primary])):
        right_way = (t["iqm_a"] < t["iqm_b"] if t["predicted"] == "a < b"
                     else t["iqm_a"] > t["iqm_b"])
        t["p_holm"] = p_h
        t["significant"] = bool(p_h < 0.05 and right_way)
    primary[0]["censored_frac_a"] = d_frac
    primary[0]["censored_frac_b"] = q_frac

    secondary = [
        entry("blindB_vs_randomA_chance_floor", "secondary",
              "blindfold success rate", "blind-B", "random-A",
              succ["blind-B"], succ["random-A"],
              "a ~ b (chance floor; descriptive)"),
        entry("sightedB_transfer_below_sightedA", "secondary",
              "blindfold success rate", "sighted-B-transfer", "sighted-A",
              succ["sighted-B-transfer"], succ["sighted-A"], "a < b"),
    ]
    for t in secondary:
        t["significant"] = bool(t["p"] < 0.05)

    dyna_faster = primary[0]["significant"]
    stranger_fails = bool(primary[1]["significant"] and
                          iqm(succ["blind-B"]) - iqm(succ["random-A"]) < 0.15)
    home_works = bool(iqm(succ["blind-A-touch"])
                      >= 0.8 * iqm(succ["sighted-A"]))
    touch_helps = bool(iqm(succ["blind-A-touch"]) >= iqm(succ["blind-A"]))
    supported = bool(dyna_faster and stranger_fails and home_works)
    summary = (
        f"Blindfolded dead reckoning through the home-learned model keeps "
        f"working in HOME_A (success IQM: sighted {iqm(succ['sighted-A']):.2f}, "
        f"blind {iqm(succ['blind-A']):.2f}, blind+touch "
        f"{iqm(succ['blind-A-touch']):.2f}) but collapses in the stranger's "
        f"HOME_B (blind {iqm(succ['blind-B']):.2f} vs random floor "
        f"{iqm(succ['random-A']):.2f}); DynaQ reached 90% eval success by "
        f"step {iqm(d_t90):.0f} (censored {d_frac:.0%}) vs "
        f"{iqm(q_t90):.0f} (censored {q_frac:.0%}) for Q-learning."
    )

    viz = {
        "homes": {"A": _home_viz(HOME_A), "B": _home_viz(HOME_B)},
        "action_names": ["up", "right", "down", "left"],
        "cap": cfg["cap"],
        "slip": cfg["slip"],
        "blind_summary": [
            {"name": n, "success_iqm": iqm(succ[n]), "success_ci": ci(succ[n]),
             "steps_iqm": iqm(steps[n]), "steps_ci": ci(steps[n])}
            for n in COND_ORDER
        ],
        "sample_efficiency": {
            "threshold": cfg["threshold"],
            "t90": {
                "dynaq-A": {"iqm": iqm(d_t90), "ci": ci(d_t90),
                            "censored_frac": d_frac},
                "qlearning-A": {"iqm": iqm(q_t90), "ci": ci(q_t90),
                                "censored_frac": q_frac},
            },
            "final_success": {
                c: {"iqm": iqm(conditions[c]["per_seed"]["final_success"]),
                    "ci": ci(conditions[c]["per_seed"]["final_success"])}
                for c in TRAIN_CONDS
            },
        },
        "blind_trajectories": results[0]["trajectories"],
        "trajectory_legend": {
            "true": "actual [row, col] before the action at t",
            "believed": "argmax of the belief, [row, col]; null for random-A",
            "entropy_bits": "belief Shannon entropy in bits; null for random-A",
            "action": "0=up 1=right 2=down 3=left; null on the terminal row",
            "bump": "touch signal felt after the action; null on terminal row",
        },
        "predictions": [
            "blind-A close to sighted-A; touch closes most of the slip gap",
            "blind-B at or near the random-A chance floor",
            "DynaQ reaches 90% success in fewer env steps than Q-learning",
        ],
    }

    return {
        "experiment": "exp1_blindfold",
        "hypothesis": HYPOTHESIS,
        "config": {**cfg, "n_seeds": n_seeds, "smoke": smoke,
                   "budget_note": ("budget counts env steps only; Dyna "
                                   "planning is imagination; eval rollouts "
                                   "are budget-free and identical across "
                                   "conditions")},
        "conditions": conditions,
        "curves": curves,
        "tests": primary + secondary,
        "conclusion": {"supported": supported, "summary": summary,
                       "criteria": {"dyna_sample_efficiency": dyna_faster,
                                    "blind_home_works": home_works,
                                    "blind_stranger_fails": stranger_fails,
                                    "touch_helps": touch_helps}},
        "viz": viz,
        "wall_clock_s": float(wall_clock),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="2 seeds, ~10x reduced budget")
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()
    cfg = make_config(smoke=args.smoke)
    n_seeds = 2 if args.smoke else args.seeds
    t0 = time.time()
    results = run_seeds(partial(run_seed, cfg=cfg), n_seeds, n_jobs=args.jobs)
    wall = time.time() - t0
    out = aggregate(results, cfg, smoke=args.smoke, wall_clock=wall)
    save_json(args.out, out)
    print(f"exp1_blindfold: {n_seeds} seeds in {wall:.1f}s -> {args.out}")
    for name in COND_ORDER:
        vals = [r["blind"][name]["success_rate"] for r in results]
        print(f"  {name:>20}: success IQM {iqm(vals):.2f}")
    print(f"  supported={out['conclusion']['supported']}: "
          f"{out['conclusion']['summary']}")


if __name__ == "__main__":
    main()
