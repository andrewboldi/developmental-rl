"""EXP1 — The Blindfold Test (H1: world models). v2 per DESIGN.md
"Amendments (v2) — EXP1".

Train Q-learning vs DynaQ(planning=20)+touch vs update-matched
Q-learning+replay (ReplayQ, 20 replayed updates per real step — the van
Hasselt 2019 fairness baseline) in HOME_A for the same number of env
steps, with greedy eval every EVAL_EVERY steps (sample-efficiency
curves). Then freeze the DynaQ artifacts (Q, transition model, bump model)
and run the blindfold table: {home A, home B} x {full state, no observation,
bump-only} plus random-policy chance floors in BOTH homes. Blind conditions
choose actions greedily on expected Q under a belief dead-reckoned through
the learned model; touch conditions additionally filter the belief with the
learned bump likelihood. Budgets are matched exactly (every agent takes
TRAIN_STEPS env steps; Dyna's planning and ReplayQ's replay are extra
updates, not env steps); eval and blindfold rollouts use their own
seed-derived rngs, never touch agent state, and follow an identical
protocol in every condition. Confirmatory seeds are offset by
--seed-offset (default 100: seeds 100..129, disjoint from every seed ever
used for tuning). Tests report Mann-Whitney (registered decision
statistic) AND Welch t p-values, Holm within the primary family for both;
the conclusion is per-claim verdicts (supported/refuted/null/boundary).

python experiments/exp1_blindfold.py --seeds 30 --out results/exp1.json [--smoke] [--jobs 20] [--seed-offset 100]
"""

import argparse
import time
from functools import partial

import numpy as np
from scipy.stats import ttest_ind

from devrl.agents.dyna import BlindNavigator
from devrl.agents.qlearning import QLearner
from devrl.agents.replayq import ReplayQ
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
    "random-B": ("B", "random"),  # v2: matched floor for HOME_B (A2)
}
COND_ORDER = list(CONDITIONS)
TRAJ_CONDS = ["blind-A", "blind-A-touch", "blind-B", "blind-B-touch",
              "random-A", "random-B"]
MAPS = {"A": HOME_A, "B": HOME_B}
TRAIN_CONDS = ("qlearning-A", "dynaq-A", "replayq-A")
KIND_ID = {"qlearning": 0, "dynaq": 1, "replayq": 2}


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
    if kind == "dynaq":
        agent = TouchDynaQ(planning_steps=cfg["planning_steps"], **common)
    elif kind == "replayq":
        # update-matched baseline (A3): same extra-update budget as Dyna's
        # planning, drawn from remembered experience instead of a model
        agent = ReplayQ(replay_steps=cfg["planning_steps"], **common)
    else:
        agent = QLearner(**common)
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


def run_seed(seed, cfg, seed_offset=0):
    """One seed: train the three agents in HOME_A, then the blindfold table
    on the frozen DynaQ artifacts. Module-top-level for fork pickling.

    The true seed is seed + seed_offset (A5): confirmatory runs use offset
    100 so seeds 100..129 are disjoint from every tuning/probe seed."""
    seed = seed + seed_offset
    _, ckpts, q_curve = train_agent("qlearning", cfg, seed)
    _, _, r_curve = train_agent("replayq", cfg, seed)
    d_agent, _, d_curve = train_agent("dynaq", cfg, seed)
    curves = {"qlearning-A": q_curve, "dynaq-A": d_curve,
              "replayq-A": r_curve}
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


def safe_welch(a, b):
    """Welch t p-value (ttest_ind, equal_var=False) with the A4 degenerate
    zero-variance guards: fully tied -> 1.0, two distinct constants -> 0.0."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if float(np.ptp(a)) == 0.0 and float(np.ptp(b)) == 0.0:
        return 1.0 if a[0] == b[0] else 0.0
    return float(ttest_ind(a, b, equal_var=False).pvalue)


def prediction_met(predicted, p, iqm_a, iqm_b):
    """Score a test's registered prediction (A4).

    Directional ('a < b' / 'a > b'): met iff the decision p (p_holm for
    primaries, raw p for secondaries) is < 0.05 with the IQMs ordered as
    predicted. Equivalence-style ('a ~ b'): met iff the RAW (uncorrected —
    deliberately the harder criterion) MW p >= 0.05 — 'no detectable
    difference', NOT a formal equivalence test."""
    if predicted.startswith("a ~ b"):
        return bool(p >= 0.05)
    if predicted.startswith("a < b"):
        return bool(p < 0.05 and iqm_a < iqm_b)
    return bool(p < 0.05 and iqm_a > iqm_b)


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
    r_t90, r_frac = censor_times([r["t90"]["replayq-A"] for r in results],
                                 cfg["train_steps"])

    def entry(name, family, metric, a_name, b_name, a, b, predicted):
        mw = safe_mann_whitney(a, b)
        return {"name": name, "family": family, "metric": metric,
                "a": a_name, "b": b_name, "iqm_a": iqm(a), "iqm_b": iqm(b),
                "u": mw["u"], "p": mw["p"], "p_welch": safe_welch(a, b),
                "predicted": predicted}

    t90_metric = "env steps to 90% eval success (censored -> budget+1)"
    primary = [
        entry("dyna_faster_than_q_t90", "primary", t90_metric,
              "dynaq-A", "qlearning-A", d_t90, q_t90, "a < b"),
        entry("replay_faster_than_q_t90", "primary", t90_metric,
              "replayq-A", "qlearning-A", r_t90, q_t90, "a < b"),
        entry("dyna_vs_replay_t90", "primary", t90_metric,
              "dynaq-A", "replayq-A", d_t90, r_t90,
              "a ~ b (update-matched; van Hasselt 2019: the replay buffer "
              "is a non-parametric model, so no model-specific edge is "
              "expected)"),
        entry("blindA_touch_beats_blindB_touch", "primary",
              "blindfold success rate", "blind-A-touch", "blind-B-touch",
              succ["blind-A-touch"], succ["blind-B-touch"], "a > b"),
        entry("touch_beats_pure_deadreckoning", "primary",
              "blindfold success rate", "blind-A-touch", "blind-A",
              succ["blind-A-touch"], succ["blind-A"], "a > b"),
    ]
    # Holm within the 5-test primary family, separately for the registered
    # decision statistic (Mann-Whitney) and the robustness Welch t (A4)
    for t, p_h, pw_h in zip(primary, holm([t["p"] for t in primary]),
                            holm([t["p_welch"] for t in primary])):
        t["p_holm"], t["p_welch_holm"] = p_h, pw_h
        if t["predicted"].startswith("a ~ b"):
            # difference detected at the decision level (no direction);
            # the equivalence-style prediction is scored on the RAW MW p
            t["significant"] = bool(p_h < 0.05)
            t["prediction_met"] = prediction_met(t["predicted"], t["p"],
                                                 t["iqm_a"], t["iqm_b"])
        else:
            # directional primaries: p_holm < 0.05 in the predicted
            # direction (v1 semantics; identical to prediction_met)
            t["significant"] = prediction_met(t["predicted"], p_h,
                                              t["iqm_a"], t["iqm_b"])
            t["prediction_met"] = t["significant"]
    cfracs = {"dynaq-A": d_frac, "qlearning-A": q_frac, "replayq-A": r_frac}
    for t in primary:
        if t["metric"] == t90_metric:
            t["censored_frac_a"] = cfracs[t["a"]]
            t["censored_frac_b"] = cfracs[t["b"]]

    secondary = [
        entry("blindA_beats_blindB", "secondary", "blindfold success rate",
              "blind-A", "blind-B", succ["blind-A"], succ["blind-B"],
              "a > b (pure dead-reckoning pair; v1 primary, demoted per A1)"),
        entry("blindB_vs_randomA_chance_floor", "secondary",
              "blindfold success rate", "blind-B", "random-A",
              succ["blind-B"], succ["random-A"],
              "a ~ b (cross-home floor; descriptive)"),
        entry("blindB_vs_randomB_matched_floor", "secondary",
              "blindfold success rate", "blind-B", "random-B",
              succ["blind-B"], succ["random-B"],
              "a > b (small residual competence above the matched floor)"),
        entry("blindBtouch_vs_randomB_matched_floor", "secondary",
              "blindfold success rate", "blind-B-touch", "random-B",
              succ["blind-B-touch"], succ["random-B"],
              "a > b (small residual competence above the matched floor)"),
        entry("sightedB_transfer_below_sightedA", "secondary",
              "blindfold success rate", "sighted-B-transfer", "sighted-A",
              succ["sighted-B-transfer"], succ["sighted-A"], "a < b"),
    ]
    for t in secondary:
        t["significant"] = bool(t["p"] < 0.05)
        t["prediction_met"] = prediction_met(t["predicted"], t["p"],
                                             t["iqm_a"], t["iqm_b"])

    # ---- per-claim verdicts (A6) -------------------------------------
    by_name = {t["name"]: t for t in primary + secondary}

    def sig_opposite(t):
        """Significant at the decision level in the unpredicted direction."""
        p = t.get("p_holm", t["p"])
        wrong = (t["iqm_a"] > t["iqm_b"] if t["predicted"].startswith("a < b")
                 else t["iqm_a"] < t["iqm_b"])
        return bool(p < 0.05 and wrong)

    d_iqm, q_iqm, r_iqm = iqm(d_t90), iqm(q_t90), iqm(r_t90)
    s_iqm = {n: iqm(succ[n]) for n in COND_ORDER}

    dq = by_name["dyna_faster_than_q_t90"]
    v1 = ("supported" if dq["prediction_met"]
          else "refuted" if sig_opposite(dq) else "null")

    dvr = by_name["dyna_vs_replay_t90"]
    rq = by_name["replay_faster_than_q_t90"]
    closure = ((q_iqm - r_iqm) / (q_iqm - d_iqm)
               if q_iqm != d_iqm else None)
    if dvr["significant"] and dvr["iqm_a"] < dvr["iqm_b"]:
        v2 = "supported"
    elif (dvr["significant"] and dvr["iqm_a"] > dvr["iqm_b"]) or (
            not dvr["significant"] and closure is not None
            and closure >= 0.8):
        v2 = "refuted"
    else:
        v2 = "null"
    closure_txt = ("undefined (no Dyna t90 advantage over Q)"
                   if closure is None else f"{closure:.0%}")

    def home_rule(cond):
        val, sighted = s_iqm[cond], s_iqm["sighted-A"]
        return ("supported" if val >= 0.8 * sighted
                else "refuted" if val < 0.5 * sighted else "boundary")

    v3, v4 = home_rule("blind-A"), home_rule("blind-A-touch")

    bat = by_name["blindA_touch_beats_blindB_touch"]
    at, bt = bat["iqm_a"], bat["iqm_b"]
    if bat["prediction_met"] and bt <= 0.5 * at:
        v5 = "supported"
    elif bat["prediction_met"] and bt <= 0.8 * at:
        v5 = "boundary"
    elif sig_opposite(bat) or bt > 0.8 * at:
        v5 = "refuted"
    else:
        v5 = "null"

    tbp = by_name["touch_beats_pure_deadreckoning"]
    v6 = ("supported" if tbp["prediction_met"]
          else "refuted" if sig_opposite(tbp) else "null")

    claims = [
        {"claim": "dyna-beats-q: DynaQ reaches 90% eval success in fewer "
                  "env steps than sample-matched (update-unmatched) "
                  "Q-learning",
         "verdict": v1,
         "evidence": (f"t90 IQM dynaq {d_iqm:.0f} vs qlearning {q_iqm:.0f} "
                      f"(censored {d_frac:.0%}/{q_frac:.0%}); MW p_holm "
                      f"{dq['p_holm']:.3g}, Welch p_holm "
                      f"{dq['p_welch_holm']:.3g}")},
        {"claim": "dyna-advantage-is-model-not-updates: the Dyna speedup "
                  "survives an update-matched replay baseline (ReplayQ, 20 "
                  "replayed updates per real step; van Hasselt 2019)",
         "verdict": v2,
         "evidence": (f"t90 IQM: qlearning {q_iqm:.0f}, replayq {r_iqm:.0f}, "
                      f"dynaq {d_iqm:.0f}; replay closes {closure_txt} of "
                      f"the Dyna advantage; dyna_vs_replay MW p "
                      f"{dvr['p']:.3g} (p_holm {dvr['p_holm']:.3g}, Welch "
                      f"p_holm {dvr['p_welch_holm']:.3g}); "
                      f"replay_faster_than_q MW p_holm {rq['p_holm']:.3g}")},
        {"claim": "pure-dead-reckoning-works-at-home: blind-A stays >= 0.8x "
                  "sighted-A (DESIGN v1's literal prediction, scored "
                  "honestly; refuted below 0.5x, boundary between)",
         "verdict": v3,
         "evidence": (f"success IQM blind-A {s_iqm['blind-A']:.3f} vs "
                      f"sighted-A {s_iqm['sighted-A']:.3f}")},
        {"claim": "touch-filtered-blind-works-at-home: blind-A-touch stays "
                  ">= 0.8x sighted-A (refuted below 0.5x, boundary between)",
         "verdict": v4,
         "evidence": (f"success IQM blind-A-touch "
                      f"{s_iqm['blind-A-touch']:.3f} vs sighted-A "
                      f"{s_iqm['sighted-A']:.3f}")},
        {"claim": "stranger-collapse: judged on the symmetric touch pair — "
                  "the same blind+touch machinery falls far below home "
                  "performance in HOME_B (<= 0.5x home), modestly above a "
                  "matched random floor",
         "verdict": v5,
         "evidence": (f"success IQM blind-A-touch {at:.3f} vs blind-B-touch "
                      f"{bt:.3f} (MW p_holm {bat['p_holm']:.3g}, Welch "
                      f"p_holm {bat['p_welch_holm']:.3g}); random floors: "
                      f"random-B {s_iqm['random-B']:.3f} (matched), "
                      f"random-A {s_iqm['random-A']:.3f} (cross-home); "
                      f"blind-B {s_iqm['blind-B']:.3f} vs matched floor MW "
                      f"p {by_name['blindB_vs_randomB_matched_floor']['p']:.3g}, "
                      f"blind-B-touch vs matched floor MW p "
                      f"{by_name['blindBtouch_vs_randomB_matched_floor']['p']:.3g}")},
        {"claim": "touch-helps-at-home: bump filtering beats pure dead "
                  "reckoning in HOME_A",
         "verdict": v6,
         "evidence": (f"success IQM blind-A-touch "
                      f"{s_iqm['blind-A-touch']:.3f} vs blind-A "
                      f"{s_iqm['blind-A']:.3f}; MW p_holm "
                      f"{tbp['p_holm']:.3g}, Welch p_holm "
                      f"{tbp['p_welch_holm']:.3g}")},
    ]
    summary = (
        f"t90 to 90% eval success (IQM env steps): dynaq {d_iqm:.0f}, "
        f"update-matched replayq {r_iqm:.0f}, qlearning {q_iqm:.0f} "
        f"(censored {d_frac:.0%}/{r_frac:.0%}/{q_frac:.0%}). HOME_A "
        f"blindfold success IQM: sighted {s_iqm['sighted-A']:.2f}, pure "
        f"blind {s_iqm['blind-A']:.2f}, blind+touch "
        f"{s_iqm['blind-A-touch']:.2f}. Stranger HOME_B, same machinery: "
        f"blind+touch {s_iqm['blind-B-touch']:.2f}, pure blind "
        f"{s_iqm['blind-B']:.2f}, vs matched random-B floor "
        f"{s_iqm['random-B']:.2f} (cross-home random-A "
        f"{s_iqm['random-A']:.2f}) — registered claim: far below home "
        f"performance, modestly above a matched random floor "
        f"(verdict: {v5}). Verdicts: "
        + "; ".join(f"{c['claim'].split(':')[0]}={c['verdict']}"
                    for c in claims) + "."
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
                "replayq-A": {"iqm": iqm(r_t90), "ci": ci(r_t90),
                              "censored_frac": r_frac},
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
            "DynaQ reaches 90% success in fewer env steps than 1-update "
            "Q-learning",
            "update-matched replay reproduces most of the speedup: replayq "
            "faster than qlearning on t90, dynaq ~ replayq (van Hasselt "
            "2019)",
            "blind-A-touch far above blind-B-touch — the symmetric "
            "stranger-collapse pair",
            "blind-B far below home performance, modestly above the matched "
            "random-B floor (and above the cross-home random-A floor)",
            "touch filtering beats pure dead reckoning at home; pure "
            "blind-A scored honestly against sighted-A (v1 literal "
            "prediction)",
        ],
    }

    return {
        "experiment": "exp1_blindfold",
        "hypothesis": HYPOTHESIS,
        "config": {**cfg, "n_seeds": n_seeds, "smoke": smoke,
                   "seed_offset": int(min(r["seed"] for r in results)),
                   "budget_note": ("budget counts env steps only; Dyna "
                                   "planning and ReplayQ replay are extra "
                                   "updates, not env steps; eval rollouts "
                                   "are budget-free and identical across "
                                   "conditions")},
        "conditions": conditions,
        "curves": curves,
        "tests": primary + secondary,
        "conclusion": {"claims": claims, "summary": summary},
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
    ap.add_argument("--seed-offset", type=int, default=100,
                    help="true seed = index + offset (A5); the default 100 "
                         "keeps confirmatory seeds 100..N+99 disjoint from "
                         "every seed ever used for tuning (0..29) and from "
                         "the verifiers' rerun (1000..1029)")
    args = ap.parse_args()
    cfg = make_config(smoke=args.smoke)
    n_seeds = 2 if args.smoke else args.seeds
    t0 = time.time()
    results = run_seeds(partial(run_seed, cfg=cfg,
                                seed_offset=args.seed_offset),
                        n_seeds, n_jobs=args.jobs)
    wall = time.time() - t0
    out = aggregate(results, cfg, smoke=args.smoke, wall_clock=wall)
    save_json(args.out, out)
    print(f"exp1_blindfold: {n_seeds} seeds "
          f"({args.seed_offset}..{args.seed_offset + n_seeds - 1}) "
          f"in {wall:.1f}s -> {args.out}")
    for name in COND_ORDER:
        vals = [r["blind"][name]["success_rate"] for r in results]
        print(f"  {name:>20}: success IQM {iqm(vals):.2f}")
    for c in out["conclusion"]["claims"]:
        print(f"  {c['claim'].split(':')[0]:>36}: {c['verdict']}")
    print(f"  {out['conclusion']['summary']}")


if __name__ == "__main__":
    main()
