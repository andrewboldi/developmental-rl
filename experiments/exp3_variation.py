"""EXP3 — Variation Practice (H3: contextual interference).

Blocked practice (drill passage A, then B, then C) versus interleaved
practice (uniform random passage each episode) on PianoPiece, at exactly
equal episode budgets, with a linear-Q agent whose shared motif weights are
the interference substrate. Prediction — the Shea & Morgan (1979) crossover:
blocked looks better during acquisition, interleaved wins retention on
A/B/C and transfer to a novel passage built from the same motifs.

Both conditions of a seed share one generated piece and identically seeded
rng streams. Greedy evals (learning OFF) run on all four passages at every
checkpoint for every condition — identical protocol, zero training budget.

Usage:
    python experiments/exp3_variation.py --seeds 40 --out results/exp3.json
                                         [--smoke] [--jobs 20]
"""

import argparse
import functools
import time

import numpy as np

from devrl.agents.linearq import LinearQ
from devrl.envs.piano import PianoPiece
from devrl.run import run_seeds, save_json
from devrl.stats import bootstrap_ci, iqm, mann_whitney

# lr sits at the stable edge for linear TD here (0.5+ diverges); the budget
# ends training before both conditions saturate, where the crossover lives.
LR = 0.4
GAMMA = 0.9
EPS = 0.1
EVAL_EPISODES = 5  # greedy rollouts per passage per checkpoint
ALPHA = 0.05
CONDITIONS = ("blocked", "interleaved")
PASSAGE_NAMES = ("A", "B", "C", "novel")

HYPOTHESIS = (
    "H3 (variation practice): blocked practice looks better during "
    "acquisition but interleaved practice wins retention and transfer, "
    "because drilling one passage at a time overwrites the shared motif "
    "weights the other passages rely on (contextual interference; the "
    "Shea & Morgan 1979 crossover, in silico)."
)


def make_schedule(condition, episodes, rng):
    """Passage index per training episode; equal budgets by construction."""
    if condition == "blocked":
        if episodes % 3:
            raise ValueError("blocked schedule needs episodes divisible by 3")
        return np.repeat(np.arange(3), episodes // 3)
    return rng.integers(0, 3, size=episodes)


def greedy_rollout(env, agent, passage, rng):
    """One greedy episode, learning OFF; returns (keys played, reward sum)."""
    played, total = [], 0.0
    s = env.reset(passage)
    done = False
    while not done:
        q = agent.q(env.features(*s))
        best = np.flatnonzero(q == q.max())
        a = int(best[0]) if len(best) == 1 else int(rng.choice(best))
        s, r, done, _ = env.step(a)
        played.append(a)
        total += r
    return played, total


def greedy_score(env, agent, passage, rng, n_episodes=EVAL_EPISODES):
    """Mean fraction-correct over greedy rollouts (consumes no budget)."""
    total = sum(greedy_rollout(env, agent, passage, rng)[1]
                for _ in range(n_episodes))
    return total / (n_episodes * env.passage_len)


def _evaluate(env, agent, rng):
    """Greedy score on every passage — identical for all conditions."""
    return [greedy_score(env, agent, p, rng) for p in range(env.n_passages)]


def _run_condition(env, condition, episodes, eval_every, children):
    """Train one condition; children = (agent, schedule, eval) seed children."""
    agent_rng, sched_rng, eval_rng = (np.random.default_rng(c) for c in children)
    agent = LinearQ(env.n_features, env.n_actions, lr=LR, gamma=GAMMA,
                    eps=EPS, rng=agent_rng)
    schedule = make_schedule(condition, episodes, sched_rng)
    checkpoints, evals, train_scores = [], [], []
    train_steps = 0
    for ep in range(episodes + 1):
        if ep % eval_every == 0:
            checkpoints.append(ep)
            evals.append(_evaluate(env, agent, eval_rng))
        if ep == episodes:
            break
        s = env.reset(int(schedule[ep]))
        ep_return, done = 0.0, False
        while not done:
            phi = env.features(*s)
            a = agent.act(phi)
            s2, r, done, _ = env.step(a)
            agent.update(phi, a, r, None if done else env.features(*s2), done)
            train_steps += 1
            ep_return += r
            s = s2
        train_scores.append(ep_return / env.passage_len)

    # acquisition = greedy score on what was just being practiced: for
    # blocked, the phase of the episode before the checkpoint; for
    # interleaved, all three passages at once. This is the fool's-progress
    # curve — it flatters blocked practice.
    acquisition = []
    for i, ep in enumerate(checkpoints):
        if condition == "blocked":
            acquisition.append(evals[i][int(schedule[max(ep - 1, 0)])])
        else:
            acquisition.append(float(np.mean(evals[i][:3])))
    practice = [float(np.mean(train_scores[checkpoints[i]:checkpoints[i + 1]]))
                for i in range(len(checkpoints) - 1)]
    final = evals[-1]
    rollouts = {name: greedy_rollout(env, agent, p, eval_rng)[0]
                for p, name in enumerate(PASSAGE_NAMES)}
    return {
        "train_steps": train_steps,
        "checkpoint_episodes": checkpoints,
        "eval_scores": [[float(v) for v in row] for row in evals],
        "acquisition_curve": [float(v) for v in acquisition],
        "acquisition_mean": float(np.mean(acquisition[1:])),  # skip pre-train
        "practice_curve": practice,
        "retention": {"A": final[0], "B": final[1], "C": final[2],
                      "mean": float(np.mean(final[:3]))},
        "transfer": final[3],
        "final_rollouts": rollouts,
    }


def run_seed(seed, episodes, eval_every):
    """Both conditions on one shared piece, from seed-derived rng streams."""
    if episodes % eval_every:
        raise ValueError("episodes must be divisible by eval_every")
    children = np.random.SeedSequence(seed).spawn(4)  # env, agent, sched, eval
    env = PianoPiece(rng=np.random.default_rng(children[0]))
    out = {"seed": seed, "structure": env.structure()}
    for cond in CONDITIONS:
        # identically seeded fresh streams per condition: matched randomness
        out[cond] = _run_condition(env, cond, episodes, eval_every,
                                   children[1:4])
    return out


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, input order preserved."""
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def _metric(results, cond, name):
    if name == "retention_mean":
        return [r[cond]["retention"]["mean"] for r in results]
    if name == "transfer":
        return [r[cond]["transfer"] for r in results]
    if name == "acquisition_mean":
        return [r[cond]["acquisition_mean"] for r in results]
    return [r[cond]["retention"][name.removeprefix("retention_")]
            for r in results]


def _ci(values, rng, n_boot):
    lo, hi = bootstrap_ci(values, n_boot=n_boot, rng=rng, statistic=iqm)
    return {"iqm": iqm(values), "lo": lo, "hi": hi}


def _ci_series(mat, rng, n_boot):
    """Per-checkpoint IQM + bootstrap CI for a (seeds, checkpoints) array."""
    out = {"iqm": [], "lo": [], "hi": []}
    for col in np.asarray(mat, dtype=float).T:
        r = _ci(col, rng, n_boot)
        for k in out:
            out[k].append(r[k])
    return out


def aggregate(results, config):
    """Full output JSON (shared contract) from per-seed results."""
    rng = np.random.default_rng(0)  # reproducible CIs
    n_boot = config["n_boot"]
    ck = results[0]["blocked"]["checkpoint_episodes"]
    assert all(r[c]["checkpoint_episodes"] == ck
               for r in results for c in CONDITIONS)
    episodes = ck[-1]
    steps = [e * PianoPiece.passage_len for e in ck]

    conditions = {}
    for cond in CONDITIONS:
        seeds = []
        for r in results:
            c = r[cond]
            seeds.append({"seed": r["seed"],
                          **{k: c[k] for k in
                             ("train_steps", "retention", "transfer",
                              "acquisition_mean", "acquisition_curve",
                              "eval_scores", "practice_curve",
                              "final_rollouts")}})
        conditions[cond] = {"n_seeds": len(results), "seeds": seeds}

    # ---- curves on the shared checkpoint grid
    def eval_col(cond, j):
        return [[row[j] for row in r[cond]["eval_scores"]] for r in results]

    metrics = {}
    for cond in CONDITIONS:
        series = {
            "acquisition": [r[cond]["acquisition_curve"] for r in results],
            "retention_A": eval_col(cond, 0),
            "retention_B": eval_col(cond, 1),
            "retention_C": eval_col(cond, 2),
            "transfer": eval_col(cond, 3),
        }
        series["retention_mean"] = (
            np.mean([series[f"retention_{p}"] for p in "ABC"], axis=0))
        for name, mat in series.items():
            metrics.setdefault(name, {})[cond] = _ci_series(mat, rng, n_boot)
    curves = {"episodes": ck, "steps": steps, "metrics": metrics}

    # ---- primary comparisons (Holm-Bonferroni family)
    primaries = [
        ("acquisition", "acquisition_mean", "blocked", "interleaved"),
        ("retention", "retention_mean", "interleaved", "blocked"),
        ("transfer", "transfer", "interleaved", "blocked"),
    ]
    tests, keys = {}, []
    for label, metric, win, lose in primaries:
        a, b = _metric(results, win, metric), _metric(results, lose, metric)
        mw = mann_whitney(a, b)
        key = f"{label}_{win}_gt_{lose}"
        keys.append(key)
        tests[key] = {"family": "primary", "metric": metric,
                      "predicted_winner": win,
                      "iqm": {win: iqm(a), lose: iqm(b)},
                      "direction_ok": bool(iqm(a) > iqm(b)),
                      "u": mw["u"], "p": mw["p"]}
    for key, ph in zip(keys, holm([tests[k]["p"] for k in keys])):
        tests[key]["p_holm"] = ph
        tests[key]["significant"] = bool(ph < ALPHA)
    for p in "ABC":  # secondary: which passages drive the retention gap
        a = _metric(results, "interleaved", f"retention_{p}")
        b = _metric(results, "blocked", f"retention_{p}")
        mw = mann_whitney(a, b)
        tests[f"retention_{p}_interleaved_gt_blocked"] = {
            "family": "secondary", "metric": f"retention_{p}",
            "predicted_winner": "interleaved",
            "iqm": {"interleaved": iqm(a), "blocked": iqm(b)},
            "direction_ok": bool(iqm(a) > iqm(b)),
            "u": mw["u"], "p": mw["p"],
            "significant_uncorrected": bool(mw["p"] < ALPHA)}

    prim = [tests[k] for k in keys]
    supported = all(t["direction_ok"] and t["significant"] for t in prim)

    def _vs(t):
        w = t["predicted_winner"]
        l = "interleaved" if w == "blocked" else "blocked"
        rel = ">" if t["direction_ok"] else "<="
        return (f"{w} {t['iqm'][w]:.3f} {rel} {l} {t['iqm'][l]:.3f} "
                f"(Holm p={t['p_holm']:.2g})")
    summary = (f"Acquisition: {_vs(prim[0])}. Retention: {_vs(prim[1])}. "
               f"Transfer: {_vs(prim[2])}. "
               + ("Full Shea-Morgan crossover: blocked practice was fool's "
                  "progress — it looked better while practicing and lost "
                  "at test." if supported else
                  "The predicted crossover was NOT fully confirmed."))

    # ---- per-condition scalar summaries for bars / crossover chart
    scalars = {cond: {m: _ci(_metric(results, cond, m), rng, n_boot)
                      for m in ("acquisition_mean", "retention_mean",
                                "transfer", "retention_A", "retention_B",
                                "retention_C")}
               for cond in CONDITIONS}
    ex = results[0]
    correct = {name: ex["structure"]["correct_keys"][p]
               for p, name in enumerate(PASSAGE_NAMES)}
    viz = {
        "story": ("Blocked drilling of one passage at a time rides the "
                  "shared motif weights to fast visible progress, then "
                  "each new passage overwrites them; interleaving forces "
                  "the weights to serve all passages at once — worse "
                  "practice scores, better musician."),
        "n_keys": PianoPiece.n_keys, "n_motifs": PianoPiece.n_motifs,
        "motif_len": PianoPiece.motif_len,
        "passage_len": PianoPiece.passage_len,
        "passage_names": list(PASSAGE_NAMES),
        "conditions": list(CONDITIONS),
        "example_structure": {"seed": ex["seed"], **ex["structure"]},
        "phase_boundaries_episodes": [episodes // 3, 2 * episodes // 3],
        "curves": curves,
        "practice_curves": {
            "episodes": ck[1:],
            **{cond: _ci_series([r[cond]["practice_curve"] for r in results],
                                rng, n_boot) for cond in CONDITIONS}},
        "retention_bars": {
            "passages": ["A", "B", "C", "mean"],
            **{cond: {"iqm": [scalars[cond][f"retention_{p}"]["iqm"]
                              for p in ("A", "B", "C")]
                      + [scalars[cond]["retention_mean"]["iqm"]],
                      "lo": [scalars[cond][f"retention_{p}"]["lo"]
                             for p in ("A", "B", "C")]
                      + [scalars[cond]["retention_mean"]["lo"]],
                      "hi": [scalars[cond][f"retention_{p}"]["hi"]
                             for p in ("A", "B", "C")]
                      + [scalars[cond]["retention_mean"]["hi"]]}
               for cond in CONDITIONS}},
        "transfer_bars": {cond: scalars[cond]["transfer"]
                          for cond in CONDITIONS},
        "crossover": {
            "phases": ["acquisition", "retention", "transfer"],
            **{cond: {"iqm": [scalars[cond][m]["iqm"] for m in
                              ("acquisition_mean", "retention_mean",
                               "transfer")],
                      "lo": [scalars[cond][m]["lo"] for m in
                             ("acquisition_mean", "retention_mean",
                              "transfer")],
                      "hi": [scalars[cond][m]["hi"] for m in
                             ("acquisition_mean", "retention_mean",
                              "transfer")]}
               for cond in CONDITIONS}},
        "example_rollouts": {
            cond: {name: {"correct": correct[name],
                          "played": ex[cond]["final_rollouts"][name],
                          "ok": [a == c for a, c in
                                 zip(ex[cond]["final_rollouts"][name],
                                     correct[name])]}
                   for name in PASSAGE_NAMES}
            for cond in CONDITIONS},
        "per_seed_points": {
            cond: {m: _metric(results, cond, m)
                   for m in ("acquisition_mean", "retention_mean", "transfer")}
            for cond in CONDITIONS},
    }

    return {"experiment": "exp3_variation",
            "hypothesis": HYPOTHESIS,
            "config": config,
            "conditions": conditions,
            "curves": curves,
            "tests": tests,
            "conclusion": {"supported": bool(supported), "summary": summary},
            "viz": viz}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=None,
                    help="number of seeds (default 40, smoke 2)")
    ap.add_argument("--out", default="results/exp3.json")
    ap.add_argument("--smoke", action="store_true",
                    help="2 seeds, ~10x reduced budget")
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()
    episodes, eval_every, n_boot = ((45, 5, 2000) if args.smoke
                                    else (450, 15, 10000))
    n_seeds = args.seeds if args.seeds is not None else (2 if args.smoke else 40)
    config = {"seeds": n_seeds, "episodes": episodes,
              "steps_per_episode": PianoPiece.passage_len,
              "eval_every_episodes": eval_every,
              "eval_episodes_per_passage": EVAL_EPISODES,
              "lr": LR, "gamma": GAMMA, "eps": EPS,
              "n_boot": n_boot, "alpha": ALPHA, "smoke": args.smoke}

    t0 = time.time()
    fn = functools.partial(run_seed, episodes=episodes, eval_every=eval_every)
    results = run_seeds(fn, n_seeds, n_jobs=min(args.jobs, n_seeds))
    t1 = time.time()
    out = aggregate(results, config)
    t2 = time.time()
    out["timing"] = {"run_seconds": round(t1 - t0, 2),
                     "aggregate_seconds": round(t2 - t1, 2)}
    save_json(args.out, out)
    print(f"wrote {args.out}  (run {t1 - t0:.1f}s, aggregate {t2 - t1:.1f}s)")
    print(out["conclusion"]["summary"])


if __name__ == "__main__":
    main()
