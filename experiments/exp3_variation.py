"""EXP3 — Variation Practice (H3: contextual interference), v2.

Blocked practice (drill one passage at a time) versus interleaved practice
(uniform random passage each episode) on PianoPiece, at exactly equal
episode budgets, with a linear-Q agent whose shared feature slots (motif
and position onehots) are the interference substrate. Prediction — the
Shea & Morgan (1979) crossover: blocked looks better during acquisition,
interleaved wins retention on A/B/C and transfer to a novel passage built
from the same motifs.

v2 hardening (registered in DESIGN.md "Amendments (v2) — EXP3"):
- Blocked passage order is counterbalanced across seeds: seed % 6 selects
  one of the 6 permutations of A/B/C, killing the fixed-order recency
  confound. Order-relative retention metrics (`last` = the passage drilled
  last under this seed's order, `earlier` = mean of the other two) replace
  the fixed per-passage secondaries.
- Mechanism-control pair (`blocked-nomotif`, `interleaved-nomotif`): the
  identical protocol with feature_map="local" — phi = onehot((passage,
  position)) only, NO shared slots (a passage-local tabular equivalent).
  Registered prediction: the retention/transfer crossover vanishes and
  nomotif transfer sits at chance.
- Every registered test reports BOTH two-sided Mann-Whitney U and Welch's
  t (scipy ttest_ind, equal_var=False); Holm is applied within the primary
  family to each statistic separately.
- Per-claim verdicts (supported | refuted | null | boundary) replace the
  boolean conclusion. Acquisition fragility at n=40 is disclosed: the
  registered rule maps direction-ok-but-not-significant to `boundary`.
- Fresh confirmatory seeds via --seed-offset (default 100): seeds
  100..139, disjoint from every seed used for tuning (0-39).

Both conditions of a seed share one generated piece and identically seeded
rng streams. Greedy evals (learning OFF) run on all four passages at every
checkpoint for every condition — identical protocol, zero training budget.

Usage:
    python experiments/exp3_variation.py --seeds 40 --out results/exp3.json
                                         [--smoke] [--jobs 20]
                                         [--seed-offset 100]
"""

import argparse
import functools
import itertools
import time

import numpy as np
from scipy.stats import ttest_ind

from devrl.agents.linearq import LinearQ
from devrl.envs.piano import PianoPiece
from devrl.run import run_seeds, save_json
from devrl.stats import bootstrap_ci, iqm, mann_whitney

# lr sits at the stable edge for linear TD here (0.5+ diverges). The
# 450-episode budget is the v1 convention kept for comparability; the v1
# comment claimed >=600 episodes washes out the acquisition gap, but
# adversarial verification found the gap replicates (and is stronger) at
# 600 episodes on fresh seeds — the budget choice is not load-bearing.
LR = 0.4
GAMMA = 0.9
EPS = 0.1
EVAL_EPISODES = 5  # greedy rollouts per passage per checkpoint
ALPHA = 0.05
MAIN_CONDITIONS = ("blocked", "interleaved")
CONTROL_CONDITIONS = ("blocked-nomotif", "interleaved-nomotif")
CONDITIONS = MAIN_CONDITIONS + CONTROL_CONDITIONS
PASSAGE_NAMES = ("A", "B", "C", "novel")
BLOCK_ORDERS = tuple(itertools.permutations(range(3)))  # seed % 6 selects
CHANCE = 1.0 / PianoPiece.n_keys  # greedy over all-zero Q = uniform = 1/8
CHANCE_TOL = 0.05  # registered band for "nomotif transfer sits at chance"

HYPOTHESIS = (
    "H3 (variation practice): blocked practice looks better during "
    "acquisition but interleaved practice wins retention and transfer, "
    "because drilling one passage at a time lets each new passage "
    "overwrite the shared feature slots (motif and position onehots) that "
    "the other passages rely on (contextual interference; the Shea & "
    "Morgan 1979 crossover, in silico). Transfer specifically isolates "
    "the motif channel. Mechanism control: with passage-local features "
    "(no shared slots) the retention/transfer crossover should vanish and "
    "transfer should collapse to chance."
)


def _cond_parts(condition):
    """(schedule kind, feature map) for a condition name."""
    kind, _, suffix = condition.partition("-")
    return kind, ("local" if suffix == "nomotif" else "motif")


def make_schedule(kind, episodes, rng, order=(0, 1, 2)):
    """Passage index per training episode; equal budgets by construction.

    `order` (v2 counterbalancing) is the blocked phase order; interleaved
    sampling is uniform and ignores it.
    """
    if kind == "blocked":
        if episodes % 3:
            raise ValueError("blocked schedule needs episodes divisible by 3")
        return np.repeat(np.array(order), episodes // 3)
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


def _run_condition(env, kind, episodes, eval_every, children, order):
    """Train one condition; children = (agent, schedule, eval) seed children."""
    agent_rng, sched_rng, eval_rng = (np.random.default_rng(c) for c in children)
    agent = LinearQ(env.n_features, env.n_actions, lr=LR, gamma=GAMMA,
                    eps=EPS, rng=agent_rng)
    schedule = make_schedule(kind, episodes, sched_rng, order=order)
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
    # blocked, the phase of the episode before the checkpoint (order-aware
    # via the schedule); for interleaved, all three passages at once. This
    # is the fool's-progress curve — it flatters blocked practice.
    acquisition = []
    for i, ep in enumerate(checkpoints):
        if kind == "blocked":
            acquisition.append(evals[i][int(schedule[max(ep - 1, 0)])])
        else:
            acquisition.append(float(np.mean(evals[i][:3])))
    practice = [float(np.mean(train_scores[checkpoints[i]:checkpoints[i + 1]]))
                for i in range(len(checkpoints) - 1)]
    final = evals[-1]
    # order-relative retention (v2): `last` = passage drilled last under
    # this seed's block order, `earlier` = mean of the other two. The same
    # per-seed order is applied to every condition so the metrics compare
    # like with like across conditions.
    retention = {"A": final[0], "B": final[1], "C": final[2],
                 "mean": float(np.mean(final[:3])),
                 "last": final[order[2]],
                 "earlier": float(np.mean([final[order[0]], final[order[1]]]))}
    rollouts = {name: greedy_rollout(env, agent, p, eval_rng)[0]
                for p, name in enumerate(PASSAGE_NAMES)}
    return {
        "train_steps": train_steps,
        "checkpoint_episodes": checkpoints,
        "eval_scores": [[float(v) for v in row] for row in evals],
        "acquisition_curve": [float(v) for v in acquisition],
        "acquisition_mean": float(np.mean(acquisition[1:])),  # skip pre-train
        "practice_curve": practice,
        "retention": retention,
        "transfer": final[3],
        "final_rollouts": rollouts,
    }


def run_seed(seed, episodes, eval_every):
    """All four conditions on one shared piece, from seed-derived streams.

    v2: the blocked phase order is counterbalanced by seed % 6 over the six
    permutations of (A, B, C); the nomotif control pair runs the identical
    protocol on the same piece with feature_map="local" (no shared slots).
    """
    if episodes % eval_every:
        raise ValueError("episodes must be divisible by eval_every")
    order = BLOCK_ORDERS[seed % len(BLOCK_ORDERS)]
    children = np.random.SeedSequence(seed).spawn(4)  # env, agent, sched, eval
    # same env child seeds both feature maps -> bit-identical piece structure
    envs = {fm: PianoPiece(rng=np.random.default_rng(children[0]),
                           feature_map=fm) for fm in ("motif", "local")}
    out = {"seed": seed, "block_order": [int(p) for p in order],
           "structure": envs["motif"].structure()}
    for cond in CONDITIONS:
        kind, fm = _cond_parts(cond)
        # identically seeded fresh streams per condition: matched randomness
        out[cond] = _run_condition(envs[fm], kind, episodes, eval_every,
                                   children[1:4], order)
    return out


def _seed_job(i, offset, episodes, eval_every):
    """Pool worker: index i runs true seed i + offset (--seed-offset)."""
    return run_seed(i + offset, episodes=episodes, eval_every=eval_every)


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, input order preserved."""
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def _mw(a, b):
    """Mann-Whitney, tolerant of fully tied samples (p=1 by convention,
    tagged `degenerate` so the JSON marks conventional values)."""
    if np.ptp(np.concatenate([np.asarray(a, float), np.asarray(b, float)])) == 0:
        return {"u": len(a) * len(b) / 2.0, "p": 1.0, "degenerate": True}
    return mann_whitney(a, b)


def welch(a, b):
    """Welch's t (scipy ttest_ind, equal_var=False), degenerate-guarded.

    Fully tied across both groups -> t=0, p=1 by convention (mirrors _mw).
    Two distinct constants (zero within-group variance) -> the statistic is
    formally infinite; reported as the bare sign (+-1.0) with p=0 so the
    JSON stays finite. Both cases carry `degenerate: True`.
    """
    af, bf = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(np.concatenate([af, bf])) == 0:
        return {"t": 0.0, "p": 1.0, "degenerate": True}
    if np.ptp(af) == 0 and np.ptp(bf) == 0:
        return {"t": float(np.sign(af.mean() - bf.mean())), "p": 0.0,
                "degenerate": True}
    t, p = ttest_ind(af, bf, equal_var=False)
    return {"t": float(t), "p": float(p)}


def claim_verdict(test):
    """Registered v2 rule for a directional primary claim.

    significant = significant_mw AND significant_welch (both Holm-adjusted
    p-values below alpha — the conjunction is deliberately the harder
    criterion). Direction as predicted and significant -> supported;
    direction as predicted but not significant -> boundary (the registered
    reading of a small real effect that is alpha-marginal at this n, per
    the disclosed acquisition fragility); direction reversed and
    significant -> refuted; otherwise -> null.
    """
    significant = test["significant_mw"] and test["significant_welch"]
    if test["direction_ok"]:
        return "supported" if significant else "boundary"
    return "refuted" if significant else "null"


def mechanism_verdict(mech):
    """Registered v2 rule for the mechanism-control claim.

    The prediction is a VANISHING act, so raw (uncorrected) p-values are
    used — the harder criterion for a no-gap claim. supported: neither the
    nomotif retention gap nor the nomotif transfer gap is detectable by
    EITHER test (all raw p >= alpha) AND nomotif transfer sits at chance
    (within CHANCE_TOL) for both conditions. refuted: a doubly significant
    (MW and Welch) gap in the main-pair direction (interleaved > blocked)
    — the crossover survived without shared slots. boundary: anything else
    (one-test-only gap, reversed gap, off-chance transfer).
    """
    ret = mech["nomotif_retention_gap"]
    tra = mech["nomotif_transfer_gap"]

    def doubly_sig(t):
        return t["p"] < ALPHA and t["p_welch"] < ALPHA

    def any_sig(t):
        return t["p"] < ALPHA or t["p_welch"] < ALPHA

    def interleaved_wins(t):
        return t["iqm"]["interleaved-nomotif"] > t["iqm"]["blocked-nomotif"]

    if any(doubly_sig(t) and interleaved_wins(t) for t in (ret, tra)):
        return "refuted"
    at_chance = all(abs(v - CHANCE) <= CHANCE_TOL
                    for v in tra["iqm"].values())
    if not any_sig(ret) and not any_sig(tra) and at_chance:
        return "supported"
    return "boundary"


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
            seeds.append({"seed": r["seed"], "block_order": r["block_order"],
                          **{k: c[k] for k in
                             ("train_steps", "retention", "transfer",
                              "acquisition_mean", "acquisition_curve",
                              "eval_scores", "practice_curve",
                              "final_rollouts")}})
        conditions[cond] = {"n_seeds": len(results), "seeds": seeds}

    # ---- curves on the shared checkpoint grid
    def eval_col(cond, j):
        return [[row[j] for row in r[cond]["eval_scores"]] for r in results]

    def order_col(cond, which):
        """Order-relative retention series (per-seed block order)."""
        out = []
        for r in results:
            o = r["block_order"]
            if which == "last":
                out.append([row[o[2]] for row in r[cond]["eval_scores"]])
            else:
                out.append([(row[o[0]] + row[o[1]]) / 2.0
                            for row in r[cond]["eval_scores"]])
        return out

    metrics = {}
    for cond in CONDITIONS:
        series = {
            "acquisition": [r[cond]["acquisition_curve"] for r in results],
            "retention_A": eval_col(cond, 0),
            "retention_B": eval_col(cond, 1),
            "retention_C": eval_col(cond, 2),
            "retention_last": order_col(cond, "last"),
            "retention_earlier": order_col(cond, "earlier"),
            "transfer": eval_col(cond, 3),
        }
        series["retention_mean"] = (
            np.mean([series[f"retention_{p}"] for p in "ABC"], axis=0))
        for name, mat in series.items():
            metrics.setdefault(name, {})[cond] = _ci_series(mat, rng, n_boot)
    curves = {"episodes": ck, "steps": steps, "metrics": metrics}

    # ---- primary comparisons (main pair; Holm within family, both stats)
    primaries = [
        ("acquisition", "acquisition_mean", "blocked", "interleaved"),
        ("retention", "retention_mean", "interleaved", "blocked"),
        ("transfer", "transfer", "interleaved", "blocked"),
    ]
    tests, keys = {}, []
    for label, metric, win, lose in primaries:
        a, b = _metric(results, win, metric), _metric(results, lose, metric)
        mw, w = _mw(a, b), welch(a, b)
        key = f"{label}_{win}_gt_{lose}"
        keys.append(key)
        tests[key] = {"family": "primary", "metric": metric,
                      "predicted_winner": win,
                      "iqm": {win: iqm(a), lose: iqm(b)},
                      "direction_ok": bool(iqm(a) > iqm(b)),
                      "u": mw["u"], "p": mw["p"],
                      "t": w["t"], "p_welch": w["p"]}
        if mw.get("degenerate") or w.get("degenerate"):
            tests[key]["degenerate"] = True
    for key, pm, pw in zip(keys,
                           holm([tests[k]["p"] for k in keys]),
                           holm([tests[k]["p_welch"] for k in keys])):
        t = tests[key]
        t["p_holm"], t["p_holm_welch"] = pm, pw
        t["significant_mw"] = bool(pm < ALPHA)
        t["significant_welch"] = bool(pw < ALPHA)
        t["significant"] = bool(t["significant_mw"] and t["significant_welch"])

    # ---- order-relative secondaries (v2: replace fixed per-passage tests)
    secondaries = [
        ("retention_last_blocked_gt_interleaved", "retention_last",
         "blocked", "interleaved",
         "just-drilled recency: DESIGN predicts blocked keeps the passage "
         "it drilled last — this is the anticipated mechanism, not a "
         "failed interleaving prediction"),
        ("retention_earlier_interleaved_gt_blocked", "retention_earlier",
         "interleaved", "blocked",
         "the two earlier-drilled passages are where blocked's "
         "overwriting cost lives (order-relative form of the v1 "
         "per-passage secondaries)"),
    ]
    for key, metric, win, lose, note in secondaries:
        a, b = _metric(results, win, metric), _metric(results, lose, metric)
        mw, w = _mw(a, b), welch(a, b)
        tests[key] = {"family": "secondary", "metric": metric,
                      "predicted_winner": win, "note": note,
                      "iqm": {win: iqm(a), lose: iqm(b)},
                      "direction_ok": bool(iqm(a) > iqm(b)),
                      "u": mw["u"], "p": mw["p"],
                      "t": w["t"], "p_welch": w["p"],
                      "significant_uncorrected": bool(mw["p"] < ALPHA
                                                      and w["p"] < ALPHA)}

    # ---- mechanism-control comparisons (nomotif pair; raw p by design)
    mech_specs = [
        ("nomotif_acquisition_gap", "acquisition_mean", False,
         "descriptive only (not verdict-gating): blocked's concentration "
         "advantage on currently-practiced material need not require "
         "shared slots"),
        ("nomotif_retention_gap", "retention_mean", True,
         "no detectable gap by either test (raw p): without shared slots "
         "the interference that drives the retention half of the "
         "crossover is impossible"),
        ("nomotif_transfer_gap", "transfer", True,
         "no detectable gap by either test (raw p), and both conditions "
         "at chance (1/8 within 0.05): the novel passage's local features "
         "are never trained"),
    ]
    for key, metric, gates, prediction in mech_specs:
        a = _metric(results, "blocked-nomotif", metric)
        b = _metric(results, "interleaved-nomotif", metric)
        mw, w = _mw(a, b), welch(a, b)
        tests[key] = {"family": "mechanism", "metric": metric,
                      "prediction": prediction, "gates_verdict": gates,
                      "iqm": {"blocked-nomotif": iqm(a),
                              "interleaved-nomotif": iqm(b)},
                      "u": mw["u"], "p": mw["p"],
                      "t": w["t"], "p_welch": w["p"],
                      "gap_detected_uncorrected": bool(mw["p"] < ALPHA
                                                       or w["p"] < ALPHA)}
        if mw.get("degenerate") or w.get("degenerate"):
            tests[key]["degenerate"] = True

    # ---- per-claim verdicts (v2 contract)
    def _pv(t):
        return (f"MW p={t['p']:.3g} (Holm {t['p_holm']:.3g}), "
                f"Welch p={t['p_welch']:.3g} (Holm {t['p_holm_welch']:.3g})")

    def _iqms(t):
        win = t["predicted_winner"]
        lose = [k for k in t["iqm"] if k != win][0]
        rel = ">" if t["direction_ok"] else "<="
        return f"{win} IQM {t['iqm'][win]:.3f} {rel} {lose} {t['iqm'][lose]:.3f}"

    acq, ret, tra = (tests[k] for k in keys)
    mech_v = mechanism_verdict(tests)
    claims = [
        {"name": "acquisition",
         "claim": "Blocked practice looks better during acquisition "
                  "(greedy score on currently-practiced material) — the "
                  "fool's-progress half of the crossover.",
         "verdict": claim_verdict(acq),
         "evidence": f"{_iqms(acq)}; {_pv(acq)}. Fragility disclosed at "
                     "registration: the effect is small (~0.03 IQM) and "
                     "alpha-marginal at n=40 (direction held in 13/13 "
                     "fresh 40-seed verification batches of the v1 "
                     "protocol; a fresh batch missed alpha)."},
        {"name": "retention",
         "claim": "Interleaved practice wins retention (mean greedy score "
                  "on A, B, C after training).",
         "verdict": claim_verdict(ret),
         "evidence": f"{_iqms(ret)}; {_pv(ret)}."},
        {"name": "transfer",
         "claim": "Interleaved practice wins transfer to a novel passage "
                  "built from the same motifs.",
         "verdict": claim_verdict(tra),
         "evidence": f"{_iqms(tra)}; {_pv(tra)}."},
        {"name": "mechanism",
         "claim": "The retention/transfer crossover lives in the shared "
                  "feature slots: with passage-local features (no shared "
                  "slots) the gaps vanish and transfer collapses to "
                  "chance.",
         "verdict": mech_v,
         "evidence": (
             "nomotif retention gap: blocked-nomotif IQM "
             f"{tests['nomotif_retention_gap']['iqm']['blocked-nomotif']:.3f}"
             " vs interleaved-nomotif "
             f"{tests['nomotif_retention_gap']['iqm']['interleaved-nomotif']:.3f}"
             f" (MW p={tests['nomotif_retention_gap']['p']:.3g}, Welch "
             f"p={tests['nomotif_retention_gap']['p_welch']:.3g}); "
             "nomotif transfer gap: "
             f"{tests['nomotif_transfer_gap']['iqm']['blocked-nomotif']:.3f}"
             " vs "
             f"{tests['nomotif_transfer_gap']['iqm']['interleaved-nomotif']:.3f}"
             f" (MW p={tests['nomotif_transfer_gap']['p']:.3g}, Welch "
             f"p={tests['nomotif_transfer_gap']['p_welch']:.3g}); chance "
             f"= {CHANCE:.3f} +- {CHANCE_TOL}.")},
    ]
    verdict_of = {c["name"]: c["verdict"] for c in claims}
    summary = (
        f"Counterbalanced main pair — acquisition: {_iqms(acq)} "
        f"[{verdict_of['acquisition']}]; retention: {_iqms(ret)} "
        f"[{verdict_of['retention']}]; transfer: {_iqms(tra)} "
        f"[{verdict_of['transfer']}]. Mechanism control (no shared "
        f"slots): [{verdict_of['mechanism']}] — nomotif transfer IQMs "
        f"{tests['nomotif_transfer_gap']['iqm']['blocked-nomotif']:.3f}/"
        f"{tests['nomotif_transfer_gap']['iqm']['interleaved-nomotif']:.3f}"
        f" vs chance {CHANCE:.3f}.")

    # ---- per-condition scalar summaries for bars / crossover chart
    scalar_metrics = ("acquisition_mean", "retention_mean", "transfer",
                      "retention_A", "retention_B", "retention_C",
                      "retention_last", "retention_earlier")
    scalars = {cond: {m: _ci(_metric(results, cond, m), rng, n_boot)
                      for m in scalar_metrics}
               for cond in CONDITIONS}
    ex = results[0]
    correct = {name: ex["structure"]["correct_keys"][p]
               for p, name in enumerate(PASSAGE_NAMES)}
    viz = {
        "story": ("Blocked drilling of one passage at a time rides the "
                  "shared feature slots (motif and position onehots) to "
                  "fast visible progress, then each new passage "
                  "overwrites them; interleaving forces the shared "
                  "weights to serve all passages at once — worse "
                  "practice scores, better musician. The nomotif control "
                  "pair repeats the protocol with passage-local features "
                  "and no shared slots: the retention/transfer gaps "
                  "should vanish and transfer collapse to chance, "
                  "pinning the crossover on shared-parameter "
                  "interference. Transfer isolates the motif channel "
                  "specifically (the novel passage's own slots are never "
                  "trained). Block order is counterbalanced across seeds "
                  "(seed % 6), so retention is also reported relative to "
                  "drill order (earlier vs last)."),
        "n_keys": PianoPiece.n_keys, "n_motifs": PianoPiece.n_motifs,
        "motif_len": PianoPiece.motif_len,
        "passage_len": PianoPiece.passage_len,
        "passage_names": list(PASSAGE_NAMES),
        "conditions": list(CONDITIONS),
        "main_conditions": list(MAIN_CONDITIONS),
        "control_conditions": list(CONTROL_CONDITIONS),
        "example_structure": {"seed": ex["seed"], **ex["structure"]},
        "block_orders": [{"seed": r["seed"], "order": r["block_order"]}
                         for r in results],
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
        "order_relative_bars": {
            "labels": ["earlier", "last"],
            **{cond: {"iqm": [scalars[cond]["retention_earlier"]["iqm"],
                              scalars[cond]["retention_last"]["iqm"]],
                      "lo": [scalars[cond]["retention_earlier"]["lo"],
                             scalars[cond]["retention_last"]["lo"]],
                      "hi": [scalars[cond]["retention_earlier"]["hi"],
                             scalars[cond]["retention_last"]["hi"]]}
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
            "conclusion": {"claims": claims, "summary": summary},
            "viz": viz}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=None,
                    help="number of seeds (default 40, smoke 2)")
    ap.add_argument("--out", default="results/exp3.json")
    ap.add_argument("--smoke", action="store_true",
                    help="2 seeds, ~10x reduced budget")
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--seed-offset", type=int, default=100,
                    help="first true seed; confirmatory runs use "
                         "100..(100+N-1), disjoint from every seed used "
                         "for tuning (0-39)")
    args = ap.parse_args()
    episodes, eval_every, n_boot = ((45, 5, 2000) if args.smoke
                                    else (450, 15, 10000))
    n_seeds = args.seeds if args.seeds is not None else (2 if args.smoke else 40)
    config = {"seeds": n_seeds, "seed_offset": args.seed_offset,
              "seed_list": list(range(args.seed_offset,
                                      args.seed_offset + n_seeds)),
              "episodes": episodes,
              "steps_per_episode": PianoPiece.passage_len,
              "eval_every_episodes": eval_every,
              "eval_episodes_per_passage": EVAL_EPISODES,
              "lr": LR, "gamma": GAMMA, "eps": EPS,
              "conditions": list(CONDITIONS),
              "block_orders": [list(o) for o in BLOCK_ORDERS],
              "n_boot": n_boot, "alpha": ALPHA, "smoke": args.smoke}

    t0 = time.time()
    fn = functools.partial(_seed_job, offset=args.seed_offset,
                           episodes=episodes, eval_every=eval_every)
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
