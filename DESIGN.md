# Experimental Design

Five falsifiable hypotheses about developmental scaffolds for RL. All
experiments: tabular/linear numpy agents, many seeds, IQM + 95% bootstrap CI,
two-sided Mann-Whitney on primary comparisons, budgets strictly matched across
conditions (every step an agent takes — drill, practice, or play — counts).

Shared output contract — every experiment script:

```
python experiments/expN_name.py --seeds N --out results/expN.json [--smoke]
```

`--smoke`: 2 seeds, ~10x reduced budget, must finish < 60 s. JSON must contain:
`experiment`, `hypothesis`, `conditions` (per-seed raw metrics),
`curves` (checkpoint steps, IQM, CI bounds per condition), `tests`
(stat results for primary comparisons), `conclusion` ({supported, summary}),
and a `viz` block with everything the website needs (layouts, example
trajectories, per-generation advice sets, body sizes, etc.).

---

## EXP1 — The Blindfold Test (H1: world models)

**Claim.** An agent that learned a world model of its home can keep acting when
observation is removed (dead reckoning through the model), but the same
machinery fails in a stranger's home. Model-based learning (Dyna) also
dominates model-free on sample efficiency.

**Setup.** `GridHome` HOME_A / HOME_B: same 13x11 shell, same bed (S) and
fridge (G) coordinates, different floor plans. slip=0.1, gamma=0.97, step cap
60 (~3x shortest path). Train Q-learning vs DynaQ(planning=20) in HOME_A for
40k steps; eval greedy success every 1k steps (sample-efficiency curves).

**Blindfold protocol.** Using the trained DynaQ's model+Q from HOME_A:

| condition | home | observation |
|---|---|---|
| sighted-A | A | full state |
| blind-A | A | none — `BlindNavigator` dead reckoning |
| blind-A-touch | A | bump signal only (belief filtered by learned bump model) |
| blind-B | B | none, using A's model+Q |
| blind-B-touch | B | bump only, using A's model+Q |
| sighted-B-transfer | B | full state, A's Q (control: layout knowledge, not vision, is the bottleneck) |
| random-A | A | none, random policy (chance floor) |

Touch model: bump counts per (s,a) collected during training (new module, do
not modify dyna.py); belief update multiplies by P(bump-observation | s,a).
30 seeds. Metrics: success rate, steps-to-goal. Export home layouts + 3 example
blind trajectories (position sequence + belief entropy per step) per condition.

**Predictions.** blind-A close to sighted-A (touch closes most of the slip-noise
gap); blind-B at or near random-A; Dyna reaches 90% success in fewer steps
than Q-learning.

---

## EXP2 — Microtasks (H2: drills beat whole-game practice)

**Claim.** A reverse curriculum over start states (shoot from near goal ->
dribble from midfield -> full game) reaches full-game mastery in fewer total
env steps than playing full games from the start. No synthetic rewards — only
the start-state distribution changes, so there is no shaping contamination.

**Setup.** `SoccerGrid`: 11x7 pitch, goal = middle 3 cells of right edge.
State: (agent cell, ball state) where ball is carried, at a cell, or scored
(~6k states). Actions: 4 moves + shoot (only useful when carrying). Pickup by
walking onto ball. Shot scores with p = max(0, 1 - d/5), d = Chebyshev
distance to goal center; a miss ends the episode. Reward 1 on goal only.
Cap 100 steps. Budget 60k steps, eval (20 greedy episodes from kickoff) every
2k steps.

**Conditions.**
- `whole`: all 60k steps on full games from kickoff.
- `drills-varied`: 20% shoot drill (spawn carrying, random attacking-third
  cell), 20% dribble drill (spawn random left-half cell, ball at center),
  60% full games. One shared Q-table throughout.
- `drills-fixed`: same phases but each drill uses ONE fixed spawn cell
  (blocked practice ablation — links to H3).

**Metrics.** Time to 90% eval success (censored runs reported), final success,
30 seeds. Export field layout + one greedy trajectory per condition at 25%,
50%, 100% of budget.

**Predictions.** drills-varied < whole on time-to-threshold; drills-fixed
between or worse.

---

## EXP3 — Variation Practice (H3: contextual interference)

**Claim.** Blocked practice (drill one passage at a time) looks better during
acquisition but loses on retention and transfer; interleaved/varied practice
looks worse during training and wins at test — the Shea & Morgan (1979)
crossover, in silico. Mechanism: shared-parameter interference.

**Setup.** `PianoPiece`: 3 passages, each 12 positions long, built from a
shared library of 6 motifs (each 3 notes). Correct key at a position depends
on (motif, position-in-motif) via a motif table, with exactly one per-passage
exception position (context-dependent fingering). 8 keys. Episode = play one
passage; +1 per correct note (episode continues on error); score = fraction
correct.

**Agent.** Linear Q with SGD (`linearq.py`): Q(s,a) = w_a . phi(s),
phi = onehot(motif, pos-in-motif) ++ onehot(passage) ++ onehot(position).
Shared motif features are where interference lives.

**Conditions** (equal total episodes, 40 seeds):
- `blocked`: passage A for 1/3 of budget, then B, then C.
- `interleaved`: passage sampled uniformly each episode.

**Test phase** (no learning): retention = mean score on A, B, C; transfer =
score on a NOVEL passage (new arrangement of the same motifs, own exception
disabled). Also record the acquisition curve (score on the currently
practiced passage) — this is the "fool's progress" curve.

**Predictions.** blocked acquisition curve above interleaved during training;
interleaved wins retention (especially passages A, B after C was drilled) and
transfer. Export acquisition curves + retention/transfer bars + motif
structure for viz.

---

## EXP4 — Generational Teaching (H4: iterated distillation)

**Claim.** A lineage of short-lived agents, each distilling its best
experience to a fresh, plastic student through a narrow bottleneck, ratchets
past (a) weight-copy transfer and (b) one agent living the combined lifetime
with decaying plasticity. Mechanism: the bottleneck transmits the rare peak
experience ("the one time I reached the mountain") without transmitting
consolidated mediocrity or rigid habits.

**Setup.** `TrapGrid` 15x11: start left; big reward 10 at far right; 3 candy
cells (reward 0.3, terminal) near the start — locally optimal traps that end
episodes early and throttle deep exploration. Deterministic moves, cap 120
steps, gamma 0.99, Q init 0.

**Lifetime & aging.** One life = 15k steps with plasticity decay:
lr(age) = 0.3 * 2^(-age/5k), eps(age) = 0.4 * 2^(-age/5k) — old agents are
rigid. Each agent also keeps an episodic memory of its top-3 episodes by
return.

**Advice bottleneck.** At death, teacher emits advice = deduplicated (s, a)
pairs from its top-3 episodes, capped at 100 pairs. Student (fresh table,
fresh schedules) is pretrained: Q[s, a_advice] = 5.0, then lives normally.

**Conditions** (5 generations x 15k = 75k total steps each, 30 seeds):
- `generational-distill`: as above.
- `weight-copy`: student starts from a full copy of teacher's Q (fresh
  schedules) — transfer learning.
- `one-long-life`: single agent, 75k steps, same decay constants.
- `one-long-life-slow`: single agent, decay halflife 25k (tuned fair baseline).
- `no-inheritance`: 5 independent lives; report per-generation (flat line
  expected — proves inheritance is causal).

**Metrics.** After each generation: greedy return, P(big goal), and the gap
between best-remembered episode and greedy performance (unconsolidated
knowledge). Export per-generation curves + the actual advice trajectories.

**Predictions.** generational-distill ratchets upward across generations and
ends highest; weight-copy plateaus (inherits trap-shaped values); both
long-life conditions stall when plasticity dies.

---

## EXP5 — Growing Bodies (H5: morphological curriculum)

**Claim.** Training a small body first (falls are cheap, relative strength is
high) and growing toward the adult body reaches adult competence with far
less cumulative damage — and balance-first beats walking-from-the-start.

**Setup.** `BalanceBot`: torque-limited inverted pendulum with body scale s.
l = s, m = 15 s^3, tau_max = 40 s^2 (square-cube law: small = relatively
stronger, but faster dynamics per fixed control step — not rigged in either
direction). Fall at |theta| > pi/5 ends episode with damage cost
(s / s_adult)^4 (impact energy ~ m g l). Torque noise 5% of tau_max.
State: (theta 17 bins, theta-dot 15 bins, current target in {-0.15, 0, +0.15})
= 765 states; 5 torque actions. "Walking" = tracking a lean target that
switches every 40 steps (weight transfer); +1/step upright, +2/step on
target. Cap 400 steps. Budget 120k steps; eval on the ADULT body (greedy,
target-switching) every 4k steps for all conditions.

**Conditions** (20 seeds):
- `adult-walk`: s=1.0 throughout, walking task from the start.
- `adult-balance-first`: s=1.0; targets fixed at 0 for first 30%, then walking.
- `grow-linear`: s from 0.5 to 1.0 over first 60% of budget; balance-first.
- `grow-adaptive`: grow by 0.05 whenever rolling no-fall rate > 70%
  ("grow when ready"); balance-first.
- `grow-jump`: s=0.5 for 60% then s=1.0 (tests gradualism); balance-first.
- `grow-linear-walk`: grow-linear but walking task from the start
  (isolates the balance-first factor).

**Metrics.** Steps to adult competence (eval return threshold), cumulative
damage at that point (headline), final adult performance. Export body-size
schedules, fall events (step, size, damage), and 200-step theta traces of the
adult greedy policy per condition.

**Predictions.** grow-adaptive & grow-linear reach adult competence with a
fraction of adult-direct's damage (and no slower in steps); grow-jump worse
than grow-linear (gradualism matters); balance-first beats walk-direct within
matched morphology conditions.

---

## Statistical protocol (all experiments)

- Point estimate: IQM across seeds; uncertainty: 95% percentile bootstrap
  (10k resamples).
- Primary comparisons: two-sided Mann-Whitney; Holm-Bonferroni within an
  experiment's primary family. Alpha 0.05.
- Time-to-threshold with censoring: censored seeds reported explicitly and
  assigned budget+1 for rank tests (conservative); censoring fraction always
  shown.
- Every condition sees identical eval protocols; eval steps are excluded from
  training budgets identically across conditions.

---

## Amendments (v2) — EXP2

Adversarial verification (logic lens) confirmed that the drills-vs-whole
effect REVERSES under optimistic initialization (Q0 = 1.0): an optimistic
whole-game learner reaches threshold about as fast as the shipped
drills-varied arm, and the drill curriculum actively delays an optimistic
explorer. This is a genuine boundary condition the literature predicts —
the value of guide policies / start-state curricula is capped for
optimistic explorers (Xie et al. 2021; JSRL) — so the claim must be scoped,
not merely restated. v2 replaces the single boolean conclusion with
per-claim verdicts and adds the two conditions needed to test the boundary
and the mechanism.

**New conditions** (training budget still exactly 60k steps each; eval
protocol, hyperparameters, and env unchanged):

- `whole-optimistic`: identical to `whole` (all steps full games from
  kickoff) except the Q-table is initialized at Q0 = 1.0 — a principled
  upper bound, since the only reward is a single terminal 1 and gamma < 1.
  All other arms keep Q0 = 0.
- `explore-starts`: the classic exploring-starts alternative. For the first
  40% of the budget (exactly the drill fraction, 24k steps) each episode
  starts from a uniformly random non-terminal state: agent uniform over the
  77 cells, ball uniform over the 78 non-terminal ball options (77 cells +
  carried; spawning on the ball's cell picks it up). Remaining 60%: full
  games from kickoff. Q0 = 0. Start-state diversity without drill structure.

**Registered primary family** (Holm within the family, m = 5; metric
time-to-90% with censored seeds at budget+1, as before):

1. drills-varied vs whole            (claim 1 headline — unchanged)
2. drills-fixed  vs whole            (unchanged)
3. drills-varied vs drills-fixed     (unchanged)
4. drills-varied vs whole-optimistic (claim 2)
5. drills-varied vs explore-starts   (claim 3)

Every primary comparison reports BOTH the two-sided Mann-Whitney U and
Welch's t (scipy `ttest_ind`, `equal_var=False`), each Holm-adjusted within
the 5-comparison family. Registered decisions ride on the Mann-Whitney Holm
p at alpha 0.05 (the originally registered test); Welch is reported
alongside because Colas et al. (2019) show Mann-Whitney's false-positive
rate inflates under unequal shapes/spreads. Descriptive context (no claim
rides on them, no Holm): whole-optimistic vs whole and explore-starts vs
whole on t90; final-success comparisons for the primary pairs.

**Per-claim conclusion** replaces `conclusion.supported`:
`conclusion = {claims: [{claim, verdict, evidence}], summary}`, verdict in
{supported, refuted, null, boundary}. Decision rules:

- Claim 1 — "drills beat whole-game practice for epsilon-greedy learners":
  supported iff comparison 1 is significant with drills-varied faster;
  refuted iff significant with whole faster; null otherwise.
- Claim 2 — "the drill benefit is an exploration effect: it disappears or
  reverses under optimistic initialization": decided on comparison 4.
  Significant with whole-optimistic faster -> boundary (reversal);
  non-significant -> boundary (the drill advantage disappears against an
  optimistic whole-game learner); significant with drills-varied still
  faster -> refuted (the benefit is not exploration-bound).
- Claim 3 — "drill structure adds benefit beyond mere start-state
  diversity": decided on comparison 5, reported whichever way it lands:
  significant with drills-varied faster -> supported; significant with
  explore-starts faster -> refuted; otherwise null.

**Fresh confirmatory seeds.** The v2 confirmatory run uses seeds 100-129
via a new `--seed-offset` flag (default 100) — disjoint from every seed
ever used for tuning, dev checks, or verification of this experiment
(0-29, 1000-1029, 3000-3019). Smoke mode keeps 2 seeds at 1/10 budget.

Reporting hygiene from the verification: Mann-Whitney/Welch entries that
hit a degenerate-sample guard (fully tied, or zero within-group variance)
are tagged `degenerate` in the JSON; the viz block carries the two new
conditions (phases, trajectories, eval curves) plus a per-condition `q0`
map so the site can annotate the optimistic arm.

---

## Amendments (v2) — EXP1

Registered changes mandated by the adversarial verification (3 lenses) of
the v1 run, written down before the v2 confirmatory run.

**A1 — Symmetric stranger-fails criterion (registered primary).** The v1
primary `blindA_beats_blindB` (pure dead-reckoning pair) passed on a
knife-edge (p_holm=0.0404; leave-one-out flips 27/60, paired Wilcoxon
p=0.061, Welch p=0.148), while the far stronger like-for-like evidence —
blind-A-touch vs blind-B-touch, the same machinery on both sides
(verifiers recomputed MW p=1.3e-9) — was registered nowhere. v2 registers
`blindA_touch_beats_blindB_touch` (predicted a > b) as a PRIMARY test and
judges the stranger-fails leg on this symmetric strong-variant pair. The
pure pair `blindA_beats_blindB` is kept as a secondary (predicted a > b).

**A2 — Matched random floor (new condition `random-B`).** v1 cited
random-A (IQM 0.00) as blind-B's floor, but HOME_B is intrinsically easier
(shortest path 11 vs 17) and blind-B is significantly above every floor —
the v1 "collapses to the random floor" wording overstated. v2 adds
`random-B`: a random policy in HOME_B under the identical episode protocol
and rng streams. Both floors are reported. The claim is reworded to: in
the stranger's home the same machinery performs **far below home
performance, modestly above a matched random floor**. New secondaries:
`blindB_vs_randomB_matched_floor` (predicted a > b: small residual
competence — shared start/goal coordinates — above the matched floor) and
`blindBtouch_vs_randomB_matched_floor` (predicted a > b). The v1
cross-home secondary `blindB_vs_randomA_chance_floor` is kept, with its
equivalence-style prediction now explicitly scored (A4).

**A3 — Update-matched replay baseline (new condition `replayq-A`).**
RESEARCH.md flags this as the central fairness risk (van Hasselt, Hessel &
Aslanides 2019): DynaQ(planning=20) does 21 Q-updates per env step vs
Q-learning's 1, so "Dyna is more sample-efficient" may be an update-count
artifact, not a world-model result. v2 adds `replayq-A`: tabular
Q-learning + uniform experience replay doing 20 replayed updates per real
step (drawn uniformly with replacement from its own transition buffer),
trained in HOME_A with the identical env-step budget (40k), identical
hyperparameters (lr, eps, gamma, optimistic_init) and its own rng stream —
update-for-update matched with DynaQ (1 real + 20 extra per env step).
Implemented as a new module `src/devrl/agents/replayq.py`; `dyna.py` and
`qlearning.py` are untouched. New registered PRIMARY tests:
`replay_faster_than_q_t90` (predicted a < b: update count alone should
reproduce most of the speedup) and `dyna_vs_replay_t90` (predicted a ~ b:
van Hasselt — in this near-deterministic tabular world the replay buffer
IS a non-parametric model, so no model-specific advantage is expected).

**A4 — Test reporting.** Every test entry now carries `p_welch` (Welch t,
scipy.stats.ttest_ind equal_var=False; degenerate zero-variance cases
guarded: fully tied -> p=1, two distinct constants -> p=0) alongside the
Mann-Whitney p. Holm is applied within the 5-test primary family to both
statistics separately (`p_holm` for MW — the registered decision statistic
— and `p_welch_holm`, reported for robustness). Every test also carries a
`prediction_met` field: for directional predictions, met iff the decision
p (p_holm for primaries, raw p for secondaries) is < 0.05 with the IQMs
ordered as predicted; for equivalence-style predictions ("a ~ b"), met iff
the RAW (uncorrected — deliberately the harder criterion) MW p >= 0.05,
documented as "no detectable difference", NOT a formal equivalence test.
`significant` keeps its v1 meaning (difference detected at the decision
level; for directional primaries it additionally requires the predicted
direction, as in v1).

**A5 — Fresh confirmatory seeds.** A `--seed-offset` flag (default 100) is
added; the v2 confirmatory run uses seeds 100-129 — disjoint from every
seed ever used for tuning or probing this experiment (0-29) and from the
verifiers' rerun (1000-1029). Budgets are unchanged and matched exactly
across all arms (every arm takes exactly 40k env steps; eval and blindfold
rollouts remain budget-free and identical across conditions).

**A6 — Per-claim verdicts replace the boolean conclusion.** The output
contract becomes `conclusion = {claims: [{claim, verdict, evidence}],
summary}` with verdict in {supported, refuted, null, boundary}. Registered
claims and decision rules (IQMs across seeds; "prediction_met" refers to
the tests above):

1. *dyna-beats-q*: DynaQ reaches 90% eval success in fewer env steps than
   sample-matched (update-unmatched) Q-learning. Supported iff
   `dyna_faster_than_q_t90` prediction_met; refuted iff significant in the
   opposite direction; else null.
2. *dyna-advantage-is-model-not-updates*: the Dyna speedup survives an
   update-matched replay baseline. Supported iff `dyna_vs_replay_t90` is
   significant (p_holm < 0.05) with Dyna faster; refuted iff significant
   with replay faster, OR not significant while replay closes >= 80% of
   Dyna's t90 advantage over plain Q-learning (closure = (q - replay) /
   (q - dyna) on IQM t90); null otherwise.
3. *pure-dead-reckoning-works-at-home* (DESIGN v1's literal blind-A
   prediction, now scored honestly): supported iff IQM(blind-A) >= 0.8 x
   IQM(sighted-A); refuted iff < 0.5 x; boundary between.
4. *touch-filtered-blind-works-at-home*: same rule on blind-A-touch.
5. *stranger-collapse* (symmetric touch pair): supported iff
   `blindA_touch_beats_blindB_touch` prediction_met AND
   IQM(blind-B-touch) <= 0.5 x IQM(blind-A-touch); boundary iff
   prediction_met with ratio in (0.5, 0.8]; refuted iff significant in the
   opposite direction or ratio > 0.8; null otherwise. Evidence must report
   both random floors (random-A and random-B).
6. *touch-helps-at-home*: supported iff `touch_beats_pure_deadreckoning`
   prediction_met; refuted iff significant in the opposite direction; else
   null.

The viz block gains the new conditions (`random-B` in the blindfold
summary and example trajectories; `replayq-A` in the training curves and
sample-efficiency block) and its predictions list is updated to the v2
registrations above.
