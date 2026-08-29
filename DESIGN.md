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

---

## Amendments (v2) — EXP3

Registered changes mandated by the adversarial verification (3 lenses) of
the v1 run, written down before the v2 confirmatory run.

**A1 — Counterbalanced block order.** v1 drilled A, B, C in a fixed order
for every seed, so blocked's retention deficit on A/B and advantage on C
were partly an order/recency artifact by construction (flagged in
RESEARCH.md and by all three lenses). v2 counterbalances: `seed % 6`
selects one of the 6 permutations of (A, B, C) as the blocked phase
order. Retention gains two order-relative metrics computed under each
seed's own order and applied identically to every condition:
`retention_last` (the passage drilled last) and `retention_earlier`
(mean of the other two). The v1 fixed per-passage secondaries
(retention_A/B/C, all labeled predicted_winner=interleaved — including
retention_C, which mislabeled the DESIGN-anticipated recency effect as a
failed prediction) are replaced by two order-relative secondaries:
`retention_last_blocked_gt_interleaved` (predicted winner BLOCKED —
just-drilled recency, the anticipated mechanism, with an explanatory
note) and `retention_earlier_interleaved_gt_blocked` (predicted winner
interleaved — where the overwriting cost lives). With 40 seeds, 40 = 6x6
+ 4, so four orders appear 7 times and two appear 6 times — disclosed,
and orthogonal to the conditions since every condition of a seed shares
the seed's order.

**A2 — Mechanism-control pair (new conditions `blocked-nomotif`,
`interleaved-nomotif`).** The logic lens confirmed a mechanism-attribution
confound: the v1 hypothesis credited "shared motif weights", but the
featurization has a second shared channel (the position onehot), and a
position-only ablation reproduced the acquisition/retention crossover.
The umbrella mechanism (shared-parameter interference) and the
motif-specificity of transfer both survived. v2 (a) rewords the
mechanism claim to "shared feature slots (motif and position onehots)",
noting transfer alone isolates the motif channel, and (b) adds the
registered mechanism control: the identical protocol run with
`PianoPiece(feature_map="local")` — phi = onehot((passage, position)),
one indicator per state, NO shared slots, a passage-local tabular
equivalent. Implemented as a feature-map option on the env (the agent
and protocol are unchanged), not as a copy of the agent. Same piece
structure per seed (same env seed child), same schedules, same budgets
(450 episodes x 12 steps per condition). Registered prediction: the
retention/transfer crossover VANISHES and nomotif transfer sits at
chance (1/8 +- 0.05), because the novel passage's local features are
never trained. The nomotif acquisition gap is reported descriptively but
does not gate the verdict (blocked's concentration advantage on
currently-practiced material need not require shared slots).
Mechanism-control tests use RAW (uncorrected) p-values — deliberately
the harder criterion for a vanishing prediction.

**A3 — Both test statistics everywhere.** Every registered test reports
the two-sided Mann-Whitney U (u, p) AND Welch's t (scipy
`stats.ttest_ind`, `equal_var=False`; t, p_welch), with degenerate
samples guarded (fully tied -> p=1; two distinct constants -> p=0 with t
reported as the bare sign; both tagged `degenerate`). Holm is applied
within the 3-test primary family to each statistic separately (`p_holm`,
`p_holm_welch`). A primary's `significant` requires BOTH Holm-adjusted
p-values below alpha — the conjunction is deliberately the harder
criterion. Secondary `significant_uncorrected` likewise requires both
raw p-values below alpha.

**A4 — Per-claim verdicts replace the boolean conclusion.** The output
contract becomes `conclusion = {claims: [{name, claim, verdict,
evidence}], summary}` with verdict in {supported, refuted, null,
boundary}. Registered decision rules for the three directional primaries
(acquisition: blocked > interleaved; retention_mean and transfer:
interleaved > blocked): direction as predicted AND significant (per A3
conjunction) -> supported; direction as predicted but not significant ->
BOUNDARY — registered in advance because verification showed the
acquisition effect is real but alpha-marginal at n=40 (direction held in
13/13 fresh 40-seed batches, pooled n=400 p=1.7e-28, yet one fresh batch
missed alpha at p=0.0998): a directionally-consistent miss at this n is
a power boundary, not a null; direction reversed and significant ->
refuted; otherwise null. Mechanism claim: supported iff neither the
nomotif retention gap nor the nomotif transfer gap is detectable by
EITHER test (all raw p >= alpha) and nomotif transfer is at chance for
both conditions; refuted iff a gap is doubly significant (MW and Welch)
in the main-pair direction (interleaved > blocked); boundary otherwise.

**A5 — Fresh confirmatory seeds.** A `--seed-offset` flag (default 100)
is added; the v2 confirmatory run uses seeds 100-139 — disjoint from
every seed ever used for tuning this experiment (0-39). (Seeds 100-139
appeared once in the verifiers' out-of-sample replication of the v1
protocol — where acquisition was directionally correct but missed alpha,
the disclosed fragility A4 scores honestly — and the v2 protocol draws
different rng streams in any case: counterbalanced orders and four
arms.) Budgets are unchanged and matched exactly across all four arms
(450 episodes x 12 steps = 5400 training steps per condition per seed;
greedy evals remain budget-free and identical across conditions).

**A6 — Comment correction (no behavior change).** The v1 code comment
justified the 450-episode budget as "before saturation washes out the
acquisition half"; verification found the acquisition gap replicates
(and is stronger) at 600 episodes on fresh seeds. The comment now states
the budget is a v1-comparability convention, not load-bearing.

The viz block gains the two nomotif conditions (curves, practice curves,
bars, crossover, rollouts, per-seed points), the per-seed block orders,
and `order_relative_bars` (earlier vs last) alongside the per-passage
retention bars.

---

## Amendments (v2) — EXP5

Registered changes mandated by the adversarial verification (3 lenses) of
the v1 run, written down before the v2 confirmatory run. The verification's
invalidating finding was about CLAIM STRUCTURE, not the damage effect: v1's
single `supported` boolean encoded "less damage AND no slower" while the
JSON's hypothesis string was a longer conjunction whose other parts the same
data refuted (balance-first significantly reversed, gradualism directionally
reversed); the "no slower" conjunct rested on accepting a null that failed
to replicate on fresh seeds (tuning seeds 0-9 overlapped confirmatory 0-19),
and the headline pairing (grow-linear vs adult-walk) confounded morphology
with the independently-refuted balance-first task staging.

**A1 — New condition `grow-adaptive-walk`** (the missing factorial cell):
the grow-adaptive size gate (grow s by 0.05 when the rolling 20-episode
no-fall rate exceeds 70%) with the walking task from the start. This
completes the growth {none, linear, adaptive} x staging {balance-first,
walk-direct} cells needed to isolate morphology from task staging. Training
budget exactly 120k steps, identical eval protocol, shared hyperparameters —
all unchanged.

**A2 — Primary family re-registered as the confound-free walk-direct
pairing** (Holm within family, m=4): grow-linear-walk vs adult-walk and
grow-adaptive-walk vs adult-walk, on damage-at-competence AND
steps-to-competence. Both sides walk from the start, so the pairing isolates
the morphological curriculum — v1's headline pairing (grow-linear vs
adult-walk) differed in BOTH morphology and staging, and the verification
showed its fresh-seed slowdown traces to the refuted balance-first factor
riding inside grow-linear. The full v1 family (4 damage + 4 steps
comparisons) is retained unchanged as a SECONDARY family (Holm within, m=8).

**A3 — Per-claim verdicts replace the boolean.**
`conclusion = {claims: [{key, claim, verdict, evidence}], summary}`,
verdict in {supported, refuted, null, boundary}. Registered claims and
decision rules (`directional_verdict`: both MW and Welch significant ->
supported/refuted by direction; exactly one -> boundary; neither -> null;
two-comparison claims combine per-comparison verdicts: supported+refuted ->
boundary, any refuted -> refuted, any boundary -> boundary, all supported ->
supported, supported+null -> boundary, all null -> null):

1. `damage_at_competence` — growth (walk-direct) reaches adult competence
   with less cumulative damage: directional_verdict on the two primary
   damage tests, combined. Evidence must carry the censoring-sensitivity
   checks (drop-censored and worst-case-rank MW p) and the standing-only
   guard (lean-target time at competence).
2. `steps_parity` — growth (walk-direct) is no slower: scored by
   `parity_verdict`, an equivalence view that NEVER accepts a null. With
   diff = IQM(growth) - IQM(adult) steps (censored = budget+1), its 95%
   bootstrap CI, and a pre-registered margin of +20% of the adult IQM:
   slowdown detected (either statistic, Holm) with CI upper bound beyond
   the margin -> refuted; detected but bounded within the margin ->
   boundary; no detected slowdown and CI upper bound within the margin ->
   supported (equivalence shown); otherwise null (underpowered — reported
   as such, not as "no effect").
3. `gradualism` — grow-linear < grow-jump on damage (secondary family),
   directional_verdict; the summary must state the DIRECTION of the point
   estimate (v1's "not significant" concealed a reversal).
4. `balance_first` — adult-balance-first < adult-walk and grow-linear <
   grow-linear-walk on steps (secondary family), directional_verdict per
   comparison, combined. A reversal is reported plainly as refuted.

**A4 — Both test statistics everywhere.** Every comparison reports the
two-sided Mann-Whitney U AND Welch's t (scipy `ttest_ind`,
`equal_var=False`; degenerate guard: fully tied -> p=1, two distinct
constants -> p=0), each Holm-adjusted within its family (`p_holm`,
`p_welch_holm`). Verdicts use both (see A3) per Colas et al. (2019):
Mann-Whitney miscalibrates under unequal shapes/spreads, the expected
regime here (bimodal solved/unsolved seeds).

**A5 — Physics-robustness arms** (reduced seed count, clearly labeled, NOT
confirmatory — they probe whether the damage ordering is an artifact of two
physics choices):

- `tau3`: tau_max = 40 s^3 — muscle torque ~ L^3 (force ~ L^2 x lever arm
  ~ L, per RESEARCH.md), replacing the v1 s^2 law whose relative authority
  ~ 1/s^2 over-assists small bodies relative to biology (~1/s).
- `damp2`: damping b = 1.0 s^2 — size-scaled damping. DISCLOSURE of a v1
  deviation: the env applied a constant damping coefficient b = 1.0 at
  every size, absent from DESIGN v1; relative damping b/(m l^2) =
  1/(15 s^5) is ~32x stronger for s=0.5 than for the adult, an undeclared
  stabilization aid to small bodies. The default is now documented in the
  env docstring; damp2 shrinks the advantage to ~s^-3. (Full scale
  invariance would need b ~ s^5 at fixed dt; s^2 is the physically motivated
  viscous-joint law and a strict reduction of the aid.)

Also disclosed from verification: falls yield a fixed -5 reward at every
size (uniform across conditions; damage is recorded separately and is the
metric). Now stated in the env docstring and viz.meta.reward.

Each variant runs the four growth conditions (grow-linear, grow-adaptive,
grow-linear-walk, grow-adaptive-walk) at 20 seeds, same seed offset, same
budget/eval. The adult-walk comparator is reused from the main arm
restricted to those seeds: both variant laws coincide with the default at
s=1.0 and rng streams are variant-independent, so a variant rerun of an
adult-only condition is bit-identical. Reported per variant: per-condition
damage, MW+Welch damage tests vs adult-walk (Holm, m=4), and the damage
ordering. No registered claim rides on these arms.

**A6 — Fresh confirmatory seeds.** A `--seed-offset` flag (default 100) is
added; the v2 confirmatory run uses seeds 100-139 (40 seeds — doubled from
v1's 20 because the verification's non-replication concerned the
underpowered steps conjunct) — disjoint from the v1 tuning seeds (0-9) and
the v1 confirmatory range (0-19). Verifier probes touched 20-39 and
100-119 read-only (no tuning decisions were derived from them); no
hyperparameter, threshold, or schedule was changed from v1, so seed 100+
data never informed any analysis choice. Robustness arms use the first 20
of the same range. Budgets stay matched exactly (every condition trains
exactly 120k env steps; eval excluded identically).

**A7 — Transience and reward-hacking guards.** Eval records time-on-target
fractions (overall, and restricted to nonzero lean targets — unearnable by
a pure stander), reported per seed at competence and as viz curves; a
durable-competence variant (two consecutive checkpoints >= 500,
`steps_to_durable`) is reported descriptively per condition alongside
first-crossing competence. Claims continue to ride on the registered
first-crossing metric.

The viz block gains the new condition everywhere, `ontarget` curves, a
`robustness` summary (damage orderings under each variant), and meta
disclosures for reward and damping.

---

## Amendments (v2) — EXP4

Registered changes mandated by the adversarial verification (3 lenses) of
the v1 run. The decision rules below were fixed in `tests/test_exp4.py`
(TDD) before the v2 confirmatory run was executed.

**Why.** (1) The v1 headline was fragile: 21/30 successes was exactly the
minimum clearing Holm at n=30, a same-protocol disjoint-seed rerun landed
19/30 (n.s.), and the EpisodicMemory tie-break (DESIGN-silent, empirically
calibrated: "earliest tied episode wins") flipped significance under the
rejected "shortest" variant. (2) RESEARCH.md (and the verifiers) flag the
central confound: Q=5.0 advice priming is optimism, so "advice content
helps" was confounded with "optimism at 100 entries helps" — no
optimism-matched control existed. (3) The smoke JSON silently concluded
the opposite of the full run. v2 addresses all three: more seeds on a
fresh range, both tie-breaks reported, content-vs-optimism controls added,
per-claim verdicts, and dual test statistics.

**New conditions** (training budget still exactly 5 x 15k = 75k env steps
each; env, hyperparameters, eval protocol, and the original five arms
unchanged — the originals keep their v1 rng stream indices, verified
bit-identical on overlapping seeds):

- `generational-distill-shortest`: identical to `generational-distill`
  except the episodic memory breaks return ties by SHORTEST episode
  (equal lengths: earlier wins) instead of earliest. `EpisodicMemory`
  gains a `tie_break` parameter ("earliest" remains the default and the
  primary arm); the v1 fragility to this choice is now reported, not
  hidden.
- `random-advice`: optimism-matched control. Every generation (gen 1
  included — the control always receives its full dose, which can only
  flatter it relative to distill's advice-free gen 1), the fresh student
  is primed Q[s, a] = 5.0 on 100 RANDOM distinct (s, a) pairs drawn from
  its condition rng over actable non-terminal cells (walls, candy, goal
  excluded — the agent never acts from those, so including them would
  dilute the dose). Same pair count, same blessing value, no content.
- `optimistic-init`: one long life (75k steps, standard halflife-5k
  decay), Q0 = 5.0 EVERYWHERE — maximal content-free optimism in the
  long-life format (the classic optimistic-initialization fix).
- `constant-eps-life`: one long life, eps fixed at 0.4 forever, lr
  decaying as usual (halflife 5k) — the rescue test for "plasticity decay
  is what strands the long life": if undying exploration does not
  un-strand the long life, exploration decay was not the binding
  constraint.

**Registered test families** (metric unchanged: final greedy return at
generation 5; Holm-Bonferroni within each family, applied separately to
each statistic). Every comparison reports BOTH two-sided Mann-Whitney U
and Welch's t (scipy `ttest_ind`, `equal_var=False`; degenerate guards:
fully tied -> p=1, two distinct constants -> p=0 with t=null).
Registered significance requires BOTH Holm-adjusted p values < 0.05.

Primary family (m=7): distill vs weight-copy / one-long-life /
one-long-life-slow / no-inheritance (the four v1 comparisons), distill vs
random-advice, distill vs optimistic-init, and constant-eps-life vs
one-long-life. Robustness family (m=7): distill-shortest vs the same six
baselines, and distill vs distill-shortest (tie-break sensitivity,
reported whichever way it lands).

**Per-claim conclusion** replaces `conclusion.supported`:
`conclusion = {claims: [{claim, verdict, evidence}], summary}`, verdict in
{supported, refuted, null, boundary}. Per-comparison verdict rule:
supported iff both tests significant with the first arm above (IQM,
falling back to means on IQM ties); refuted iff both significant the
other way; boundary iff exactly one test significant; null iff neither.

1. *peak-experience distillation ratchets across generations* — decided
   on distill vs no-inheritance. Evidence must report the ratchet as
   retention-of-peaks (goal-bearing advice hand-offs, consolidations,
   peak-loss events) and the bimodal final distribution (failed lineages
   end at 0.0, below the 0.3 trap floor), per the verifiers.
2. *the bottleneck beats weight copying* — decided on distill vs
   weight-copy.
3. *advice content matters beyond optimism scatter* — conjunctive over
   distill vs random-advice AND distill vs optimistic-init: any leg
   refuted -> refuted; both supported -> supported; both null -> null;
   otherwise boundary. This is the decisive new test: the strong reading
   of H4 requires the advice CHANNEL to beat content-free optimism at
   matched dose (random-advice) and at maximal dose (optimistic-init).
4. *plasticity decay is what strands the long life* — decided on
   constant-eps-life vs one-long-life, mapped as a rescue test:
   comparison supported -> supported; boundary -> boundary; null OR
   refuted -> refuted. A null rescue is registered as refutation, not
   absence of evidence: with 60 seeds of a near-deterministic outcome,
   undying exploration failing to lift the long life off the 0.3 floor
   is a positive demonstration that exploration decay was not the
   binding constraint (the lr-still-decays caveat is carried in the
   evidence, alongside weight-copy — fresh schedules, inherited values —
   and one-long-life-slow as the complementary plasticity probes).

**Fresh confirmatory seeds, n=60.** A `--seed-offset` flag (default 100)
shifts the whole stream family; the v2 confirmatory run uses 60 seeds =
range 100-159 (`--seeds 60 --seed-offset 100`), disjoint from every seed
ever used for tuning or calibrating this experiment (0-29, plus the
tie-break calibration on prototype code at 0-9). Disclosure: verifiers
reran the UNMODIFIED v1 code on 100-129 and 300-329 while probing; no v2
design choice derives from those outcomes beyond the mandates listed
here, and the v2 range is fixed by the harmonized cross-experiment
protocol (100..100+N-1). n=60 also fixes the v1 power problem (the
one-seed significance margin at n=30).

**Reporting hygiene.** Seed numbers in the JSON carry the offset (seed
100 is stream [100, ci]); `config.eval_protocol` states that eval is a
single deterministic argmax rollout (a fixed-tie-break proxy for the
agent's randomized-tie greedy policy — per-seed 0.0 finals mean the
argmax policy loops); `curves.big_goal.stat_note` says to plot the mean
(P(big goal)), not the IQM of a 0/1 variable; smoke runs stamp
`config.smoke=true` and a `conclusion.note` marking them non-confirmatory
(v1's smoke JSON silently concluded the opposite of the full run). The
viz block gains the four new conditions everywhere (curves, final greedy
paths, per-generation records) plus a `condition_labels` map.
