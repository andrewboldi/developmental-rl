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
