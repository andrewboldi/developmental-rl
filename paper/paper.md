---
title: "Growing Up to Learn: Developmental Scaffolds for Reinforcement Learning"
author: "Andrew Boldi"
date: "August 28, 2026"
abstract: |
  Reinforcement learning is usually framed as a choice between imitation and
  evaluation: copy a teacher, or learn from reward alone. Human learners live
  in the middle — they take instruction, keep world models, drill microtasks,
  practice with deliberate variation, inherit compressed experience from
  teachers, and do all of it inside bodies that start small. We turn five
  developmental intuitions into falsifiable hypotheses and test them in
  deliberately small systems (tabular and linear agents; gridworlds and a
  torque-limited pendulum) where budgets are matched to the environment step,
  confirmatory runs use fresh seeds disjoint from all tuning, and every primary
  comparison is Holm-corrected under both Mann-Whitney and Welch tests. Three
  core effects replicate and survive adversarial verification: (1) a learned
  world model supports near-sighted competence with observations removed —
  97.5% blindfolded success at "home" versus 29% in a "stranger's home" — even
  though its celebrated sample-efficiency advantage is fully explained by
  update count; (2) drill curricula over start states reach whole-task mastery
  2.7x faster for epsilon-greedy learners, an effect driven by targeted
  structure (uniform start diversity is worthless) and bounded by exploration
  regime (optimistic initialization reproduces it); (3) the Shea–Morgan
  contextual-interference crossover appears in silico — interleaved practice
  loses the practice room and wins retention and transfer — and vanishes when
  shared features are ablated, establishing shared-parameter interference as
  the mechanism. A fourth result is new, to our knowledge: iterated
  distillation through an episodic-memory bottleneck, from plasticity-decaying
  teachers to fresh students, consolidates rare peak experiences that no
  dose-matched control can (weight copying, random optimistic advice, longer
  lives), while a global-optimism control marks the boundary where teaching is
  unnecessary. Fifth, growing bodies: on the confound-free pairing, a growth
  curriculum reaches adult competence with 9–42x less cumulative fall damage
  and 30–40% fewer steps — while two folk sub-claims (gradual growth,
  balance-before-walking) are significantly reversed. Two extensions close the
  loop: an agent can coach itself — restarting from moments in its own best
  episodes beats raw practice by 28%, though expert-designed drills remain
  another 2x faster (the measured value of instruction) — and a
  reset-with-full-replay control shows curation is causal: fresh students
  replaying entire goal-bearing teacher lifetimes consolidate 0/35 discovered
  goals versus the distilled lineage's 129/129. Both headline effects
  replicate across freshly generated layouts. We report the failures
  alongside the successes; the surviving thesis is that developmental
  scaffolds work by curating which experience a learner gets, and that they
  pay precisely where exploration is undirected and mistakes are expensive —
  the condition animals actually live under.
geometry: margin=1.05in
fontsize: 10pt
colorlinks: true
---

# 1. Introduction

An agent trained by pure evaluation must rediscover, within a single lifetime,
regularities that every previous lifetime already paid for. An agent trained by
pure imitation inherits those regularities but is ceilinged by its teacher and
stranded the moment the world drifts from the demonstrations. Humans occupy the
productive middle: we take *instruction* — compressed generational experience —
and cash it out through our own trial and error.

This paper takes five specific intuitions about that middle ground, states them
as falsifiable hypotheses, and tests them end-to-end:

**H1 (The Blindfold Test).** Close your eyes in your own home and you can still
find the fridge; close them in a stranger's home and you are lost immediately.
The motor system did not change — the *map* did. We test whether a learned
world model supports competent blind action in a familiar environment and
fails in an unfamiliar one, and whether the model's celebrated sample-efficiency
benefit survives an update-matched control.

**H2 (Microtasks).** Nobody learns soccer by only playing soccer, and no piano
teacher assigns only full run-throughs. We test whether a curriculum that
changes *nothing but the start-state distribution* — shooting drills, then
dribbling drills, then full games — beats whole-game practice at matched
budgets, and against two controls that decompose the mechanism: uniform random
starts (diversity without structure) and optimistic initialization (a better
explorer with no curriculum at all).

**H3 (Variation Practice).** Practicing a passage identically every time feels
efficient and is not. Motor-learning research calls it the
contextual-interference effect: blocked practice wins acquisition and loses
retention and transfer. We test whether the crossover emerges in a minimal
learner, and — by ablating shared features — *why*.

**H4 (Generational Teaching).** A teacher does not transplant their brain into
a student; they transmit a curated sliver — the passages that matter, the one
time they reached the summit. We test whether a lineage of short-lived,
plasticity-decaying agents, each distilling its best episodes through a narrow
bottleneck into a fresh student, outperforms weight copying, longer single
lives, and dose-matched optimism controls.

**H5 (Growing Bodies).** Toddlers fall constantly and cheaply; by the
square-cube law small bodies are also relatively strong. We test whether
starting small and growing — paying for mistakes at toddler prices — reaches
adult competence cheaper and no slower than training the adult body directly,
and whether the folk staging (balance first, grow gradually) actually helps.

Our experimental philosophy is deliberate miniaturization. Every environment is
small enough that budgets can be matched to the individual step, every
comparison run across 30–60 fresh seeds, and every mechanism exposed by
ablation. The claims are correspondingly scoped: these are existence proofs and
mechanism demonstrations in minimal systems, not benchmark results. The program
was subjected to an adversarial verification pass (fifteen independent audits,
three lenses per experiment) followed by a pre-registered hardening round:
every hypothesis re-run on fresh seeds with the auditors' mandated controls.
What failed is reported next to what survived. Notably, several failures are
the most informative results in the paper.

<!--RELATED_WORK-->

# 3. Methods

## 3.1 Shared protocol

All agents are tabular Q-learners or linear-function-approximation Q-learners
implemented in numpy; all environments are custom and fully specified in the
repository. Statistical protocol, applied uniformly: interquartile-mean (IQM)
point estimates; 95% percentile-bootstrap confidence intervals (10,000
resamples); two-sided Mann-Whitney *and* Welch t tests for every primary
comparison, Holm-corrected within each experiment's primary family; censored
time-to-threshold values assigned budget+1 (conservative) with censoring
fractions always reported. Training budgets are matched to the environment
step across conditions — drill steps, practice steps, and growth-phase steps
all count. Evaluation is greedy, identical across conditions, uses rng streams
disjoint from training, and consumes no training budget. Hyperparameters were
selected on seed ranges disjoint from the confirmatory runs (a v1 violation of
this rule, caught by the audit, is disclosed in §5); all confirmatory results
below are from fresh seed ranges (100+). Design amendments between v1 and v2
are versioned in the repository's DESIGN.md.

## 3.2 EXP1 — The Blindfold Test

**Environment.** `GridHome`: two 13x11 apartments (HOME_A, HOME_B) with
identical shells, identical start ("bed") and goal ("fridge") coordinates, and
different interior walls. Motor noise: 10% probability the executed move is
perpendicular to the intent. Reward 1 at the fridge; episode cap 60 steps
(~3.5x the 17-step shortest path); gamma 0.97.

**Agents.** Q-learning; Dyna-Q with 20 planning updates per real step (its
transition/reward model doubles as the agent's imagination); and — mandated by
the audit — ReplayQ, an update-matched control performing 20 uniformly-replayed
updates per real step from its own transition buffer, with no model.

**Blindfold protocol.** After 40k training steps in HOME_A, observations are
removed. The agent dead-reckons: a belief over states, initialized at the known
start, advanced through the learned transition model by each executed action;
actions chosen greedily by belief-weighted Q. The *touch* variant additionally
filters the belief by the learned bump likelihood — a hand trailing the wall.
Conditions: sighted-A, blind-A, blind-A-touch, blind-B, blind-B-touch (both
using HOME_A's model and values inside HOME_B), sighted-B-transfer (HOME_A's
values with full observation in HOME_B), and matched random floors in both
homes. 30 seeds.

## 3.3 EXP2 — Microtasks

**Environment.** `SoccerGrid`: an 11x7 pitch; the ball is carried, loose at a
cell, or scored; 4 moves + shoot; shots score with probability
max(0, 1 − d/5) at Chebyshev distance d from goal center, and a miss ends the
episode; reward only on goals; cap 100 steps.

**Conditions** (60k steps each, one shared Q-table per agent, 30 seeds):
`whole` (full games from kickoff); `drills-varied` (20% of budget spawning
with the ball at random attacking-third cells, 20% spawning at random
own-half cells with the ball at center, then full games — a reverse curriculum
over start states with no reward changes); `drills-fixed` (same phases, single
fixed spawn each); `explore-starts` (uniformly random agent and ball positions
for 40% of budget — diversity without structure); `whole-optimistic`
(full games, Q0 = 1.0 — a better explorer, no curriculum).

## 3.4 EXP3 — Variation Practice

**Environment.** `PianoPiece`: three 12-note passages assembled from a shared
library of six 3-note motifs, one exception position per passage; 8 keys;
+1 per correct note; score = fraction correct.

**Agent.** Linear Q(s,a) = w_a·phi(s), with phi = onehot(motif,
position-in-motif) ++ onehot(passage) ++ onehot(position). The shared motif
slots are where interference can live. The mechanism-control agent replaces
phi with passage-local features only (no shared slots).

**Conditions** (450 episodes each, 40 seeds): `blocked` (one passage per third
of budget, block order counterbalanced across seeds — seed % 6 selects one of
the six orders, removing the v1 recency confound); `interleaved` (uniform
random passage each episode); both feature maps crossed with both schedules
for the mechanism control. Test phase (learning off): retention on all three
passages; transfer to a novel passage built from the same motifs.

## 3.5 EXP4 — Generational Teaching

**Environment.** `TrapGrid` (15x11): start at the left; reward 10 far right;
three candy cells (reward 0.3, terminal) near the start that end episodes
early and throttle deep exploration; cap 120 steps; deterministic moves.

**Lifetimes.** One life = 15k steps with plasticity decay: lr(age) =
0.3·2^(−age/5000), eps(age) = 0.4·2^(−age/5000). Old agents are rigid. Each
agent keeps an episodic memory of its top-3 episodes by return.

**The bottleneck.** At death, a teacher emits advice: the deduplicated (s,a)
pairs of its best-remembered episodes, capped at 100. A fresh student is
primed Q[s, a_advice] = 5.0, then lives normally.

**Conditions** (5 generations x 15k = 75k steps each, 60 seeds):
`generational-distill` (and a tie-break robustness variant);
`weight-copy` (student inherits the full Q-table, fresh schedules);
`one-long-life` and `one-long-life-slow` (single agents, two decay rates);
`no-inheritance` (five independent lives); and the audit-mandated
optimism-decomposition controls: `random-advice` (100 *random* state-action
pairs primed to 5.0 each generation — dose-matched enthusiasm, no content),
`optimistic-init` (one long life, Q0 = 5.0 everywhere), and
`constant-eps-life` (exploration never decays — tests whether plasticity decay
alone strands the long life).

## 3.6 EXP5 — Growing Bodies

**Environment.** `BalanceBot`: a torque-limited inverted pendulum with body
scale s: length s, mass 15s^3, torque limit 40s^2 (square-cube: small bodies
are relatively stronger but dynamically faster per fixed control step — the
per-step difficulty is not rigged in either direction). Falls (|theta| > pi/5)
end the episode and cost damage (s)^4 — impact energy. Torque noise 5% of the
limit. "Walking" = tracking a lean target that switches every 40 steps;
evaluation is always on the adult body (s = 1.0) with walking targets.
765 discretized states, 5 torque actions; 120k steps; 40 fresh seeds.

**Conditions.** `adult-walk`; `adult-balance-first` (targets pinned to zero
for the first 30%); `grow-linear` and `grow-adaptive` (size 0.5 -> 1.0 on a
schedule, or +0.05 whenever the recent no-fall rate exceeds 70%, both
balance-first); `grow-jump` (0.5 until 60%, then 1.0); `grow-linear-walk` and
`grow-adaptive-walk` (growth with walking from the start — the audit-mandated
factorial cells that isolate morphology from task staging). Robustness arms
rerun the primary pairing under a muscle-torque s^3 law and under size-scaled
damping.

## 3.7 EXP6 — The Self-Coach

The program's founding question: with no teacher, can an agent devise its own
microtasks? Environment: `SoccerGrid` unchanged. Conditions (60k steps, 30
fresh seeds): `whole`; `teacher-drills` (EXP2's drills-varied protocol,
bit-identical rerun, enforced by test); `self-drills` — the agent keeps an
online episodic memory of its top-10 episodes by return, and for the first 40%
of budget starts 75% of episodes from a uniformly sampled state snapshot drawn
from those episodes' visited states ("practice moments from my best games"),
falling back to restarts from any visited state until the first score;
`self-drills-late` (the same mechanism with the memory-start probability
annealed linearly over the whole budget).

## 3.8 EXP7 — Layout-resampling robustness

Generators (`layouts.py`, guarded and tested) produce fresh TrapGrid instances
(candy positions resampled under a random-walk absorption guard) and fresh
home pairs (random room partitions, same shell and endpoints, solvability and
path-length guards). The two headline pairings are replicated per instance:
EXP4's distill vs no-inheritance on 5 TrapGrids (20 seeds each) and EXP1's
blind-A-touch vs blind-B-touch on 5 home pairs (15 seeds each). Registered
rule: robustness holds iff the direction replicates at per-instance p < 0.05
in at least 4 of 5 instances.

# 4. Results

Every claim below is scored by its pre-registered rule; "supported" and
"refuted" refer to those rules, evaluated on fresh confirmatory seeds with
Holm-corrected Mann-Whitney and Welch tests. Figures 1–5 show the primary
curves; the interactive version of every figure, with per-seed data, is at
the project site.

## 4.1 EXP1: the map is for acting blind, not for learning fast

![EXP1. (a) Sample efficiency in HOME_A: the update-matched replay control (no model) matches Dyna-Q exactly; the speedup is update count. (b) The blindfold test: touch-filtered blind navigation matches sighted at home and collapses in the stranger's home.](figs/fig1_blindfold.pdf)

**The blindfold results (Fig. 1b).** Sighted competence in HOME_A is 0.973
(IQM success). Pure dead reckoning collapses to 0.204 — the v1 prediction that
an internal model alone sustains blind competence under motor noise is
**refuted**; compounding slip drift destroys the open-loop belief, exactly as
the imitation-learning literature's compounding-error analyses would predict.
But dead reckoning *plus a learned touch filter* — belief updates multiplied
by the bump likelihood, a hand on the wall — reaches **0.975, statistically
indistinguishable from sighted** (0.975 vs 0.973). The same machinery
transported to the stranger's home collapses to 0.290 against a measured
matched random floor of 0.033 (blind-A-touch vs blind-B-touch: Welch Holm
p = 6.7e-18; the symmetric touch-pair comparison was promoted to primary by
the audit after the pure-pair comparison proved knife-edge). Sighted transfer
with HOME_A's values fails entirely in HOME_B (0.000): the bottleneck is
layout knowledge, not vision.

**The sample-efficiency twist (Fig. 1a).** Dyna-Q reaches 90% success in
2,562 steps versus 25,500 for Q-learning (Welch Holm p = 7.3e-31) — the
classic result. The update-matched ReplayQ control lands at **2,500 steps**,
closing 100% of the gap (Dyna vs replay: p = 0.55). The strong claim
"the model causes the speedup" is **refuted**: at matched update counts the
advantage is fully an update-count effect (van Hasselt et al.'s warning,
confirmed in the smallest possible setting). What replay cannot do is walk
blind — no buffer of past transitions yields a belief update for a novel
action sequence. The world model's non-redundant value in this system is
counterfactual use, not sample efficiency.

## 4.2 EXP2: drills work — for the learner that needs them

![EXP2. Full-game success from kickoff. Drilled start-state curricula lift off during the drill phases; uniform exploring starts do nothing; optimistic initialization needs no curriculum.](figs/fig2_drills.pdf)

On fresh seeds, `drills-varied` reaches 90% full-game success in **15,125**
steps (IQM) versus **40,375** for `whole` (MW Holm p = 3.9e-9, Welch Holm
p = 3.4e-9; 0/30 vs 6/30 seeds censored): a 2.7x speedup from changing
nothing but where episodes start. The two controls decompose the mechanism.
`explore-starts` — the same 40% of budget spent on uniformly random starts —
helps not at all (43,375; p = 0.39 vs whole): start-state *diversity* is
worthless; the drills' *targeted* structure (starts concentrated where value
already flows) carries the entire effect (drills vs explore-starts: Welch Holm
p = 4.9e-14). `whole-optimistic` reaches 16,000 steps with no curriculum at
all — statistically indistinguishable from the drilled learner (p = 0.18) —
so the drill benefit is an **exploration effect, bounded by exploration
regime**: it is large for undirected (epsilon-greedy) learners and unnecessary
for optimistic ones, precisely the boundary the guide-policy theory predicts.

## 4.3 EXP3: the crossover replicates, and the mechanism is proven

![EXP3. (a) Acquisition: blocked practice leads during training. (b) Test with learning off: interleaved wins retention and transfer — the contextual-interference crossover.](figs/fig3_variation.pdf)

With block order counterbalanced (killing the v1 recency confound), fresh-seed
results: acquisition — blocked 0.796 vs interleaved 0.777, direction as
predicted but no longer significant (MW p = 0.076): **boundary**. Retention —
interleaved **0.910** vs blocked 0.793 (Welch Holm p = 5.1e-10): **supported**.
Transfer to a novel passage — interleaved **0.779** vs blocked 0.621 (Welch
Holm p = 4.7e-6): **supported**. The order-relative analysis shows the
retention gap lives on earlier-drilled passages, as shared-weight drift
predicts, and the v1 "blocked wins the just-drilled passage" reversal shrinks
to noise once orders are counterbalanced — it was an order artifact.

The mechanism control is the sharpest result: rebuild the learner with
passage-local features (no shared motif slots) and the entire crossover
**vanishes** — retention gap p = 0.92, transfer at chance for both schedules.
Contextual interference here is not a story about attention or effort; it is
shared-parameter interference, demonstrated by ablation.

## 4.4 EXP4: the bottleneck beats every dose-matched control

![EXP4. Greedy return at each generation's end. The distill lineage ratchets to the distant goal; every dose-matched control stays at candy; random advice poisons below the floor.](figs/fig4_generations.pdf)

At 60 fresh seeds: `generational-distill` ends at IQM **10.0** (48/60 lineages
consolidate the distant goal; mean 8.0). Every v1 baseline — weight-copy, both
long lives, no-inheritance — ends at exactly **0.3 on every seed** (candy;
all four comparisons Welch Holm p = 1.5e-20). The mechanism telemetry shows
why, and shows the ratchet is genuinely cumulative: only 19 of 60 long-lived
agents ever end holding a memory of the mountain (best-episode 10) — and every
one of them still *does* candy at death, knowledge they can no longer
consolidate. Likewise only 16 of 60 first-generation teachers remember the
mountain at all, yet 48 of 60 lineages end there: each generation's own
exploration adds new peaks to the next generation's inheritance, and the
bottleneck converts every remembered peak into a prior the student can finish.

The audit-mandated controls decompose the effect. `random-advice` — the same
dose of optimism on random state-action pairs — ends at **0.0**, *below* the
candy floor (scattered optimistic pins poison the greedy policy into loops):
advice works because of its content, not its enthusiasm (Welch Holm
p = 2.0e-20). `constant-eps-life` (exploration never decays) still ends at 0.3
on all 60 seeds — undecayed epsilon alone does not rescue the long life, so
the registered "plasticity decay is the strander" rescue-claim is **refuted**
as stated (the lr component still decayed; the triangulation is in the
repository). And `optimistic-init` — global Q0 = 5.0, one long life — solves
TrapGrid outright, 60/60, beating even the distill lineage (Welch Holm
p = 6.0e-4).

The v3 identifiability controls settle the "isn't this just resets?" question
directly. A fresh student that replays its teacher's *entire* 15,000-transition
lifetime log before living — raw experience inheritance at matched plasticity —
ends at the candy floor on all 60 seeds, as does the dose-matched variant
replaying 100 transitions (both vs distill: MW Holm p = 2.5e-9, Welch Holm
p = 1.5e-20). The consolidation accounting is the sharpest statistic in the
program: 35 goal-bearing full logs were handed to fresh students and replayed
in their entirety, and **0 of 35** were consolidated into policy; the distill
lineage consolidated **129 of 129** goal-bearing advice hand-offs. A shuffled
single pass of TD updates cannot propagate value down a long path; priming the
exact trajectory can. Inheriting raw experience is not inheriting the lesson —
the curation is causal. The strong conjunctive claim "advice content matters beyond any
optimism control" is therefore **refuted**, and the honest statement of H4 is
conditional: *among content-bearing channels and dose-matched controls,
episodic distillation is uniquely effective; where blanket optimism over the
whole state space is feasible, no teacher is needed.* Blanket optimism is
feasible in a 165-state gridworld; it is not a scalable strategy where states
are vast and optimism is expensive to burn off — which is where teaching
should matter, a prediction this miniature cannot itself test.

## 4.5 EXP5: growth is cheaper and faster; the folk stagings reverse

![EXP5. Cumulative fall damage during training, by body plan. Growth curricula reach adult competence at a fraction of the damage.](figs/fig5_growing.pdf)

On the audit-mandated confound-free pairing (both arms walking from the
start): `grow-linear-walk` reaches adult competence with damage IQM **100.4**
vs **928.5** for `adult-walk` (9.2x; Welch Holm p = 3.8e-35);
`grow-adaptive-walk` with **22.0** (42x; Welch Holm p = 4.3e-30) — robust to
censoring treatment and to both physics robustness arms (muscle-torque s^3
law; size-scaled damping). Steps-to-competence: growth is not merely
no-slower but significantly **faster** — grow-linear-walk 49,000 vs 81,000
(difference CI [−40,600, −23,800]); grow-adaptive-walk 52,000 vs 81,000
(CI [−39,800, −15,400]).

Both folk sub-claims **reverse**. Balance-first staging is slower on both
matched-morphology pairings (adult: 101,600 vs 81,000, Welch Holm p = 7.4e-4;
grow-linear: 95,800 vs 49,000, Welch Holm p < 1e-9): weight-shift practice
visits the very states balance needs, and a balance-only phase starves the
rest of the table. And gradualism reverses: one abrupt jump from half-size to
adult is *cheaper* than fine-grained growth (damage 156.2 vs 446.8, Welch Holm
p = 5.1e-7). The savings come from being small while incompetent and spending
little total time mid-growth — not from smoothness of the schedule, and not
from imitating infant task staging. What infant development optimizes is
evidently not the value function's learning curve alone.

## 4.6 EXP6: the agent can coach itself — and the teacher is still worth double

![EXP6. The Self-Coach: an agent restarting from moments in its own best
games beats whole-game practice; the teacher-designed curriculum remains
substantially faster.](figs/fig6_selfcoach.pdf)

`self-drills` reaches 90% full-game success in **29,000** steps (IQM) versus
**40,375** for `whole` — 28% faster, from nothing but the agent's own episodic
memory (MW Holm p = 0.021, Welch Holm p = 0.020; 2/30 vs 6/30 censored). The
registered "self matches teacher" claim is **refuted**, informatively:
`teacher-drills` at **15,125** steps beats the self-coach by 13,875 steps
(Welch Holm p = 2.5e-6; gap CI [9,125, 19,500] against a pre-registered
+3,025 margin). The schedule ablation (`self-drills-late`, 28,750) shows the
result is not an artifact of the phase boundary. The mechanism telemetry
mirrors the design: fallback exploration restarts dominate only until the
first scored episode, after which the practice-start distribution contracts
onto the discovered scoring path. Read together with EXP4, this quantifies the
founding intuition: an agent can curate its own experience within one
lifetime (worth 28% here), and instruction — experience curated across
lifetimes by someone who already knows the path — is worth roughly a further
2x. Both channels beat raw practice; neither replaces the other.

## 4.7 EXP7: the headline effects are not layout artifacts

On five freshly generated TrapGrids, distill-vs-no-inheritance replicates in
4/5 instances with zero significant reversals (the miss: IQM 8.0 vs 0.3 at
n = 20, p = 0.078 — direction strongly held). On five freshly generated home
pairs, the blindfold contrast replicates in **5/5** instances (per-instance
MW p between 2.4e-5 and 3.8e-4; pooled n = 75 vs 75: IQM 0.89 vs 0.13, Welch
Holm p = 2.0e-30). Both registered robustness rules are met; the program's
two headline results are properties of the mechanisms, not of the two
hand-drawn maps they were discovered on.

# 5. What failed, and what the failures bought

Seven registered claims were refuted and two landed on boundaries; none were
quietly dropped. Refuted: (1) the strong Dyna claim failed its update-matched
control — the correct citation for the speedup is update count, and the
model's real contribution in this system is blind action; (2) pure dead
reckoning is not home-competent under motor noise — competence needs
closed-loop touch; (3) H4's strong conjunction failed against global optimism,
which converts to the paper's sharpest scope statement: instruction pays where
exploration is undirected and optimism is unaffordable; (4) H4's
plasticity-rescue claim failed — undecayed exploration alone does not save the
long life; (5–6) gradualism and balance-first reversed — folk developmental
staging did not survive contact with a system where only the value function
matters; (7) self-devised drills do not match teacher-designed drills — the
self-coach works, and the teacher premium is real and large. Boundaries: EXP2's drill benefit is exploration-regime-dependent, and
EXP3's acquisition edge is direction-consistent but not significant once block
order is counterbalanced — the practice-room advantage of blocked practice is
partly recency artifact even in silico.

We also disclose two v1 protocol violations caught by the audit and repaired
in v2: hyperparameters tuned on seed ranges overlapping confirmation (fixed by
fresh-seed confirmatory reruns with disjoint ranges), and one
outcome-calibrated design choice (EXP4's episodic-memory tie-break) that is
now reported under both variants (IQM 10.0 earliest-tie vs 9.33 shortest-tie;
the conclusion is tie-break-robust at 60 seeds).

# 6. Discussion

The five results compose into one operational thesis. Every scaffold that
survived works by **curating which experience the learner gets** — drills
reposition it, variation shuffles it, teachers select its peaks across
lifetimes, small bodies discount its price — while the learner's update rule
stays untouched and free to disagree. The two boundary results (EXP2, EXP4)
add the scope condition: curation pays when exploration is undirected and
optimism is expensive. That condition is not a weakness of the thesis; it is
a description of animal life. A creature with real fall costs, a finite
lifetime, and a state space it cannot enumerate is precisely the epsilon-greedy
learner of these experiments, not the globally-optimistic one.

The blindfold result reframes what a world model is *for* in the smallest
possible setting: not (only) faster credit propagation — replay does that —
but action under counterfactual observation: acting on experience the agent is
not currently having. The generational result, to our knowledge the first
demonstration of iterated distillation through an episodic bottleneck from
plasticity-decaying teachers to fresh tabular students, exhibits in miniature
the ratchet that iterated-learning theory predicts: the transmission
bottleneck filters consolidated mediocrity and transmits structure — here,
literally the map of the one good day. And the growing-bodies result puts a
number on a developmental folk claim while refuting two others: what makes
starting small valuable is the price of falling, not the smoothness of growth
or the staging of tasks.

# 7. Limitations

Everything here is deliberately miniature: tabular and linear learners,
hand-built environments, one layout per world (a layout-resampling replication
is future work). Effects proven by ablation in minimal systems demonstrate
mechanism existence, not magnitude at scale. The optimism boundary is
specific to enumerable state spaces. EXP5's damage law and torque scaling are
first-principles stylizations; two robustness arms support the ordering, but
a full sensitivity sweep over the physics is open. Statistical power at 30–60
seeds resolves the large effects reported; several boundary results (EXP3
acquisition) are direction-consistent but under-powered by design honesty
rather than certainty.

# 8. Reproducibility

Everything is public: environments, agents, experiment scripts, 321 tests, the
versioned design document with amendments, all raw per-seed results, the paper
source, and the interactive site. One command reruns any experiment
(`python -m devrl.run_all`); every number in this paper is generated by a
committed script from a committed result file.
Repository: <https://github.com/andrewboldi/developmental-rl>. Interactive
results: <https://andrewboldi.github.io/developmental-rl/>.

# Acknowledgments

Experiments, code, verification, and drafting were carried out with Claude
(Anthropic), operating as an autonomous research assistant under the author's
direction; all hypotheses originate with the author.

<!--REFERENCES-->
