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

# Related Work

Each of our five experiments sits on an established literature: the channel spectrum between imitation and pure reinforcement
(EXP4's advice bottleneck, EXP2's boundary condition), latent learning and world models (EXP1), curricula and part-whole training
(EXP2), variable practice and contextual interference (EXP3), generational transmission (EXP4), and morphological development
(EXP5). We review each thread, then locate our contribution.

## Between imitation and evaluation

Behavioral cloning compounds its own errors — per-step error epsilon costs up to O(T^2 epsilon) regret — and DAgger restores O(T
epsilon)-order guarantees by reducing interactive imitation to no-regret online learning (Ross et al. 2011). Demonstrations inside
value-based RL buy a head start but can anchor: DQfD won the first million steps on 41 of 42 Atari games, yet its persistent
margin loss kept it above its own demonstrator on only 14 (Hester et al. 2018). Annealing the teacher away removes the tether —
kickstarted students reach teacher level ~10x faster and end 42% above baseline because the distillation weight decays to zero
(Schmitt et al. 2018) — and Jump-Start RL keeps instruction in the exploration distribution only, rolling in a guide policy for a
receding horizon while the objective stays pure RL (Uchendu et al. 2023). Offline-to-online handoffs are delicate — distribution
shift can destroy a good initial policy (Lee et al. 2021), over-conservative pretraining dips before recovering (Nakamoto et al.
2023) — while mixing offline data into the online buffer is provably efficient under coverage (Song et al. 2023). Teacher quality
caps the value of imitation, not of RL (Kumar et al. 2022); RLHF chains the stages — supervised imitation, then optimization
against a learned proxy under a KL tether (Ouyang et al. 2022).

Theory bounds when instruction helps: a guide policy covering the optimal policy's states makes naive epsilon-greedy exploration's
exponential-in-horizon sparse-reward sample complexity polynomial (Uchendu et al. 2023), while the lower bound of Xie et al.
(2021) shows the worst-case optimal algorithm either reduces purely to the reference policy or ignores it — instruction is worth
exponential factors to naive explorers and approximately nothing to optimistic ones. That asymmetry is load-bearing here: it
predicts EXP2's boundary result, where the drill curriculum's advantage disappears against an optimistically initialized
whole-game learner. The channel also matters independently of content: priming values or policies from a demonstration gave no
speedup where priming a dynamics model enabled one-trial pole balancing (Schaal 1997); Q-value initialization is update-for-update
equivalent to potential-based shaping (Wiewiora 2003), itself the only reward modification guaranteed policy-invariant (Ng et al.
1999); and a fixed advice budget is best spent at high-importance states rather than early (Torrey & Taylor 2013). The organizing
lesson: acceleration comes from artifacts reshaping exploration, lock-in from artifacts persisting in the objective. EXP4's advice
prime — a value artifact with no persistent loss term, freely overwritten by TD updates — sits on the benign side by design.

## World models and latent learning

Latent learning is the original evidence that agents acquire structure without reward: rats that explored mazes unrewarded
collapsed their error curves within a day or two of the first reward to the level of always-rewarded controls (Blodgett 1929;
Tolman & Honzik 1930), synthesized by Tolman (1948) into the cognitive-map hypothesis. The cautionary half matters equally: the
famous sunburst shortcut probe (Tolman et al. 1946) replicates poorly, exceeding chance in only 17% of 47 experiments (Duvelle &
Grieves 2026), so we rest conclusions on repeated probes, never one dramatic display. Dead reckoning — the competence our
blindfold test isolates — is biologically real and lawful: gerbils home to displaced pups in darkness on direct vectors from
self-motion cues (Mittelstaedt & Mittelstaedt 1980); blind and sighted humans path-integrate equally well, error growing with path
length and complexity (Loomis et al. 1993); and accumulated drift is reset by landmark fixes (Etienne & Jeffery 2004) — the
pattern our slip noise, distance effects, and bump-driven belief re-anchoring reproduce.

In RL, Dyna made planning "RL applied to simulated experience" (Sutton 1990), and world models scaled the idea: policies trained
inside a learned dream transfer to reality, though at low dream temperature the policy exploited model flaws, scoring ~2086 in the
dream but ~193 in reality (Ha & Schmidhuber 2018); one-step errors compound with rollout horizon, mitigated by short branched
rollouts (Janner et al. 2019); latent models predicting only reward, value, and policy plan at superhuman level (Schrittwieser et
al. 2020); and one fixed Dreamer configuration spans 150+ tasks (Hafner et al. 2023). Two findings discipline our claims. Blind
navigation success alone does not require an explicit model — agents sensing only egomotion reach ~95% success and spontaneously
develop decodable maps (Wijmans et al. 2023) — so our inference rides on the home-versus-stranger and touch/no-touch contrasts,
not on blind success per se. And experience replay is effectively a non-parametric model, so claimed model-based sample-efficiency
gains can be pure update-count artifacts (van Hasselt et al. 2019) — hence EXP1's update-matched replay baseline.

## Curricula and part-whole practice

Curriculum learning entered ML as easy-examples-first (Bengio et al. 2009); curriculum RL was formalized as task generation,
sequencing, and transfer (Narvekar et al. 2020) using the metrics of Taylor & Stone (2009), whose sharpest lesson is budget
accounting: most surveyed work demonstrates only "weak transfer," drill time treated as sunk cost. The earliest RL curriculum
already won under strong accounting — easy-pole-first balancing took 73 total failures against 119 direct (Selfridge et al. 1985)
— as did the direct ancestors of our soccer drills: shooting learned by starting episodes near the goal and moving them back
(Asada et al. 1996), and auto-generated shoot and dribble microtasks in Half Field Offense that beat from-scratch learning with
curves offset by all source-task time (Narvekar et al. 2016). Start states alone are a powerful curriculum: reverse curricula grow
the start distribution backward from the goal (Florensa et al. 2017), with theory from restart distributions that cover a good
policy's state visitation (Kakade & Langford 2002); hindsight relabeling is the main non-curriculum alternative for sparse goals
(Andrychowicz et al. 2017). EXP2's drills change only the start-state distribution inside the same MDP, so we attribute any win to
this restart mechanism — not to "task decomposition" as such — and charge every drill step to the budget.

The human part-task literature conditions the prediction rather than guaranteeing it: part training works best when subtask
complexity is high and inter-subtask organization low (Naylor & Briggs 1963); the segmentation wins used backward chaining
(Wightman & Lintern 1985), yet forward chaining beat backward on a keyboard skill (Ash & Holding 1990), so our shoot-first
ordering is one point in a task-dependent space. Curricula also have a budget regime: curriculum ordering conferred no benefit in
standard supervised settings, helping only under limited budget or label noise (Wu et al. 2021) — so we separate time-to-threshold
from asymptotic claims and scope the effect to the tested budget.

## Variable practice and contextual interference

Shea & Morgan (1979) is EXP3's target result: blocked practice looked better during acquisition (1.32 vs 1.69 s), then the
ordering reversed at retention (1.31 vs 1.73 s favoring random acquisition) and transfer, the blocked deficit largest under
changed test contexts. Schema theory predicted the related variability-of-practice effect (Schmidt 1975), most strikingly when
varied practice that never included the criterion distance beat practice at the criterion itself (Kerr & Booth 1978); the umbrella
concept is desirable difficulties — conditions that impair acquisition while enhancing retention and transfer (Bjork 1994). The
boundary conditions sit exactly where our task lives: contextual interference is most reliable when variants require different
motor programs, weaker for parameter-only variation (Magill & Hall 1990); it is meta-analytically large in laboratory tasks (SMD
0.92) yet null in applied sport (Czyz et al. 2024); and in the closest analog of our paradigm — piano-like sequences under tempo
variability — lower variability and non-random schedules transferred better, reversing the textbook prediction (Caramiaux et al.
2018). We therefore pre-committed to publishing whichever direction the in-silico effect takes.

Across ~80 years of motor learning, categorization, perception, language, and ML the synthesis is consistent: low-variability
input is learned fast and generalizes poorly; high-variability input, the reverse (Raviv et al. 2022). RL operationalizes practice
variation as environment diversity: domain randomization transfers from non-realistic simulation to the real world (Tobin et al.
2017), CoinRun and Procgen show an explicit train-worse/test-better crossover as level diversity grows (Cobbe et al. 2019; Cobbe
et al. 2020), and small-scale agents memorize fixed configurations by default (Zhang et al. 2018). What that literature does not
do — and EXP3 does — is hold the variant set fixed and manipulate only the schedule, over an engineered shared-feature substrate
for interference, with the full acquisition/retention/transfer battery.

## Generational transmission

Distillation lineages show students exceeding teachers: policy-distillation students match or beat their teachers at a fraction of
the size (Rusu et al. 2016), and born-again networks — equal-capacity students trained on teacher outputs, generation after
generation — outperform their teachers, gains arriving early and saturating (Furlanello et al. 2018); self-distillation theory
predicts continued rounds eventually underfit, an inverted-U over generations (Mobahi et al. 2020). Reincarnating RL formalizes
reuse of prior computation but transfers the full teacher, with no curated bottleneck and no teacher aging (Agarwal et al. 2022);
distilling into a freshly initialized network within one lineage repairs damage from early non-stationarity (Igl et al. 2021); and
imitating one's own top-return episodes helps within a single lifetime (Oh et al. 2018) — a confound any generational claim must
separate from the fresh-student step.

The iterated-learning tradition supplies the theory of transmission itself: human diffusion chains learning artificial languages
through a bottleneck become more learnable and structured without anyone intending it (Kirby et al. 2008); for posterior-sampling
learners the chain converges to the learner's prior, bottleneck size setting the rate, not the endpoint (Griffiths & Kalish 2007);
and structure requires joint pressure for compressibility and expressivity — transmission-only chains collapse (Kirby et al.
2015). In our chains, the student's post-imitation reward-driven lifetime is that expressivity pressure. Cultural transmission has
reached deep RL as a single expert-to-novice step acquired zero-shot (Bhoopchand et al. 2023) and as generational accumulation
that beats one long life at matched cumulative experience (Cook et al. 2024); human chains given two lives per generation
accumulate by passing distilled written messages (Tessler et al. 2021) — the nearest relative of our 100-pair advice artifact. The
null model is structural drift: chains can wander into degenerate absorbing states, so improvement is not guaranteed (Crutchfield
& Whalen 2010).

The aging half of EXP4 is motivated by plasticity loss: deep continual learners progressively lose the ability to learn (Dohare et
al. 2024). But the same literature raises the identifiability question our controls exist to answer: periodically resetting
network parameters while keeping the replay buffer already improves deep RL by curing primacy bias (Nikishin et al. 2022), so an
apparent generational benefit could be "just resets" — restored plasticity rather than transmitted content. EXP4's battery targets
exactly this: no-inheritance lineages isolate the reset effect alone, weight-copy separates inherited content from inherited
freshness, and one-long-life controls with matched and slowed plasticity decay price the cost of aging. We present teacher aging
as a model inspired by deep-network plasticity loss, not a phenomenon intrinsic to tabular learners — which is what keeps the
testbed clean.

## Morphological development

Bongard (2011) found that robots whose body plans changed during evolution found successful controllers faster and ended more
robust — but by evolutionary search over controllers, not within-lifetime RL, which is exactly the transfer our experiment tests.
Developmental robotics articulates the rationale of immaturity as a dimensionality-reducing scaffold (Lungarella et al. 2003),
with the complication that a single monotonic freeze-then-free schedule can be insufficient (Berthouze & Lungarella 2004). The
empirical record is mixed: across five morphological-development strategies on a bipedal walker, development helped in some setups
and hurt in others (Naya-Varela et al. 2023), while the strongest recent positive result required realistic child-to-adult
anthropometry — uniformly scaled morphology failed (Badie et al. 2025). Morphology causally modulates learnability (Gupta et al.
2021), and transferring policies through a continuous sequence of intermediate bodies beats abrupt transfer (Liu et al. 2022) —
the prior for grow-linear over grow-jump.

The infant data license our cost model: walking infants average ~2,400 steps and 17 falls per hour (Adolph et al. 2012), and
falling is cheap — over 90% of spontaneous infant falls are uneventful, and measured fall energy is 18.4x lower than the same
falls at adult size and speed (Han & Adolph 2021), the anchor for our (s/s_adult)^4 damage law. The counter-evidence is equally
real: balance knowledge can fail to transfer across postures (Adolph 2000), so a balance-first null would be developmentally
consistent. We also model the physics honestly: the inverted pendulum's time constant sqrt(l/g) means the small body falls faster
and is harder for any delay-limited controller (Milton et al. 2009); smallness's true advantages are the L^4 fall cost and
relative actuation authority, and we disclose that direction rather than rig it. Standing precedes walking in the WHO milestone
windows (WHO Multicentre Growth Reference Study Group 2006), the basis for balance-first; assistive forces annealed to zero (Yu et
al. 2018) are the established non-morphological easy-start against which "growing the body specifically helps" would ultimately be
tested.

## Positioning

Much of this program is deliberate replication: preregistered developmental-psychology and motor-learning effects — latent
learning, part-task curricula, the Shea & Morgan crossover, cheap infant falls — reproduced in minimal, fully inspectable RL
systems where every value table, model, belief, and advice set can be printed and audited. Against that baseline we claim four
protocols as new. (a) The blindfold evaluation of a learned world model: prior work trains blind agents end-to-end (Wijmans et al.
2023) or evaluates models by dream-training transfer (Ha & Schmidhuber 2018); using an already-learned Dyna model as the substrate
for belief-filtered dead reckoning, with home-versus-stranger and touch/no-touch contrasts as discriminating controls, is a new
evaluation protocol rather than a new algorithm. (b) Iterated distillation through an episodic-memory bottleneck with
plasticity-decaying teachers: the pieces exist separately — generational accumulation (Cook et al. 2024), zero-shot cultural
transmission (Bhoopchand et al. 2023), full-policy reuse (Agarwal et al. 2022), born-again distillation (Furlanello et al. 2018),
fresh-network distillation within a lineage (Igl et al. 2021), resets whose channel is the entire unfiltered buffer (Nikishin et
al. 2022), and the bottleneck theory of Kirby et al. (2008) and Griffiths & Kalish (2007) — but no published work we found
combines an aging teacher, fresh tabular students, and a hard countable trajectory cap with the weight-copy / long-life /
no-inheritance causal battery. (c) In-silico contextual interference via engineered shared-feature interference: the RL
generalization literature varies environment diversity and measures gaps (Cobbe et al. 2019) but does not run the Shea & Morgan
schedule manipulation over a fixed variant set with a mechanism hypothesis and an acquisition-retention-transfer battery. (d)
Morphological growth curricula with square-cube-honest physics: prior art is evolutionary (Bongard 2011), neuroevolutionary with
mixed results (Naya-Varela et al. 2023), or deep and muscle-actuated (Badie et al. 2025); none runs within-lifetime tabular RL
with a disclosed sqrt(l/g) difficulty direction, L^4 fall-cost scaling anchored to Han & Adolph (2021), and cumulative damage as
the headline metric. Where we replicate, we say so loudly; where we claim novelty, we claim exactly these four protocols and no
more.


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
max(0, 1 - d/5) at Chebyshev distance d from goal center, and a miss ends the
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
0.3·2^(-age/5000), eps(age) = 0.4·2^(-age/5000). Old agents are rigid. Each
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
(difference CI [-40,600, -23,800]); grow-adaptive-walk 52,000 vs 81,000
(CI [-39,800, -15,400]).

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

# References

- A. S. Etienne and K. J. Jeffery (n.d.). *Path integration in mammals*. Hippocampus.
- Agrim Gupta and Silvio Savarese and Surya Ganguli and Li Fei-Fei (n.d.). *Embodied intelligence via learning and evolution*. Nature Communications.
- Andrei A. Rusu and Sergio Gomez Colmenarejo and Caglar Gulcehre and Guillaume Desjardins and James Kirkpatrick and Razvan Pascanu and Volodymyr Mnih and Koray Kavukcuoglu and Raia Hadsell (n.d.). *Policy Distillation*. ICLR.
- Andrew Patterson and Samuel Neumann and Martha White and Adam White (n.d.). *Empirical Design in Reinforcement Learning*. Journal of Machine Learning Research.
- Andrew Y. Ng and Daishi Harada and Stuart Russell (n.d.). *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*. ICML.
- Ashvin Nair and Abhishek Gupta and Murtaza Dalal and Sergey Levine (n.d.). *AWAC: Accelerating Online Reinforcement Learning with Offline Datasets*. https://arxiv.org/abs/2006.09359.
- Aviral Kumar and Joey Hong and Anikait Singh and Sergey Levine (n.d.). *When Should We Prefer Offline Reinforcement Learning Over Behavioral Cloning?*. ICLR.
- Avishkar Bhoopchand and Bethanie Brownfield and Adrian Collister and others (n.d.). *Learning Few-Shot Imitation as Cultural Transmission*. Nature Communications.
- Baptiste Caramiaux and Frederic Bevilacqua and Marcelo M. Wanderley and Caroline Palmer (n.d.). *Dissociable effects of practice variability on learning motor and timing skills*. PLoS ONE.
- Carlos Florensa and David Held and Markus Wulfmeier and Michael Zhang and Pieter Abbeel (n.d.). *Reverse Curriculum Generation for Reinforcement Learning*. CoRL.
- Cedric Colas and Olivier Sigaud and Pierre-Yves Oudeyer (n.d.). *A Hitchhiker's Guide to Statistical Comparisons of Reinforcement Learning Algorithms*. https://arxiv.org/abs/1904.06979.
- Chiyuan Zhang and Oriol Vinyals and Remi Munos and Samy Bengio (n.d.). *A study on overfitting in deep reinforcement learning*. https://arxiv.org/abs/1804.06893.
- D. Ha and J. Schmidhuber (2018). *World Models*. NeurIPS.
- D. Hafner and J. Pasukonis and J. Ba and T. Lillicrap (n.d.). *Mastering Diverse Domains through World Models (DreamerV3)*. https://arxiv.org/abs/2301.04104.
- Daniel W. Ash and Dennis H. Holding (n.d.). *Backward versus Forward Chaining in the Acquisition of a Keyboard Skill*. Human Factors.
- Danyang Han and Karen E. Adolph (n.d.). *The impact of errors in infant development: Falling like a baby*. Developmental Science.
- Dennis C. Wightman and Gavan Lintern (n.d.). *Part-Task Training for Tracking and Manual Control*. Human Factors.
- E. C. Tolman (n.d.). *Cognitive maps in rats and men*. Psychological Review.
- E. C. Tolman and B. F. Ritchie and D. Kalish (n.d.). *Studies in spatial learning. I. Orientation and the short-cut*. Journal of Experimental Psychology.
- E. C. Tolman and C. H. Honzik (n.d.). *Introduction and removal of reward, and maze performance in rats*. University of California Publications in Psychology.
- E. Duvelle and R. M. Grieves (n.d.). *Tolman's Sunburst Maze 80 Years on: A Meta-Analysis Reveals Poor Replicability and Little Evidence for Shortcutting*. European Journal of Neuroscience.
- E. Wijmans and M. Savva and I. Essa and S. Lee and A. S. Morcos and D. Batra (2023). *Emergence of Maps in the Memories of Blind Navigation Agents*. ICLR.
- Eric Wiewiora (n.d.). *Potential-Based Shaping and Q-Value Initialization are Equivalent*. Journal of Artificial Intelligence Research.
- Evgenii Nikishin and Max Schwarzer and Pierluca D'Oro and Pierre-Luc Bacon and Aaron Courville (n.d.). *The Primacy Bias in Deep Reinforcement Learning*. ICML.
- H. C. Blodgett (n.d.). *The effect of the introduction of reward upon the maze performance of rats*. University of California Publications in Psychology.
- H. van Hasselt and M. Hessel and J. Aslanides (n.d.). *When to use parametric models in reinforcement learning?*. NeurIPS.
- Hossein Mobahi and Mehrdad Farajtabar and Peter L. Bartlett (n.d.). *Self-Distillation Amplifies Regularization in Hilbert Space*. NeurIPS.
- Ikechukwu Uchendu and Ted Xiao and Yao Lu and Banghua Zhu and Mengyuan Yan and Josephine Simon and Matthew Bennice and Chuyuan Fu and Cong Ma and Jiantao Jiao and Sergey Levine and Karol Hausman (n.d.). *Jump-Start Reinforcement Learning*. ICML.
- J. M. Loomis and R. L. Klatzky and R. G. Golledge and J. G. Cicinelli and J. W. Pellegrino and P. A. Fry (n.d.). *Nonvisual navigation by blind and sighted: Assessment of path integration ability*. Journal of Experimental Psychology: General.
- J. Schrittwieser and I. Antonoglou and T. Hubert and K. Simonyan and L. Sifre and S. Schmitt and A. Guez and E. Lockhart and D. Hassabis and T. Graepel and T. Lillicrap and D. Silver (n.d.). *Mastering Atari, Go, chess and shogi by planning with a learned model*. Nature.
- James C. Naylor and George E. Briggs (n.d.). *Effects of Task Complexity and Task Organization on the Relative Efficiency of Part and Whole Training Methods*. Journal of Experimental Psychology.
- James P. Crutchfield and Sean Whalen (n.d.). *Structural Drift: The Population Dynamics of Sequential Learning*. https://arxiv.org/abs/1005.2714.
- Jette Randlov and Preben Alstrom (n.d.). *Learning to Drive a Bicycle using Reinforcement Learning and Shaping*. ICML.
- John B. Shea and Robyn L. Morgan (n.d.). *Contextual interference effects on the acquisition, retention, and transfer of a motor skill*. Journal of Experimental Psychology: Human Learning and Memory.
- John Milton and Juan Luis Cabrera and Toru Ohira and Shigeru Tajima and Yukinori Tonosaki and Christian W. Eurich and Sue Ann Campbell (n.d.). *The time-delayed inverted pendulum: Implications for human balance control*. Chaos.
- Jonathan Cook and Chris Lu and Edward Hughes and Joel Z. Leibo and Jakob Foerster (n.d.). *Artificial Generational Intelligence: Cultural Accumulation in Reinforcement Learning*. NeurIPS.
- Josh C. Bongard (n.d.). *Morphological change in machines accelerates the evolution of robust behavior*. Proceedings of the National Academy of Sciences.
- Josh Tobin and Rachel Fong and Alex Ray and Jonas Schneider and Wojciech Zaremba and Pieter Abbeel (n.d.). *Domain randomization for transferring deep neural networks from simulation to the real world*. IEEE/RSJ IROS.
- Junhyuk Oh and Yijie Guo and Satinder Singh and Honglak Lee (n.d.). *Self-Imitation Learning*. ICML.
- Karen E. Adolph (n.d.). *Specificity of learning: Why infants fall over a veritable cliff*. Psychological Science.
- Karen E. Adolph and Whitney G. Cole and Meghana Komati and Jessie S. Garciaguirre and Daryaneh Badaly and Jesse M. Lingeman and Gladys L. Y. Chan and Rachel B. Sotsky (n.d.). *How Do You Learn to Walk? Thousands of Steps and Dozens of Falls per Day*. Psychological Science.
- Karl Cobbe and Christopher Hesse and Jacob Hilton and John Schulman (n.d.). *Leveraging procedural generation to benchmark reinforcement learning*. ICML.
- Karl Cobbe and Oleg Klimov and Chris Hesse and Taehoon Kim and John Schulman (n.d.). *Quantifying generalization in reinforcement learning*. ICML.
- Limor Raviv and Gary Lupyan and C. Shawn Green (n.d.). *How variability shapes learning and generalization*. Trends in Cognitive Sciences.
- Lisa Torrey and Matthew E. Taylor (n.d.). *Teaching on a Budget: Agents Advising Agents in Reinforcement Learning*. AAMAS.
- Long Ouyang and Jeff Wu and Xu Jiang and Diogo Almeida and Carroll L. Wainwright and others (n.d.). *Training Language Models to Follow Instructions with Human Feedback*. NeurIPS.
- Luc Berthouze and Max Lungarella (n.d.). *Motor skill acquisition under environmental perturbations: On the necessity of alternate freezing and freeing of degrees of freedom*. Adaptive Behavior.
- M. Janner and J. Fu and M. Zhang and S. Levine (n.d.). *When to Trust Your Model: Model-Based Policy Optimization*. NeurIPS.
- M. L. Mittelstaedt and H. Mittelstaedt (n.d.). *Homing by path integration in a mammal*. Naturwissenschaften.
- Marcin Andrychowicz and Filip Wolski and Alex Ray and Jonas Schneider and Rachel Fong and Peter Welinder and Bob McGrew and Josh Tobin and Pieter Abbeel and Wojciech Zaremba (n.d.). *Hindsight Experience Replay*. NeurIPS.
- Martin Naya-Varela and Andres Faina and Richard J. Duro (n.d.). *Engineering morphological development in a robotic bipedal walking problem: An empirical study*. Neurocomputing.
- Matthew E. Taylor and Peter Stone (n.d.). *Transfer Learning for Reinforcement Learning Domains: A Survey*. Journal of Machine Learning Research.
- Max Lungarella and Giorgio Metta and Rolf Pfeifer and Giulio Sandini (n.d.). *Developmental robotics: a survey*. Connection Science.
- Maximilian Igl and Gregory Farquhar and Jelena Luketina and Wendelin Boehmer and Shimon Whiteson (n.d.). *Transient Non-stationarity and Generalisation in Deep Reinforcement Learning*. ICLR.
- Michael Henry Tessler and Jason Madeano and Pedro A. Tsividis and Brin Harper and Noah D. Goodman and Joshua B. Tenenbaum (2021). *Learning to solve complex tasks by growing knowledge culturally across generations*. CogSci.
- Minoru Asada and Shoichi Noda and Sukoya Tawaratsumida and Koh Hosoda (n.d.). *Purposive Behavior Acquisition for a Real Robot by Vision-Based Reinforcement Learning*. Machine Learning.
- Mitsuhiko Nakamoto and Yuexiang Zhai and Anikait Singh and Max Sobol Mark and Yi Ma and Chelsea Finn and Aviral Kumar and Sergey Levine (n.d.). *Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning*. NeurIPS.
- Nadine Badie and Firas Al-Hafez and Pierre Schumacher and Daniel F. B. Haeufle and Jan Peters and Syn Schmitt (n.d.). *Bioinspired morphology and task curricula for learning locomotion in bipedal muscle-actuated systems*. Communications Engineering.
- Oliver G. Selfridge and Richard S. Sutton and Andrew G. Barto (n.d.). *Training and Tracking in Robotics*. IJCAI.
- Peter Henderson and Riashat Islam and Philip Bachman and Joelle Pineau and Doina Precup and David Meger (n.d.). *Deep Reinforcement Learning that Matters*. AAAI.
- R. S. Sutton (n.d.). *Integrated architectures for learning, planning, and reacting based on approximating dynamic programming*. Proceedings of the Seventh International Conference on Machine Learning (ICML).
- Richard A. Magill and Kellie G. Hall (n.d.). *A review of the contextual interference effect in motor skill acquisition*. Human Movement Science.
- Richard A. Schmidt (n.d.). *A schema theory of discrete motor skill learning*. Psychological Review.
- Richard S. Sutton and Andrew G. Barto (n.d.). *Reinforcement Learning: An Introduction*.
- Rishabh Agarwal and Max Schwarzer and Pablo Samuel Castro and Aaron Courville and Marc G. Bellemare (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice*. NeurIPS.
- Rishabh Agarwal and Max Schwarzer and Pablo Samuel Castro and Aaron Courville and Marc G. Bellemare (n.d.). *Reincarnating Reinforcement Learning: Reusing Prior Computation to Accelerate Progress*. NeurIPS.
- Robert A. Bjork (n.d.). *Memory and metamemory considerations in the training of human beings*. Metacognition: Knowing About Knowing.
- Robert Kerr and Bernard Booth (n.d.). *Specific and varied practice of motor skill*. Perceptual and Motor Skills.
- Sanmit Narvekar and Bei Peng and Matteo Leonetti and Jivko Sinapov and Matthew E. Taylor and Peter Stone (n.d.). *Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey*. Journal of Machine Learning Research.
- Sanmit Narvekar and Jivko Sinapov and Matteo Leonetti and Peter Stone (n.d.). *Source Task Creation for Curriculum Learning*. AAMAS.
- Seunghyun Lee and Younggyo Seo and Kimin Lee and Pieter Abbeel and Jinwoo Shin (n.d.). *Offline-to-Online Reinforcement Learning via Balanced Replay and Pessimistic Q-Ensemble*. CoRL.
- Sham Kakade and John Langford (n.d.). *Approximately Optimal Approximate Reinforcement Learning*. ICML.
- Shibhansh Dohare and J. Fernando Hernandez-Garcia and Qingfeng Lan and Parash Rahman and A. Rupam Mahmood and Richard S. Sutton (n.d.). *Loss of plasticity in deep continual learning*. Nature.
- Simon Kirby and Hannah Cornish and Kenny Smith (n.d.). *Cumulative cultural evolution in the laboratory: An experimental approach to the origins of structure in human language*. Proceedings of the National Academy of Sciences.
- Simon Kirby and Monica Tamariz and Hannah Cornish and Kenny Smith (n.d.). *Compression and communication in the cultural evolution of linguistic structure*. Cognition.
- Simon Schmitt and Jonathan J. Hudson and Augustin Zidek and Simon Osindero and Carl Doersch and Wojciech M. Czarnecki and Joel Z. Leibo and Heinrich Kuttler and Andrew Zisserman and Karen Simonyan and S. M. Ali Eslami (n.d.). *Kickstarting Deep Reinforcement Learning*. https://arxiv.org/abs/1803.03835.
- Stanislaw H. Czyz and Aleksandra M. Wojcik and Petra Solarska and Pawel Kiper (n.d.). *High contextual interference improves retention in motor learning: systematic review and meta-analysis*. Scientific Reports.
- Stefan Schaal (n.d.). *Learning from Demonstration*. NIPS 9.
- Stephane Ross and Geoffrey J. Gordon and J. Andrew Bagnell (n.d.). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS.
- Tengyang Xie and Nan Jiang and Huan Wang and Caiming Xiong and Yu Bai (n.d.). *Policy Finetuning: Bridging Sample-Efficient Offline and Online Reinforcement Learning*. NeurIPS.
- Thomas L. Griffiths and Michael L. Kalish (n.d.). *Language Evolution by Iterated Learning With Bayesian Agents*. Cognitive Science.
- Todd Hester and Matej Vecerik and Olivier Pietquin and Marc Lanctot and Tom Schaul and Bilal Piot and Dan Horgan and John Quan and Andrew Sendonaris and Gabriel Dulac-Arnold and Ian Osband and John Agapiou and Joel Z. Leibo and Audrunas Gruslys (n.d.). *Deep Q-learning from Demonstrations*. AAAI.
- Tommaso Furlanello and Zachary C. Lipton and Michael Tschannen and Laurent Itti and Anima Anandkumar (n.d.). *Born Again Neural Networks*. ICML.
- Wenhao Yu and Greg Turk and C. Karen Liu (n.d.). *Learning symmetric and low-energy locomotion*. ACM Transactions on Graphics (SIGGRAPH).
- WHO Multicentre Growth Reference Study Group (n.d.). *WHO Motor Development Study: Windows of achievement for six gross motor development milestones*. Acta Paediatrica.
- Xiaoxia Wu and Ethan Dyer and Behnam Neyshabur (2021). *When Do Curricula Work?*. ICLR.
- Xingyu Liu and Deepak Pathak and Kris M. Kitani (2022). *REvolveR: Continuous Evolutionary Models for Robot-to-robot Policy Transfer*. ICML.
- Yoshua Bengio and Jerome Louradour and Ronan Collobert and Jason Weston (n.d.). *Curriculum Learning*. ICML.
- Yuda Song and Yifei Zhou and Ayush Sekhari and J. Andrew Bagnell and Akshay Krishnamurthy and Wen Sun (n.d.). *Hybrid RL: Using Both Offline and Online Data Can Make RL Efficient*. ICLR.
