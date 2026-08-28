# RESEARCH.md — Literature Backbone

**Growing Up to Learn: Developmental Scaffolds for Reinforcement Learning**

This document synthesizes the structured literature reviews behind the five hypotheses of DESIGN.md. Each section states what is known (with the original
numbers), what it implies for our experimental design (noting where DESIGN.md already complies and where it deviates), and the validity risks a reviewer would
raise. Sources: the eight research JSON files in the project scratchpad; nothing is cited here that is not in those files.

## Overview

The five hypotheses each sit on top of an established literature. H1 (world models / the blindfold test) descends from Tolman's latent-learning and
cognitive-map work in rats, the path-integration literature in ants, gerbils, and blindfolded humans, and the model-based RL line from Sutton's Dyna through Ha
& Schmidhuber's World Models, Dreamer, and MuZero. H2 (microtask curricula) instantiates curriculum RL (Selfridge/ Sutton/Barto 1985, Asada 1996, Narvekar's
framework and Half Field Offense drills) and the human part-task-training literature (Wightman & Lintern's taxonomy, Naylor & Briggs' complexity-organization
hypothesis). H3 (variation practice) is the contextual-interference and variability-of-practice tradition (Shea & Morgan 1979, Schmidt 1975, Bjork's desirable
difficulties) meeting RL generalization work (CoinRun/Procgen, domain randomization). H4 (generational teaching through a distillation bottleneck) draws on
distillation lineages (policy distillation, born-again networks, kickstarting, Reincarnating RL), the iterated-learning tradition (Kirby's
transmission-bottleneck chains; Griffiths & Kalish's convergence-to-prior theory), cultural transmission and accumulation in RL (Bhoopchand 2023, Cook 2024,
Tessler 2021), and the plasticity-loss and reset literature that grounds the aging-teacher model (Dohare 2024, Nikishin 2022), with evolutionary boundary
conditions (Rogers' paradox, critical social learning). H5 (growing bodies) descends from developmental robotics and morphological scaffolding (Bongard 2011,
Lungarella 2003, Badie 2025) grounded in infant motor development (Adolph's step-and-fall counts, Han & Adolph's fall-energy accounting). A cross-cutting
methodology literature (Agarwal's statistical precipice, Colas' test calibrations, Patterson's empirical-design guide, Taylor & Stone's transfer accounting)
governs how all five are measured.

---

## 1. The Instruction Spectrum: from Imitation to Reinforcement

The channel space in which H4's advice bottleneck lives: how prior experience, compressed into an artifact, is delivered to a learner — and when it accelerates
learning versus locking it in.

### What is known

- **Compounding error separates imitation from RL.** Behavioral cloning with per-step classification error epsilon incurs up to O(T^2 * epsilon) regret over
  horizon T, because the learner's own mistakes shift the state distribution off the demonstration manifold. DAgger (Ross, Gordon & Bagnell, AISTATS 2011)
  reduces interactive imitation to no-regret online learning, restoring O(T * epsilon)-order guarantees with a single stationary deterministic policy (validated
  on Super Tux Kart, Super Mario Bros, handwriting sequence labeling). Static demonstrations alone cannot be trusted at states the teacher never visited.
- **Demonstrations inside value-based RL buy a head start but can anchor.** DQfD (Hester et al., AAAI 2018) — a large-margin supervised loss on demonstrator
  actions plus 1-step and n-step TD losses, demo transitions kept permanently in prioritized replay — beat Prioritized Dueling Double DQN on 41 of 42 Atari
  games in the first million steps with only ~5.5k-75k human transitions per game; PDD DQN needed on average 83 million steps to catch up. But DQfD exceeded the
  best human demonstration on only 14 of 42 games: the persistent imitation loss anchors many runs near demonstrator level.
- **Annealing the teacher away enables surpassing the teacher.** Kickstarting (Schmitt et al. 2018, arXiv:1803.03835) adds an auxiliary policy-distillation
  cross-entropy from a trained teacher, with weight lambda annealed by population-based training. On DMLab-30 the kickstarted student reached teacher-level
  performance in roughly 10x fewer steps and ended 42% above the from-scratch baseline. The load-bearing mechanism is the annealing: because the imitation term
  is driven to zero, the student is not tethered to the teacher.
- **Instruction can live purely in the exploration distribution.** Jump-Start RL (Uchendu et al., ICML 2023): a guide policy rolls in for the first h steps (h
  receding as the student improves) while the student's objective stays pure RL, so there is no imitation term to cause lock-in. JSRL dominates in low-data
  regimes: antmaze-umaze with 1k offline transitions, IQL+JSRL 71.7 +/- 14.5 vs IQL 55.5 +/- 12.5; antmaze-medium-play with 10k, 86.7 +/- 3.7 vs 32.8 +/- 32.6;
  simulated instance grasping with just 20 demonstrations, QT-Opt+JSRL 0.54 +/- 0.02 success vs QT-Opt 0.29 +/- 0.20 — only the JSRL variant learns at all at 20
  demos.
- **Theory bounds when instruction helps.** For non-optimistic exploration (epsilon-greedy), sparse-reward sample complexity is exponential in horizon, and a
  guide policy covering the optimal policy's states reduces it to polynomial (tabular suboptimality O(C * H^{5/2} * S^{1/2} * A / T^{1/2}); Uchendu et al.
  2023). But Xie et al. (NeurIPS 2021) prove a lower bound Omega(H^3 * S * min{C*, A} / eps^2): in the worst case the optimal algorithm either does pure offline
  reduction on the reference policy or ignores it entirely. Instruction is worth an exponential factor to naive learners and can be worth approximately nothing
  to optimistic ones.
- **Demonstrations-as-data is a sound default channel.** Hybrid RL (Song et al., ICLR 2023): Q-learning on a mixed offline+online buffer (Hy-Q) is provably
  statistically and computationally efficient whenever offline data covers a high-quality policy (bounded bilinear rank), no optimism needed; it beats pure
  online, pure offline, and other hybrid baselines including on Montezuma's Revenge.
- **How prior knowledge is encoded in values determines whether it helps.** AWAC (Nair, Gupta, Dalal & Levine 2020): a constraint strong enough to make offline
  learning stable is typically too strong to permit online improvement unless implicit/adaptive. Lee et al. (CoRL 2021): state-action distribution shift at the
  offline-to-online handoff causes bootstrap error that "destroys the good initial policy," fixed by balanced replay plus a pessimistic Q-ensemble. Cal-QL
  (Nakamoto et al., NeurIPS 2023): conservative offline pretraining yields arbitrarily over-pessimistic Q-values, producing an initial dip and slow recovery;
  calibrating conservative values against a reference policy's returns (a one-line change) removes the dip and wins on 9/11 fine-tuning benchmarks.
- **Quality of the teacher caps the value of imitation, not of RL.** Kumar, Hong, Singh & Levine (ICLR 2022): offline RL beats BC on long-horizon sparse-reward
  tasks and on noisy/suboptimal data; RL on sufficiently noisy suboptimal data can beat BC on expert data on critical-state problems; BC is preferable mainly
  with near-expert data on short horizons. Hao et al. (ICML 2023) prove the Bayesian bandit counterpart: achievable improvement from demonstrations scales with
  the demonstrator's competence, via an informed-Thompson-sampling regret bound.
- **Human priors impose escapable ceilings; the transfer channel matters.** AlphaGo Zero (Silver et al., Nature 2017) trained tabula rasa beat the human-data-
  initialized AlphaGo Lee 100-0, its pure-RL curve crossing above the supervised counterpart within roughly a day. Schaal (NIPS 1997), at laptop scale: priming
  Q-values, value functions, or policies from a 30-second demonstration gave no significant speedup on pendulum swing-up, while priming a MODEL of the dynamics
  enabled one-trial pole balancing on a real anthropomorphic arm.
- **RLHF is the canonical staged imitation-to-RL pipeline, with a quantified failure mode.** Christiano et al. (NeurIPS 2017) trained novel behaviors (Hopper
  backflips) from preferences on <1% of interactions; InstructGPT (Ouyang et al., NeurIPS 2022) chains SFT -> reward model -> PPO with a KL tether, and the 1.3B
  RLHF model was preferred over 175B GPT-3. Gao, Schulman & Hilton (ICML 2023): as distance d = sqrt(KL) from the initial policy grows, gold reward follows
  d*(alpha - beta*d) for best-of-n and d*(alpha - beta*log d) for RL — proxy reward keeps rising while true performance peaks then degrades (Goodhart regime).
- **The anti-lock-in mechanisms share one design: teacher influence decays or is gated by the student's own estimates.** Q-filter BC loss applied only where the
  critic scores the demo action above the policy's (Nair et al., ICRA 2018 — order-of-magnitude speedup on multi-step block stacking; final policies beat the
  demonstrator); annealed distillation (kickstarting; QDagger in Reincarnating RL, Agarwal et al., NeurIPS 2022); receding roll-ins (JSRL); balanced replay (Lee
  et al. 2021); decaying reuse probability psi in pi-reuse (Fernandez & Veloso, AAMAS 2006); budgeted action advice, where the same budget spent at
  high-importance states (large Q-range) beats spending it early (Torrey & Taylor, AAMAS 2013).
- **A tabular equivalence collapses two channels into one.** Initializing Q-values with a potential function Phi is update-for-update identical to
  potential-based reward shaping with Phi (Wiewiora, JAIR 2003), and potential-based shaping is necessary and sufficient for policy invariance (Ng, Harada &
  Russell, ICML 1999); non-potential shaping can change the optimal policy — the classic failure being Randlov & Alstrom's (ICML 1998) bicycle agent riding in
  circles to harvest shaping reward.

### Implications for our design

- The organizing frame: instruction = a lossy compression of a prior generation's experience into an artifact (transitions / policy / value-potential / action
  advice / start states / proxy reward), delivered through a channel (replay data, auxiliary loss, initialization, exploration override, roll-in curriculum)
  with a trust schedule (persistent, annealed, performance-gated, budgeted). DQfD = transitions + persistent margin loss; kickstarting = policy + annealed CE
  loss; JSRL = start states + receding roll-ins; Q-init/shaping = value artifact that TD updates naturally overwrite; RLHF = proxy reward + persistent KL
  tether.
- The literature's central prediction: acceleration comes from the artifact reshaping the EXPLORATION distribution; lock-in comes from the artifact persisting
  in the OBJECTIVE. **Our EXP4 advice bottleneck (Q[s, a_advice] = 5.0 pretraining, then normal life) sits on the benign side of this split by design:** it is a
  value-artifact channel with no persistent loss term, so TD updates can overwrite bad advice — the tabular analog of an annealed trust schedule. DESIGN.md
  follows this recommendation implicitly.
- Deviation to note: the Q = 5.0 advice prime is action-dependent, hence NOT a potential-based initialization in Wiewiora's sense — it can transiently bias the
  policy toward advised actions, which is the intent, but the paper must present it as budgeted action advice via optimistic initialization, not as
  policy-invariant shaping. DESIGN.md also lacks the recommended optimism-matched control (e.g., the same number of random (s, a) pairs primed to 5.0), so
  "advice content helps" is currently confounded with "optimism at 100 entries helps."
- The recommended trust-schedule sweep (fixed vs annealed vs performance-gated imitation weight) and the six-channel arm comparison are NOT in DESIGN.md; EXP4
  runs a single channel (advice-prime) against weight-copy and long-life baselines. That is a deliberate scoping choice, but the paper should frame EXP4's
  conditions as points in this channel space rather than claim to have mapped it.
- The bits-transmitted vs steps-saved frontier (artifact description length vs interactions saved) is cheap to report given EXP4's <=100-pair advice cap and
  would make "compressed experience" quantitative; DESIGN.md exports the advice sets but does not currently commit to this figure.

### Validity risks

- Tabular Q-learning with persistent epsilon-greedy converges to optimal with probability 1 (GLIE), so lock-in cannot be asymptotic here — it must be defined as
  finite-budget regret or induced via decaying exploration. (EXP4's plasticity decay eps(age) = 0.4 * 2^(-age/5k) does exactly this; the paper must say so
  explicitly.)
- Q-initialization confounds information content with optimism level; without an optimism-matched control, any advice benefit may be plain optimistic
  initialization. Wiewiora's equivalence also means a Q-init arm and a potential-shaping arm are the same experiment — never present them as independent
  evidence.
- Several headline deep-RL phenomena (Cal-QL dip, balanced-replay failure, BC compounding error at its worst) are function-approximation artifacts and may not
  reproduce tabularly; absence of a dip is a scoping result, not a refutation.
- Base-learner dependence: instruction is worth the most to naive epsilon-greedy explorers and can be worth ~nothing to optimistic/count-bonus learners (Xie et
  al.); claims must be scoped to naive-exploration learners or an optimistic arm added.
- Ceiling/floor effects: worlds where pure Q-learning converges quickly make all arms tie; worlds where it never sees reward make comparisons degenerate.
- Teacher construction can leak experimenter knowledge if hand-designed; EXP4's endogenous teachers (the agent's own lived experience) avoid this.
- Non-potential follow-the-teacher reward bonuses study reward misspecification, not instruction (Ng et al. 1999; Randlov & Alstrom 1998); deterministic argmax
  tie-breaking over primed Q-tables can hard-code the advised path and masquerade as lock-in — randomize tie-breaking and log it.
- With bimodal outcomes (escape vs lock-in), mean +/- SEM comparisons mislead; pre-register P(lock-in)-style metrics with bootstrap CIs.

---

## 2. World Models and the Blindfold Test (H1)

### What is known

- **Latent learning is real and has an exact behavioral signature.** Blodgett (1929): rats exploring a 6-unit alley maze unrewarded showed an abrupt error drop
  to the always-rewarded control's level almost immediately after food was introduced. Tolman & Honzik (1930), the canonical 3-group protocol in a 14-unit
  T-maze (~17-22 days, one trial/day): HR (rewarded from day 1) improved steadily; HNR (never rewarded) barely improved; HNR-R (no reward days 1-10, food from
  day 11) collapsed its error curve within 1-2 days of first reward to match — in Tolman's plots even slightly beat — the always-rewarded group. Tolman (1948)
  synthesized this into the cognitive-map hypothesis: rats acquire map-like structural knowledge during unrewarded exploration, not stimulus-response chains.
- **The famous shortcut probe replicates poorly; latent-learning designs replicate well.** Tolman, Ritchie & Kalish (1946) reported 19/53 rats (36%) choosing
  the arm pointing at the goal in the novel sunburst array; Duvelle & Grieves (Eur. J. Neurosci. 2026) meta-analyzed 13 studies / 47 experiments and found
  shortcutting above chance in only 17% of experiments (32% favored arms adjacent to the trained route; 26% no preference). Thistlethwaite's (1951) review
  identified the key latent-learning confounds: amount/coverage of unrewarded pre-exposure and residual motivation during "unrewarded" trials.
- **Dead reckoning is a real biological competence with lawful error growth.** Mittelstaedt & Mittelstaedt (1980): gerbil mothers retrieve displaced pups in
  total darkness on a direct straight vector using idiothetic self-motion cues (slow platform rotation deflects the homing vector). Muller & Wehner (1988):
  desert ants home via approximate iterative path integration, with systematic errors that vary with outbound turn angle; Wehner & Srinivasan (1981): after
  running off its home vector, the ant switches to systematic search loops centered on the fictive nest. Loomis et al. (1993), the human blindfold protocol
  (triangle completion): congenitally blind, late blind, and blindfolded-sighted adults showed no significant differences in path-integration ability, and
  errors grew with path length/complexity; Etienne & Jeffery (2004): mammalian path integration accumulates error with distance and is reset by landmark fixes.
- **Dyna's sample-efficiency curves are the canonical calibration target.** Sutton (1990, 1991): planning is RL applied to simulated experience from a learned
  model. In the Dyna Maze (Sutton & Barto 2nd ed., Example 8.1: 9x6 grid, gamma 0.95, alpha 0.1, epsilon 0.1, optimal path 14 steps, 30 runs): n=0 planning
  steps (pure Q-learning) took ~25 episodes to the optimal path, n=5 took ~5, n=50 took ~3. The blocking- and shortcut-maze experiments (Section 8.3) show plain
  Dyna-Q keeps planning with stale transitions and never finds a newly opened shortcut, while Dyna-Q+ (bonus kappa * sqrt(tau)) adapts.
- **Blind rollouts through a learned model are exploitable — quantified.** Ha & Schmidhuber (2018): CarRacing-v0 score 906 +/- 21 (first past the 900
  threshold); a VizDoom policy trained entirely inside the dream transferred at 1092 +/- 556. Critically, at low dream temperature (tau=0.1) the policy
  exploited model flaws — ~2086 in the dream but only ~193 in reality; raising uncertainty (tau=1.15) fixed transfer. Janner et al. (MBPO, NeurIPS 2019):
  one-step model errors compound over rollout horizon; many short rollouts branched from real states reach model-free asymptotic performance with roughly 10x
  less data. Lambert, Pister & Calandra (2022) characterize compounding-error growth with horizon; Talvitie (AAAI 2017) shows self-correction on model-generated
  states predicts planning performance better than one-step error.
- **The model need not reconstruct observations, and imagination scales.** MuZero (Schrittwieser et al., Nature 2020) plans in a latent model predicting only
  reward, value, and policy: Atari-57 median human-normalized ~2041% at 200M frames (above R2D2's ~1921%), matching AlphaZero at Go/chess/shogi without given
  rules. DreamerV1 (ICLR 2020): ~823 average on 20 visual control tasks within 5x10^6 steps, surpassing D4PG's 786 at 10^8 steps; DreamerV2 first world-model
  agent at human level on 55-game Atari at 200M frames; DreamerV3 one fixed configuration across 150+ tasks, first to collect Minecraft diamonds from scratch.
- **Blind navigation competence alone does not require an explicit model.** Wijmans et al. (ICLR 2023 outstanding paper): agents whose ONLY sensing is egomotion
  reach ~95% PointGoal success in unseen environments; their LSTM memory spontaneously develops decodable metric maps, collision-detection neurons,
  wall-following, shortcut-taking, and selective forgetting. Banino et al. (Nature 2018): path-integration training pressure produces grid-cell-like units, and
  an RL agent using them outperformed an expert human and took novel shortcuts.
- **Fairness warning with numbers.** Van Hasselt, Hessel & Aslanides (NeurIPS 2019): experience replay is effectively a non-parametric model — in deterministic
  tabular worlds Dyna's learned model IS a replay buffer — and in a like-for-like Atari-100k comparison, data-efficient Rainbow (replay, more updates per step)
  matched/beat model-based SimPLe (Kaiser et al., ICLR 2020) with less compute. Claimed model-based sample-efficiency gains can be pure update-count artifacts.
- **Revaluation probes dissociate algorithm classes.** Momennejad et al. (Nature Human Behaviour 2017): humans are better at reward revaluation (goal value
  changed) than transition revaluation (map changed) — the successor-representation signature. Reward-move vs wall-change probes discriminate MF vs SR vs full
  model-based, since SR handles goal moves but fails structure changes.

### Implications for our design

- EXP1 follows the core recommendations: slip = 0.1 makes the blindfold test non-trivial (belief drifts, error compounds with distance, mirroring Loomis and
  Etienne & Jeffery); the bump/touch channel is manipulated deliberately (blind-A vs blind-A-touch, blind-B vs blind-B-touch) rather than left implicit — the
  prediction that bump feedback re-anchors belief at home but helps far less in a stranger's home is exactly the file's "publishable signature that the map is
  doing the work"; the stranger's-home condition (blind-B on A's model+Q) is the control that blind success depends on the learned map, per Wijmans generic
  blind competence is learnable; random-A gives the chance floor; sighted-B-transfer isolates layout knowledge (not vision) as the bottleneck; 30 seeds with
  belief entropy and trajectory exports.
- Layout matching: HOME_A/HOME_B share the same 13x11 shell and identical bed/ fridge coordinates, matching start-goal geometry by construction; wall density
  and branching-factor matching is not specified in DESIGN.md and should be verified (the sighted-B-transfer condition partially serves as the sighted control
  that both layouts are learnable).
- **Deviation (the central fairness risk):** DESIGN.md's claim "Dyna reaches 90% success in fewer steps than Q-learning" has no update-matched replay baseline.
  DynaQ(planning=20) performs 20 extra updates per real step; van Hasselt et al. predict a Q-learning + replay agent granted 20 replayed updates/step matches
  Dyna on raw sample-efficiency curves. Without this arm (or compute-matched curves alongside sample-matched ones), the sample-efficiency half of H1 is a claim
  about update counts, not world models. The honest claim is that the model's distinctive value appears where replay cannot go: simulating forward from a belief
  state under blindfold.
- Deviations of scope: no Tolman 3-group latent-learning arm (reward introduced mid-training), no revaluation probes (goal move / blocked path / opened
  shortcut, with Dyna-Q+), and no SR agent. These are the probes the literature says separate algorithm classes; the blindfold contrast alone separates "has a
  usable map of home" from "does not," which is H1's actual claim — the paper should scope its wording accordingly.
- The recommended plan-then-re-anchor-on-bump execution (tabular analog of MBPO's short branched rollouts) is what blind-A-touch implements via belief
  filtering; reporting success as a function of geodesic distance (the PI-literature plot) is cheap given the exported trajectories and worth adding.

### Validity risks

- Blindfold success alone cannot support a world-model claim (Wijmans: ~95% blind success with no explicit model); the home-vs-stranger contrast and touch
  manipulation must carry the inference.
- Ceiling effect: with deterministic dynamics and known start, dead reckoning is exact and any decent policy succeeds blind at home. (Mitigated: slip=0.1.)
- Compounding belief/rollout error can flip the sign: long open-loop plans through a noisy belief may underperform a reactive policy (Ha & Schmidhuber's ~2086
  dream vs ~193 reality; MBPO); report performance vs required horizon, not a single number.
- Update-count confound: without a replay-matched baseline, "Dyna is more sample efficient" is not a claim about world models at all.
- Hidden observation channels partially un-blind the agent: bump feedback, fixed start states, timeout signals, goal-reached resets all leak position; they must
  be documented and equalized across conditions. (DESIGN.md manipulates bump deliberately; the others need auditing.)
- Latent-learning confounds (Thistlethwaite): unequal pre-exposure coverage, and reward-structure leakage (per-step costs during "unrewarded" phases) — relevant
  if a latent-learning arm is added; also optimistic initialization partially mimics model-based behavior and must be identical and reported.
- Stale-model pathology: plain Dyna-Q is worse than no planning after environment change (never finds the opened shortcut); any change condition must include
  Dyna-Q+.
- Single dramatic probes are unreliable (sunburst meta-analysis: 17% of 47 experiments above chance vs the original's 36%-of-rats); rest conclusions on repeated
  probes, many seeds, IQM + stratified bootstrap CIs.
- Stranger's-home unfairness in the other direction: if HOME_B is harder (longer geodesics, more branching), worse blind performance reflects layout hardness,
  not map absence; verify with the sighted control.

---

## 3. Microtask Curricula (H2)

### What is known

- **Curriculum learning formalized.** Bengio, Louradour, Collobert & Weston (ICML 2009): easier examples first sped convergence and improved test generalization
  (shapes classification, growing-vocabulary language modeling); curricula framed as a continuation method smoothing a non-convex objective.
- **The canonical framework and the budget-accounting rule.** Narvekar et al. (JMLR 2020) decompose curriculum RL into task generation, task sequencing, and
  transfer, adopting Taylor & Stone's (JMLR 2009) metrics: time-to-threshold, asymptotic performance, jumpstart, total reward. Crucially they formalize "weak
  transfer" (drill time treated as sunk cost) vs "strong transfer" (curves offset by all source-task time, or the policy frozen while learning sources); most
  surveyed works only demonstrate weak transfer, and "achieving asymptotic performance improvements implies strong transfer."
- **The earliest RL curriculum won under strong accounting.** Selfridge, Sutton & Barto (IJCAI 1985), 1-D pole balancing: the hard (short, light) pole took 119
  failures to criterion directly; easy pole first (67 failures) then switching added only 6 more — 73 total vs 119, a win even counting source-task failures.
- **Direct soccer precedents.** Asada et al. (Machine Learning 1996), Learning from Easy Missions: a vision-based robot learned to shoot by initializing
  episodes close to the goal state and moving the start distribution progressively farther — the ancestor of shooting drills from near the goal. Narvekar,
  Sinapov, Leonetti & Stone (AAMAS 2016), simulated Half Field Offense (Sarsa + CMAC): auto-generated exactly our microtasks — a SHOOT task via near-goal
  "PromisingInitializations" and a DRIBBLE task via LinkSubTask + ActionSimplification; curricula of shoot and dribble drills significantly improved final
  goal-scoring rate vs learning 2v2 from scratch, with 25 trials and — critically — learning curves OFFSET by source-task time (strong-transfer accounting).
  Keepaway (Stone, Sutton & Kuhlmann 2005) and HFO itself were built as sub-games because full-game learning was intractable.
- **Start states alone are a powerful curriculum.** Florensa et al. (CoRL 2017), reverse curriculum generation: sample starts by short Brownian walks backward
  from the goal, adaptively keeping starts with success rate between 10% and 90%; solved manipulation tasks "not solvable by state-of-the-art RL methods" from
  the original start distribution. Kakade & Langford (ICML 2002, Conservative Policy Iteration) give the theory: improvement guarantees depend on how well the
  restart distribution covers a good policy's state distribution — value information propagates backward from the goal only after it is reached.
- **The main non-curriculum alternative for sparse goals.** Hindsight Experience Replay (Andrychowicz et al., NeurIPS 2017): relabeling failed episodes with
  achieved goals; vanilla DQN fails beyond ~13 bits in bit-flipping while DQN+HER solves up to 50 bits; DDPG+HER learned Fetch push/slide/pick-and-place from
  binary rewards where vanilla DDPG essentially fails.
- **Principled switching rules exist.** Teacher-Student Curriculum Learning (Matiisen et al. 2017): sample the subtask with the highest absolute learning-curve
  slope, revisiting tasks whose performance decays (combats forgetting); solved a Minecraft maze unsolvable by direct training, an order of magnitude faster
  than uniform subtask sampling. Automatic environment curricula: PAIRED (Dennis et al., NeurIPS 2020) maximizes regret to produce solvable-but-challenging
  mazes with better zero-shot transfer; POET (Wang et al. 2019) coevolves obstacle courses whose eventual solutions could NOT be found by direct optimization
  from scratch.
- **Temporal abstraction supports drill-then-compose.** Options (Sutton, Precup & Singh 1999) formalize treating shoot/dribble as reusable skills; Konidaris &
  Barto (NeurIPS 2009) grow chains of options BACKWARD from the goal in Pinball — skill-chaining agents beat flat learners, pre-learned options did best.
- **Human part-whole training conditions the prediction.** Naylor & Briggs (1963): part-task training is most effective when subtask complexity is high and
  inter-subtask organization is LOW; whole-task training wins for highly interdependent tasks. Fontana et al.'s meta-analysis (2009; 44 articles, 20 usable)
  found effect sizes generally consistent with this. Wightman & Lintern (Human Factors 1985) taxonomy — segmentation, fractionation, simplification; in their
  segmentation studies part-task training beat whole-task in 3 of 4, all 3 wins using BACKWARD chaining (train the final segment first). But Ash & Holding
  (1990): for a musical keyboard passage BOTH chaining methods beat whole-task practice, and FORWARD chaining beat backward — chaining direction is
  task-dependent.
- **Practice quality, not quantity.** Ericsson, Krampe & Tesch-Romer (1993): deliberate practice — effortful, teacher-designed, targeting weaknesses at the edge
  of ability — with the best Berlin violinists past 10,000 hours by age 20; caveat: Macnamara & Maitra's preregistered replication (2019) found practice hours
  did NOT separate the best from the good group. Duke, Simmons & Cash (2009): among 17 advanced pianists, next-day retention was UNRELATED to total practice
  time or trial counts and predicted by strategy (immediate error identification/correction, slowing problem segments). Chaffin & Imreh (2002): expert practice
  isolates difficult segments, drills them, then reassembles into progressively longer runs.
- **When curricula do NOT work.** Wu, Dyer & Neyshabur (ICLR 2021 oral): across thousands of orderings on CIFAR-10/100, curriculum ordering gave no benefit over
  random in the standard setting; benefits appeared only with LIMITED BUDGET or noisy labels. Curriculum advantages should be largest mid-budget and may wash
  out asymptotically.
- **Contextual interference warns against judging drills during drilling.** Shea & Morgan (1979): blocked practice looks better during acquisition; random/
  interleaved wins retention and transfer (full numbers in Section 4).

### Implications for our design

- **Strong-transfer accounting is the headline requirement and DESIGN.md complies:** budgets strictly matched, "every step an agent takes — drill, practice, or
  play — counts," 60k steps for every arm, budgets in environment steps (not episodes), and frozen greedy full-game probes from kickoff every 2k steps excluded
  from the budget identically across arms. This is exactly the Narvekar strong-transfer standard and the Taylor & Stone frozen-policy convention.
- Drills are same-MDP restrictions, as recommended: only the start-state distribution changes (shoot drill = spawn carrying in the attacking third; dribble
  drill = spawn left-half with ball at center), reward stays sparse goal-only — "no shaping contamination," which respects Ng et al. (1999) and avoids the
  Randlov & Alstrom failure mode by construction. Tabular transfer works because drilled states are exactly the full game's states.
- The drills-fixed arm (one fixed spawn per drill) is the blocked-practice ablation linking H2 to H3 — a design element the human literature (Shea & Morgan)
  motivates directly.
- **Interpretive note rather than deviation:** because EXP2's drills change only start states (and which phase is active), the "restart-distribution vs skill-
  decomposition" confound largely dissolves — the drills ARE a reverse-curriculum restart manipulation in the Asada/Florensa/Kakade-Langford sense. The paper
  should attribute any drills-varied win to the restart-distribution mechanism, not to "task decomposition" per se; a separate start-state-only control arm
  would be needed to distinguish those, and DESIGN.md does not include one.
- Deviations: single budget (60k) with no 2-4x budget replication — Wu et al. predict the effect is budget-regime-dependent, so time-to-threshold and asymptotic
  claims should be separated and the chosen-budget risk acknowledged; no drill-order factor (the shoot-first ordering is backward chaining, supported by
  Wightman & Lintern and reverse curricula, but Ash & Holding show forward can win); no adaptive TSCL-style switch (fixed 20/20/60 split); no HER baseline;
  whether epsilon/learning-rate schedules key to the global step counter is not stated in DESIGN.md and must be pinned down (per-episode or per-phase clocks
  silently gift the drill arm a different exploration schedule).
- Instrumentation worth adding cheaply: count of above-init Q-entries and their spatial spread over time (drills should seed value backward from the goal
  earlier), first-goal time in full games, and post-training re-probes of the drill tasks to detect forgetting.

### Validity risks

- Weak-transfer accounting invalidates the headline claim; every drill transition must be charged to the shared budget. (DESIGN.md complies.)
- Per-episode confounds: drill episodes are shorter, so per-episode epsilon/lr decay or per-episode budgets give the drill arm more resets, more terminal
  rewards, and a different effective exploration schedule; all schedules must key to global steps. (Unspecified in DESIGN.md — audit.)
- Start-state vs skill-decomposition confound: without a full-game-with-near-goal- resets control, drill benefits cannot be attributed to microtask practice per
  se. (EXP2 should claim the restart mechanism.)
- Tabular representation mismatch: drills that altered the encoding would drill states the full game never visits; EXP2 avoids this by same-MDP drills.
- Budget-regime dependence (Wu et al.): too-large budgets erase the advantage, too-small budgets teach nothing; report multiple budgets or scope the claim.
- Catastrophic forgetting across phases: full-game training can erode drilled skills (TSCL's motivation); fixed switch points risk under/over-training — probe
  drill tasks after later phases.
- Acquisition-performance illusion: reward earned during drilling is inflated (denser terminal events); only frozen full-game-from-kickoff probes are valid.
  (DESIGN.md complies.)
- Shaping-reward hacking does not apply (no auxiliary rewards), but must stay true through implementation.
- Curriculum hyperparameter unfairness: the drill arm has extra knobs (split, order, phase lengths); tuning them while leaving `whole` untuned counts
  curriculum-generation cost as free — sweep both or report the full sweep.
- Task-organization limits of the analogy: a soccer possession is sequentially interdependent (high organization), where Naylor & Briggs favor whole or
  progressive-part training; EXP2's 60% full-game phase is the re-integration phase that keeps the design on the right side of this.
- Eval contamination and seed hygiene: eval episodes excluded identically (done); >=25-30 paired seeds (30, done); sparse binary outcomes need distributional
  statistics.

---

## 4. Variation Practice and Contextual Interference (H3)

### What is known

- **Shea & Morgan (1979), the target result, in full.** 72 students, 3 variants of a speeded barrier-knockdown task, 54 acquisition trials (18/variant), blocked
  vs random (max 2 consecutive repeats, Latin-square counterbalanced). Acquisition: blocked BETTER — mean total time 1.32 s vs 1.69 s, F(1,68)=45.61, and fewer
  than half as many errors (4.42 vs 9.13); the gap was largest on the first block (~1.45 vs ~2.55 s) and nearly closed by the final block (~1.2 vs ~1.3 s).
  Retention (10 min or 10 days, tested blocked or random): the ordering REVERSED — random-acquisition 1.31 s vs blocked-acquisition 1.73 s, F(1,64)=49.97;
  errors 2.10 vs 4.33. The worst cell was blocked-trained tested under the changed (random) context (~2.1-2.2 s); at 10 days blocked-blocked was actually
  slightly FASTER than random-random — the varied advantage appears under changed test contexts, not same-context testing. Transfer to two novel barrier orders:
  random faster on both (1.58 vs 1.73 s, F(1,64)=6.93), "most notable" on the harder 5-barrier task (1.84 vs 2.04 s). No forgetting over 10 days in either
  group.
- **Schema theory and the bracketing result.** Schmidt (1975): learners acquire a generalized motor program plus a schema mapping parameters to outcomes;
  variability of practice predicts better transfer to novel parameter values. Van Rossum (1990) reviewed 73 experiments (1975-1987): support mixed in adults,
  stronger in children. Kerr & Booth (1978): 64 children practicing a beanbag toss; varied groups practicing at bracketing distances that NEVER included the
  criterion (2 & 4 ft; 3 & 5 ft) were significantly MORE accurate at the criterion distance than groups that practiced only at the criterion itself — the
  signature schema-theory result.
- **Desirable difficulties and the challenge point.** Bjork (1994; Bjork & Bjork 2011): conditions that impair acquisition (spacing, interleaving, variation,
  retrieval practice) often enhance retention and transfer, because current performance (retrieval strength) dissociates from learning (storage strength);
  difficulties are desirable only if the learner can still succeed. Guadagnoli & Lee (2004): optimal practice difficulty rises with skill level and falls with
  task difficulty.
- **Boundary conditions — including for our exact perturbation family.** Magill & Hall (1990): the CI effect is most reliable when variants require different
  generalized motor programs, weaker/inconsistent for parameter-only modifications (speed/timing). Czyz et al. (2024; 54 studies, 2,068 participants, 194 effect
  sizes): high-CI advantage at delayed retention SMD = 0.63 (95% CI 0.33-0.93; 0.43 after outlier removal) — large in laboratory tasks (SMD = 0.92, CI
  0.48-1.36) but negligible and non-significant in applied/sport tasks (SMD = 0.23, CI -0.16-0.62, p = .24). Ammar et al. (2023, "The myth of contextual
  interference learning benefit in sports practice") likewise found no CI benefit in sports settings.
- **Music evidence is split and directly cautionary.** Carter & Grahn (2016): 10 advanced clarinetists, blocked (12 min/piece) vs interleaved (3-min
  alternations); wherever ratings differed, interleaved-practiced pieces were rated better. But Caramiaux et al. (2018) — the closest analog of our paradigm:
  nonmusicians practicing an 8-note piano sequence under small vs large tempo-variability and randomized vs non-randomized schedules — LOWER temporal
  variability and NON-randomized schedules produced better transfer of movement smoothness, and timing accuracy was unaffected by practice condition: an
  outcome-dependent dissociation where the tempo-variability manipulation reversed the textbook prediction.
- **The cross-domain synthesis.** Raviv, Lupyan & Green (2022): across ~80 years of motor learning, categorization, perception, language, and ML —
  low-variability input is learned fast but generalizes poorly; high-variability input is learned more slowly but generalizes better.
- **The ML operationalization.** Domain randomization (Tobin et al., IROS 2017): an object detector trained entirely in randomized non-realistic simulation
  reached ~1.5 cm real-world accuracy with no real images. CoinRun (Cobbe et al., ICML 2019): with 100 training levels, ~99.5% train solved but only ~67% on
  unseen test levels (>30-point gap); as training levels grow, TRAIN performance falls while TEST performance rises (at 16k: ~90% train vs ~86-87% test) — an
  explicit train-worse/test-better crossover in practice diversity. "Substantial overfitting occurs when there are less than 4,000 training levels"; on a fixed
  500-level set, test plateaued near ~69% while train hit ~99%. Regularizers (L2 w=1e-4, dropout 0.1, Cutout-style augmentation, batch norm) each shrank the
  gap, and injected stochasticity (epsilon-greedy, entropy bonus) also improved generalization. Procgen (Cobbe et al., ICML 2020) extends this to 16
  environments: generalization requires hundreds to thousands of level variants.
- **Overfitting-to-the-practiced-configuration is the small-RL default.** Farebrother, Machado & Bowling (2018): DQN trained on one Atari flavor overspecializes
  — zero-shot performance on other flavors DECLINES as source training progresses, near zero on some variants; L2/dropout stabilize cross-flavor performance,
  and regularized pretraining + fine-tuning beat scratch (25.4 vs 7.5 on Freeway m1d0). Zhang et al. (2018): gridworld agents memorize finite maze sets, and
  sticky actions/random starts often fail to prevent it.
- **The replication target pattern** (synthesized from Shea & Morgan and CoinRun): (a) acquisition deficit — blocked's own-conditions curve lies ABOVE varied,
  largest gap early; (b) novel-test advantage — varied wins on never-experienced perturbations; (c) context specificity — under the exact practiced condition,
  blocked equal or better; the blocked deficit GROWS with degree of contextual change and difficulty; (d) dose-response — test performance rises and train
  performance falls monotonically with number of practiced variants.

### Implications for our design

- **EXP3 implements the true Shea & Morgan manipulation, avoiding the construct-mismatch trap:** both arms practice the same three passages and only the
  SCHEDULE differs (blocked = A then B then C; interleaved = uniform sampling). This is the CI comparison proper (varied-blocked vs varied-random), not the Kerr
  & Booth constant-vs-varied comparison. Note the flip side: DESIGN.md has no constant-practice arm, so H3 tests scheduling, not Schmidt's
  variability-of-practice hypothesis — scope the claims to CI.
- The shared-representation requirement is engineered, as the file demands: linear Q with phi = onehot(motif, pos-in-motif) ++ onehot(passage) ++
  onehot(position), so interference lives in the shared motif features and transfer to a novel passage (new arrangement of the same motifs) is mechanically
  possible. The per-passage exception positions supply the context-dependent component that Magill & Hall say drives reliable CI. Reporting a
  coverage/feature-overlap mechanism variable would strengthen this further.
- Budgets equal (same total episodes), 40 seeds, and the acquisition curve is recorded as the "fool's progress" curve — matching the recommendation to never
  judge arms by training-phase performance.
- The retention test (frozen scores on A, B, C after training) measures interference from later learning rather than decay — the right in-silico analog, since
  frozen linear-Q does not forget; DESIGN.md's prediction that interleaved wins retention "especially passages A, B after C was drilled" embraces the recency
  structure of blocked practice.
- **Deviations:** blocked order is fixed (A, B, C) with no Latin-square counterbalancing, so part of blocked's retention deficit is an order/recency artifact by
  construction — either counterbalance across seeds or report performance as a function of time-since-last-practice; there is no matched action-noise control
  (Cobbe: stochasticity alone improves generalization, so the interleaving benefit needs to exceed a noise control to be attributed to variation structure); no
  dose-response sweep over number of variants; the test battery has one novel passage rather than the 2x4 interpolation/extrapolation grid; a single motif
  library / passage set rather than several replication units; and the four-part replication criterion (acquisition deficit, novel-test advantage, crossover
  interaction, context specificity) is implicit in the predictions but not pre-registered as a conjunctive criterion.
- Directional honesty: the perturbation family closest to ours (timing/sequence parameters, piano-like task) is exactly where the human effect is least reliable
  (Magill & Hall; Caramiaux's reversal); pre-commit to publishing the direction found.

### Validity risks

- Construct mismatch (claiming Shea & Morgan while running Kerr & Booth) — EXP3 avoids the classic form, but must not claim to test variability-of-practice
  without a constant-practice arm.
- Tabular/feature representation can trivialize the result in either direction: fully disjoint states put both arms at floor on the novel passage; full aliasing
  makes variation free and erases the acquisition deficit. Report coverage statistics; claim phenomenon-level, not mechanism-level, replication (in-silico
  mechanism = shared-parameter interference and coverage, human mechanism = elaborative/reconstructive processing).
- Exploration confound: interleaving injects effective stochasticity; without a matched-noise control the benefit could be re-described as better exploration.
- Directional risk: the closest human analog (Caramiaux 2018) found LOWER variability and NON-random schedules transferred better on one outcome measure; a null
  or reversed result is a live possibility.
- Budget dependence can manufacture or hide the crossover (too short: varied never acquires — Guadagnoli & Lee; too long: acquisition curves converge); a single
  fixed budget is a validity threat.
- Ceiling/floor at test: perturbation magnitudes need piloting; the varied advantage decays outside the practiced range (interpolation vs extrapolation).
- Recency artifact: the last-practiced passage dominates recent updates; without counterbalancing, blocked's test score is biased by an arbitrary ordering.
- Retention overreach: frozen Q does not forget, so the delayed-retention half of Shea & Morgan has no default analog; note also that at 10 days blocked-blocked
  beat random-random — if the varied arm wins even on the exact practiced, unperturbed test, suspect a confound rather than a replication.
- External validity: the human effect is lab-large (SMD 0.92) and sport-null (SMD 0.23 n.s.); scope conclusions to laboratory-task-like settings, not "how to
  practice piano."
- Forking paths: many cells (arms x tests x measures); pre-register the conjunctive replication criterion, use paired seeds, report all cells, and check the
  crossover across multiple outcome measures (score, errors, steps) since measure-specific replication is a known failure mode.

---

## 5. Generational Teaching via a Distillation Bottleneck (H4)

The channel taxonomy for single-handoff instruction (and the evolutionary boundary conditions: Rogers' paradox, critical social learning) is in Section 1. This
section covers what is specific to ITERATED transmission: distillation lineages, the iterated-learning tradition, cultural transmission and accumulation in RL,
and the plasticity-loss literature that motivates aging teachers.

### What is known

- **Distilled students can exceed their teachers.** Policy distillation (Rusu et al., ICLR 2016) transferred DQN policies on Atari to dramatically smaller
  networks at expert performance, and a multi-task distilled agent OUTPERFORMED its single-task DQN teachers and jointly-trained DQN. Born-Again Networks
  (Furlanello et al., ICML 2018): students with IDENTICAL capacity, trained on teacher soft labels in repeated generations (each student becomes the next
  teacher), significantly outperform their teachers — DenseNet BANs reached 15.5% validation error on CIFAR-100 (then state of the art) and 3.5% on CIFAR-10 —
  with gains arriving in the first few generations and then saturating, even without any teacher aging.
- **Imitate-then-surpass, in RL.** Kickstarting (Schmitt et al. 2018): an annealed auxiliary distillation loss lets a fresh student reach teacher level ~10x
  faster on DMLab-30 and end 42% ABOVE the best teacher — the result EXP4's per-generation student phase replicates in miniature. Reincarnating RL (Agarwal et
  al., NeurIPS 2022) formalizes iterated reuse of prior computation (policy-to-value transfer of a suboptimal teacher via Dagger-style distillation) on Atari,
  locomotion, and balloon navigation — but it transfers the FULL teacher policy/data, with no curated bottleneck and no teacher aging.
- **The transmission bottleneck creates structure — the iterated-learning result.** Kirby, Cornish & Smith (PNAS 2008): in 10-generation human diffusion chains
  learning artificial languages through a bottleneck (each learner sees only a subset of meaning-string pairs), transmission error DECREASED and structure
  INCREASED cumulatively without any participant intending it. Experiment 1 produced degenerate, massively homonymous but highly learnable languages; Experiment
  2, filtering homonyms out of training data (an expressivity pressure), produced compositional structure. Bottleneck alone yields learnable-but-degenerate
  systems; bottleneck + expressivity pressure yields structured ones.
- **Iterated-learning theory predicts convergence to the learner's prior.** Griffiths & Kalish (Cognitive Science 2007): for learners who sample from their
  posterior, the iterated-learning Markov chain's stationary distribution IS the learner's prior — transmitted data's influence washes out, and bottleneck size
  changes convergence RATE, not the endpoint; with MAP-like learners even weak priors are AMPLIFIED by transmission (Kirby, Dowman & Griffiths, PNAS 2007;
  framework in Smith, Kirby & Brighton 2003). Prediction for EXP4: late-generation tabular policies should increasingly reflect student inductive biases
  (initialization, tie-breaking, discounting), more strongly at tighter bottlenecks.
- **Compressibility x expressivity is the general law.** Kirby, Tamariz, Cornish & Smith (Cognition 2015): linguistic structure emerges only under joint
  pressure for compressibility (learnability, from the bottleneck) and expressivity (from use); transmission-only chains collapse to degenerate languages. RL
  mapping: imitation-only generational chains should collapse to a few habitual paths; the student's post-imitation reward-driven exploration is the
  expressivity pressure that prevents collapse.
- **The mechanism transfers to neural learners.** Neural iterated learning (Ren, Guo, Labeau, Cohen & Kirby, ICLR 2020): resetting neural agents every
  generation plus limited-data transmission raises compositionality of emergent languages, with a learning-speed advantage for structured mappings that
  AMPLIFIES over generations; iterated learning also recovers systematic program layouts in VQA (Vani, Schwarzer, Lu, Dhekane & Courville, ICLR 2021).
- **Cultural transmission in deep RL is one step, not a chain.** Bhoopchand et al. (Nature Communications 14:7536, 2023; arXiv 2022): MEDAL ingredients (Memory,
  Expert Dropout, Attention Loss, plus automatic domain randomization) trained GoalCycle3D agents that imitate experts few-shot in real time, RETAIN the
  strategy after the expert drops out, and imitate human demonstrators zero-shot without pre-collected human data — but it studies one expert-to-novice step,
  transmitted by live co-presence, not curated trajectories.
- **Cultural ACCUMULATION beats one long life at matched experience.** Cook, Lu, Hughes, Leibo & Foerster (NeurIPS 2024): the first general models of emergent
  cultural accumulation in RL, under episodic (in-context) and train-time (in-weights) generations; generational agents balancing social learning with
  independent exploration OUTPERFORM agents trained for a single lifetime with the SAME cumulative experience. Closest deep-RL neighbor to EXP4; transmission is
  via observing predecessors in-environment, and there is no teacher aging model.
- **Humans do it with distilled messages.** Tessler et al. (2021, CogSci): chains of participants given only 2 lives each in minimalist video games, passing
  free-form written messages (dynamics, goals, risks, strategies), accumulated knowledge such that multigenerational performance tracked individuals with
  unlimited lives — transmitted "principles" substitute for direct experience.
- **Plasticity loss is real, and it is a deep-network phenomenon.** Dohare et al. (Nature 632:768-774, 2024): in continual supervised streams (Continual
  ImageNet), accuracy fell from 89% on early tasks to ~77% — about linear-network level — by task 2000 across architectures/optimizers; standard deep RL (PPO on
  ant locomotion with changing friction) also loses the ability to learn; continual backpropagation (reinitializing a small fraction of low-utility units)
  maintains plasticity indefinitely, and L2 / shrink-and-perturb substantially mitigate. This grounds EXP4's aging-teacher model — but the mechanism is
  representational and does NOT arise intrinsically in tabular learners.
- **Resets are the unfiltered-channel comparison.** Primacy bias (Nikishin, Schwarzer, D'Oro, Bacon & Courville, ICML 2022): deep RL agents overfit early
  interactions; periodically REINITIALIZING (parts of) the network while RETAINING the full replay buffer consistently improves SAC on DeepMind Control and SPR
  on Atari 100k. The reset channel is the ENTIRE experience buffer — an unfiltered transmission channel — whereas EXP4's bottleneck is curated and
  capacity-limited. Related deep-net pathologies, none involving a second agent, teaching, or data selection: capacity loss mitigated by InFeR (Lyle, Rowland &
  Dabney, ICLR 2022), plasticity loss across Atari game sequences mitigated by CReLU (Abbas et al., CoLLAs 2023), dormant neurons recycled by ReDo (Sokar et
  al., ICML 2023), plasticity injection (Nikishin et al., NeurIPS 2023).
- **Warm-starting hurts; fresh students are justified.** Ash & Adams (NeurIPS 2020): networks initialized from a previously trained solution generalize
  measurably worse than fresh random initializations given identical data; shrink-and-perturb closes the gap. The deep-learning analog of "inherited rigidity,"
  and the argument for EXP4's fresh students over copying the aged teacher's table.
- **Distillation chains are predicted to be non-monotonic.** Self-distillation theory (Mobahi, Farajtabar & Bartlett, NeurIPS 2020): in Hilbert-space
  regression, each self-distillation round progressively sparsifies the solution basis — amplified regularization — so a FEW rounds improve generalization but
  continued rounds provably underfit and degrade. Together with BAN saturation, this predicts an inverted-U performance curve over generations for
  imitation-heavy chains.
- **Distill-to-fresh-network precedents within one lineage.** ITER (Igl et al., ICLR 2021): transient non-stationarity early in RL training permanently damages
  representations and generalization; periodically distilling the current policy into a freshly initialized network improves ProcGen and Multiroom — the closest
  distill-to-fresh RL precedent, but it distills the full policy (no trajectory bottleneck) to repair deep-net damage, not to study transmission. Expert
  Iteration (Anthony, Tian & Barber, NeurIPS 2017): alternating imitation of a search-amplified expert with further improvement, tabula rasa, beat MoHex 1.0 at
  Hex. Self-Imitation Learning (Oh, Guo, Singh & Lee, ICML 2018): imitating only one's own high-return past trajectories — a within-lifetime top-K bottleneck —
  improves exploration-heavy Atari tasks.
- **Selecting what to teach matters.** Machine teaching for sequential decisions (Cakmak & Lopes, AAAI 2012): algorithms that SELECT maximally informative
  demonstration sets for IRL learners outperform arbitrary demonstration sets — precedent and baseline source for EXP4's top-episode selection rule.
- **Prior-art verdict and the null model.** Across targeted searches (iterated/generational distillation, tabular Q-learning generations, plasticity +
  teacher-student, transmission bottlenecks in RL), the review found NO published work that iterates distillation from an AGING (plasticity-decaying) teacher
  into fresh TABULAR students through a curated trajectory bottleneck — the pieces exist separately (ITER/BAN fresh-network distillation; Cook 2024 generational
  accumulation; Kirby/Griffiths bottleneck theory; Dohare 2024 aging) but their conjunction appears novel. The relevant null model is Structural Drift
  (Crutchfield & Whalen 2010): chains of learners resampling from finite data drift and can be absorbed into degenerate states — cumulative improvement is not
  guaranteed.

### Implications for our design

- **EXP4 already implements the core lineage the literature calls for:** 5 generations x 15k steps; a curated, capacity-limited artifact (deduplicated (s, a)
  pairs from the top-3 episodes by return, capped at 100) re-compressed each generation; fresh students; and an explicit, parameterized aging model (lr(age) =
  0.3 * 2^(-age/5k), eps(age) = 0.4 * 2^(-age/5k)). Per the recommendation, the paper must present aging as a model INSPIRED by deep-net plasticity loss (Dohare
  et al. 2024), not the phenomenon itself — tabular learners have no intrinsic plasticity loss, which is exactly what makes the testbed clean (bottleneck
  effects separate from the representation-repair effects that motivate resets).
- Baseline mapping to the recommended battery: `weight-copy` = full transfer (the Ash & Adams warm-start analog); `no-inheritance` = tabula-rasa no-teaching
  control (proves teaching matters); `one-long-life` / `one-long-life-slow` = the aging-cost and cumulative-experience-matched single-lifetime controls (Cook et
  al.'s comparison). **Missing: the reset-analog baseline — fresh Q-table plus replay of the teacher's ENTIRE transition history (the Nikishin-style unfiltered
  channel).** The file calls this "the comparison that answers 'how is this different from resets'" and the single most likely reviewer objection; also missing
  is a teacher-replays-own-top-K-to-itself condition to separate the generational fresh-student step from within-lifetime self-imitation (Oh et al. 2018).
- Bottleneck size K is fixed in DESIGN.md (cap 100); the literature says make K the PRIMARY manipulated variable (e.g., K in {1, 2, 5, 10, 25, 100}
  trajectories), predicting an inverted-U: too tight collapses coverage (Kirby 2008 Exp 1 degeneracy), too wide transmits the aged teacher's noise and reduces
  to the reset baseline. A K sweep is cheap and would be the headline figure.
- Measure iterated-learning SIGNATURES, not just return: (i) learnability — student episodes to reach 90% of teacher return, the analog of Kirby's transmission
  error, predicted to DECREASE across generations; (ii) policy compressibility — action entropy, number of distinct greedy trajectories; (iii) transmission
  fidelity — student-teacher greedy-action agreement on demonstrated vs undemonstrated states; (iv) state coverage (degeneracy detector); (v) innovation —
  return above best ancestor (the ratchet metric). DESIGN.md's metrics (greedy return, P(big goal), best-remembered-vs-greedy gap, exported advice trajectories)
  support (v) and part of (iii); adopting learnability and fidelity would make the iterated-learning connection testable rather than decorative.
- **Design tension to resolve: TrapGrid is deterministic with a single start.** The file warns that in deterministic single-start gridworlds all top-K
  trajectories become identical by generation 2, the bottleneck degenerates to K=1, and compression/learnability claims become trivial. EXP4's dedup cap
  partially masks this; adding slip (0.1-0.2) or multiple start states — or explicitly scoping compression claims out — is required before compression-style
  claims are made.
- Distillation mechanics: the file prefers replaying demonstrated transitions as Q-learning updates (possibly elevated alpha) or advantage-weighted cloning over
  hard-setting Q(s, a); DESIGN.md hard-primes Q[s, a_advice] = 5.0 — simpler and overwritable, but fidelity on demonstrated pairs must be logged so "what got
  through the bottleneck" is measurable and separable from what the student invents. The Griffiths-Kalish prior-amplification test (vary student initialization
  / tie-breaking / discount and show late-generation policies track student bias more at tighter bottlenecks) is a cheap, novel, citable add-on.
- Scale: 5 generations x 30 seeds meets the recommended floor (5-10 generations, 30+ chains); report PER-CHAIN trajectories and variance, not only means — drift
  and chain divergence are part of the phenomenon (Structural Drift). Ablating the selection rule (top-K by return vs uniform-random K vs coverage-maximizing K,
  per Cakmak & Lopes) would show selection matters beyond quantity.
- Prediction framing: DESIGN.md predicts a monotone upward ratchet; Mobahi/BAN predict possible non-monotonicity for imitation-heavy chains. EXP4's students
  live full RL lifetimes after the prime — strong expressivity pressure, exactly the condition Kirby 2015 says prevents collapse — so a monotone ratchet is
  defensible, but the per-generation curves should be inspected for a peak, and a generalization probe each generation (shifted goal/layout) is where
  "principles not weights" should show up (ITER/BAN/kickstarting all predict bottleneck-distilled lineages generalize better than full-copy lineages).

### Validity risks

- **Fresh-optimizer confound (the "isn't this just resets?" objection):** the student's advantage may come entirely from its fresh high learning rate and
  epsilon (restored plasticity), not from distilled content. Without the reset-with-full-replay baseline and the no-teaching control, "transmission bottleneck
  helps" is not identifiable. (`no-inheritance` exists; the full-replay arm does not — add it.)
- Trivial-aging objection: if aging = learning-rate decay in a tabular learner, resetting alpha trivially cures it; the contribution cannot be "we fix aging"
  (resets do that) and must rest on bottleneck-specific predictions — learnability trends, compression, prior amplification, generalization — as primary
  endpoints.
- Degenerate-collapse triviality: deterministic single-start TrapGrid makes increased compressibility across generations true by construction; stochastic
  dynamics / multiple starts are needed for the bottleneck to impose genuine generalization pressure.
- Convergence-to-prior downside (Griffiths & Kalish; Structural Drift): iterated transmission can wash OUT accumulated knowledge — chains may drift toward
  student inductive biases and lose hard-won discoveries; cumulative improvement is NOT guaranteed, so per-chain failures must be reported, not averaged away.
- Self-imitation confound: imitating high-return trajectories helps WITHIN one lifetime (Oh et al. 2018); a teacher-self-replay condition is needed to show the
  fresh-student step adds anything beyond top-K replay per se.
- Expressivity is built into RL: unlike language chains, students receive task reward, so "bottleneck yields structure" is confounded with "reward re-teaches
  everything"; only an imitation-only (no-exploration) arm cleanly isolates transmission dynamics, and results may differ qualitatively between arms.
- Mechanism-transfer overreach: BAN dark knowledge, kickstarting, and ITER gains are entangled with deep-net representation learning (soft labels,
  non-stationarity damage) with no tabular analog; frame the claim as testing the representation-INDEPENDENT information/selection mechanism from
  cultural-evolution theory, and cite Dohare/Lyle/Nikishin as motivation-by-analogy, never as phenomena reproduced in our agents.
- Underpowered signatures: Kirby 2008's trends emerged over 10 generations with high chain variance; with few generations and seeds the signature trends are
  indistinguishable from noise — EXP4's 5 generations x 30 seeds is the floor, and the condition grid grows fast, so pre-register primary comparisons.
- Selection-rule brittleness: top-K-by-return under stochastic dynamics preferentially transmits lucky trajectories (return noise, not policy quality),
  potentially teaching risk-seeking or non-reproducible behavior; select by mean return over repeated rollouts or by advantage, and report sensitivity to the
  rule. (Deterministic TrapGrid hides this; it surfaces the moment slip is added.)
- Carried over from Section 1 and still binding on EXP4: the Q = 5.0 advice prime needs an optimism-matched control (random pairs primed identically); argmax
  tie-breaking must be randomized; and lock-in claims must be scoped to finite budgets under decaying exploration (GLIE caveat).
---

## 6. Growing Bodies: Morphological Curricula (H5)

### What is known

- **The founding result — with an important caveat.** Bongard (2011, PNAS): ~5,000 simulated robots on phototaxis-without-tipping, body plans changing from
  anguilliform through reptilian to mammalian; robots whose bodies changed found successful controllers faster AND produced gaits more robust to being "knocked
  with a stick" than robots trained in the final upright body from the start. Proposed mechanism: early sprawled bodies are statically stable ("can't fall
  over"), so search solves locomotion first, balance later. Caveat: this is evolutionary search over controllers, not within-lifetime RL — transfer of the
  effect to TD learning is an open question our experiment actually tests.
- **The empirical regime being modeled: massive-volume, cheap-error practice.** Adolph et al. (2012; N=151 infants, 12-19 months): walking infants averaged
  2,367.6 steps/hour, 701.2 m/hour, and fell 17.4 times/hour — extrapolated to ~14,000 steps, ~46 football fields, and ~100 falls per 6-hour day. Novice
  12-month walkers fell 31.5 times/hour vs 17.4 for expert same-age crawlers, but per unit activity (time in motion, steps, or distance per fall) rates did NOT
  differ — the risky upright posture did not raise cost per unit of practice. New walkers gained speed/distance (296.9 vs 100.4 m/hour) without higher falls per
  unit activity.
- **Falls are cheap for small bodies — quantified.** Han & Adolph (2021; 563 spontaneous falls from 138 infants): 90.76% of falls uneventful; fussing/crying
  after only 4.26%; caregiver concern after 7.64%; median return to play ~1.84 s (90.7% of falls <3 s); ~2% ever required medical attention. Impact energy of
  real infant falls: M = 46.55 J vs M = 857.93 J if the same infants had adult size and ~3x walking speed — an 18.4x cost ratio, reproduced analytically (PE at
  CoM ~0.55 * height + KE at walking speed gives 18.5x; pure isometric scaling E proportional to m*g*h ~ L^4 gives ~21x). Their conclusion: infants "do not
  treat falling as an aversive penalty to avoid." This is the empirical license for penalty ~ m*g*h_CoM scaling.
- **But balance knowledge may not transfer across postures.** Adolph (2000): 9-month-olds avoided risky gaps in their experienced sitting posture but fell into
  the same gaps when tested crawling; crawl-learned slope avoidance must be relearned when walking. This cuts AGAINST an assumed balance-first benefit;
  developmental evidence permits either outcome.
- **Freeze-then-free, and its complication.** Bernstein (1967): learners freeze degrees of freedom, then release them; Vereijken et al. (1992) gave the classic
  human evidence. Berthouze & Lungarella (2004), on a humanoid learning to swing: fewer initial DOFs enabled more efficient exploration, BUT a single monotonic
  freeze-then-free episode was NOT sufficient under nonlinear body-environment coupling — ALTERNATING freezing and freeing was required. A monotonic growth
  schedule is one point in schedule space and may be suboptimal.
- **The field's honest summary is "mixed."** Lungarella, Metta, Pfeifer & Sandini (2003) articulate the rationale (immaturity as dimensionality-reducing
  scaffold); Naya-Varela, Faina & Duro (2023) — the closest prior empirical study — tested five morphological-development strategies on a simulated NAO learning
  bipedal walking via neuroevolution: development is beneficial in some setups, irrelevant or DETRIMENTAL in others; benefit requires "a suitable synergy among
  morphological development strategy, controller, task, and learning algorithm."
- **The strongest recent positive result — and what it says about scaling.** Badie et al. (2025, Communications Engineering): a 22-muscle bipedal humanoid
  trained with MPO; only the "ontogenetic double curriculum" (true child anthropometry 4yo -> 12yo -> adult PLUS balance->walk->run task curriculum) learned the
  full task set (reward ~3000 by 6x10^7 steps; velocity-tracking error ~0.1 m/s vs 0.4+ m/s otherwise); the task-only curriculum and the
  UNIFORMLY-SCALED-morphology curriculum stayed near zero, and random morphology switching failed entirely. Lessons: balance-first task curriculum ALONE failed
  there; HOW the body scales mattered (realistic proportions worked, uniform scaling did not) — our uniformly-scaled pendulum growth failing would be consistent
  with their uniform-scaling failure, not evidence against development per se.
- **Bodies causally modulate learnability.** Gupta, Savarese, Ganguli & Fei-Fei (2021, DERL): evolution selects morphologies that learn faster (a morphological
  Baldwin effect), mechanistically because physically stable, energy-efficient morphologies facilitate learning and control. REvolveR (Liu, Pathak & Kitani,
  ICML 2022): transferring an expert policy through a CONTINUOUS sequence of interpolated intermediate morphologies significantly improves target-robot sample
  efficiency, especially under sparse reward — direct support for gradual over abrupt growth.
- **The physics of "small is easier" is FALSE for balance dynamics and must be modeled honestly.** The linearized inverted pendulum diverges with time constant
  tau = sqrt(l_eff / g): toddler CoM ~0.45 m gives tau ~ 0.21 s vs adult ~1.0 m giving ~0.32 s — the SHORT pendulum falls ~50% faster and is HARDER for any
  controller with fixed feedback delay or decision rate. Delayed-feedback theory (Milton et al. 2009; Insperger & Milton 2014): stabilization requires delay <
  critical delay proportional to sqrt(l_eff / g); the human stick-balancing limit is ~0.3 m (a 32-cm stick's critical delay ~0.21 s sits at the estimated
  0.08-0.23 s human feedback delay). In discrete-time RL with fixed dt, decisions-per-fall-time N = tau/dt ~ sqrt(l): the small body gets FEWER decisions per
  divergence time. The honest advantages of smallness are: (1) fall COST ~ m*g*h ~ L^4 (18x empirically); (2) fast recovery (<3 s to resume play); (3) relative
  actuation authority — muscle torque scales ~L^3 while gravitational toppling torque scales ~L^4, so authority/requirement ~ 1/L favors small.
- **The dimensionless-equivalence hazard.** For a torque-controlled pendulum, the dynamics in nondimensional variables (theta, theta_dot * sqrt(l/g), u/(m*g*l),
  t/sqrt(l/g)) are scale-invariant: a growing pendulum differs from a fixed adult one ONLY through (a) dt/tau (decision rate), (b) u_max/(m*g*l) (authority),
  (c) reward/penalty magnitudes, (d) the induced state-discretization grid. If all four are normalized, growth is mathematically a no-op — a free
  placebo/falsification condition, and any observed benefit must be attributed to one of those four channels, not to "morphology" as an irreducible cause.
- **Fact-checks for the paper's framing.** Bones: newborns have roughly 270-300 partly cartilaginous skeletal elements fusing to the adult ~206 (adults vary
  206-213); write "approximately 270-300 partly cartilaginous skeletal elements at birth versus ~206 adult bones," never "babies have 300 bones." Milestones:
  WHO Multicentre Growth Reference Study (2006): walking alone 1st-99th percentile 8.2-17.6 months (median ~12), with standing-alone preceding walking-alone —
  the empirical basis for balance-before-walking.
- **The right non-morphological ablation exists in the literature.** Yu, Turk & Liu (2018) learned biped locomotion with a curriculum of external assistive
  forces annealed to zero — an established easy-start manipulation that distinguishes "any easy-start curriculum helps" from "growing the body specifically
  helps." Karpathy & van de Panne (2012) staged motor-skill curricula for an articulated hopper.

### Implications for our design

- **Where EXP5 follows the literature:** damage cost (s / s_adult)^4 is the impact-energy scaling (isometric E ~ L^4 gives ~21x; Han & Adolph's empirical 18.4x)
  — at s=0.5 the child's fall costs 1/16th of the adult's, inside the 18-21x anchor range for a half-scale body; fixed control dt means the small body gets
  fewer decisions per fall time (faster dynamics), which DESIGN.md explicitly flags as "not rigged in either direction" — the honest direction-of- difficulty
  disclosure the file demands; grow-jump vs grow-linear tests gradualism (REvolveR's prediction); grow-adaptive ("grow when ready," rolling no-fall rate > 70%)
  is the performance-gated staging the developmental-robotics literature recommends; grow-linear-walk isolates the balance-first factor within matched
  morphology, and adult-balance-first isolates it on the adult body — together approximating the recommended B0/B1/B2 decomposition; all conditions are
  evaluated on the identical ADULT body every 4k steps; budgets matched at 120k.
- **The largest deviation: the decomposition controls are absent.** DESIGN.md has no adult-body penalty-anneal-only arm (cheap-falls channel alone), no
  authority-anneal arm, no dt-schedule arm, and no fully-normalized placebo. Per the file's decision rule, EXP5's claim must therefore be the Adolph/Han
  cheap-falls thesis — "growing the body reaches adult competence with less cumulative damage" — and NOT "morphology per se helps beyond reward scheduling,"
  which would require beating a penalty-anneal control. The cumulative-damage headline in DESIGN.md is consistent with this scoping; the paper must resist
  drifting into the stronger claim.
- **Torque-scaling choice needs an explicit defense.** DESIGN.md sets tau_max = 40 s^2 and labels it square-cube. Muscle FORCE scales ~L^2 (cross-section), but
  muscle TORQUE scales ~L^3 (force x lever arm); with tau_max ~ s^2 and toppling torque m*g*l ~ 15 s^4 * g, relative authority goes as 1/s^2 — stronger
  small-body over-actuation than the biological 1/s. This pushes the growth condition toward an assistance curriculum (the Yu et al. analog) and should either
  be changed to s^3, defended explicitly, or reported as a declared authority trajectory A(s) = u_max/(m*g*l) per the dimensionless bookkeeping recommendation.
- Standing local optimum: the walking reward (+1/step upright, +2/step on target) contains an alive bonus; a balance-first agent could farm upright reward while
  ignoring targets. The eval threshold must be set so that target-tracking is required (or forward/tracking terms reported separately) so "balance-first wins"
  cannot be reward hacking.
- Seeds: 20 per condition is below the file's >=50 recommendation for bimodal walker-style outcomes; report the fraction of seeds that never learn.
- Missing but cheap: robustness evaluation (graded impulse pushes, +/-10-20% mass/length perturbation, observation noise) — Bongard's actual headline was
  robustness, and growth conditions may win on that axis even if matched on speed; a get-up/recovery-time cost after falls (the second Han & Adolph channel,
  distinct from penalty magnitude); and dimensionless-vs-absolute bin robustness checks (theta_dot scale ~ sqrt(g/l) shifts which cells are visited as the body
  grows — growth sweeps the policy through the table, an unintended coverage bonus or negative-transfer source).
- Position the contribution as the file recommends: no prior work tests morphological growth with size-scaled fall costs in tabular within-lifetime RL; prior
  art is evolutionary (Bongard), neuroevolutionary (Naya-Varela), or deep muscle-actuated (Badie). Cite Naya-Varela's mixed-results message so a null or
  conditional result is a finding, not a failure — and note a null under uniform scaling is consistent with Badie's uniform-scaling failure.

### Validity risks

- Reward-schedule confound: size-scaled fall penalty makes growth also a penalty-annealing curriculum; without the adult-body penalty-anneal control,
  "morphology helps" is not established (only the cheap-falls thesis is).
- Dimensionless no-op artifact: near-normalized settings make growth a no-op; any benefit observed there is a pipeline leak (extra steps, exploration-clock
  resets at growth events, Q-table re-indexing bugs).
- Direction-of-difficulty artifact: with fixed dt the small body is dynamically HARDER (tau = sqrt(l/g)); if growth helps, the mechanism cannot be "easier early
  dynamics" and must be traced to cost/authority channels.
- Torque-limit conflation: the chosen u_max exponent (s^2 vs s^3 vs s^4) changes the story between assistance curriculum and constant relative authority —
  choose deliberately and report.
- Optimism-inheritance artifact: the cheap-fall child phase leaves less-negative Q-values that act as optimistic initialization after growth; run an
  optimistic-init adult baseline to exclude this.
- Reset-frequency artifact: small bodies terminate faster and cheaply, visiting the informative near-upright start distribution more often per unit steps; match
  or report episode counts.
- Discretization aliasing: results that flip between absolute and dimensionless binning indicate the effect lives in the representation, not the physics; run
  bin-count robustness (e.g., 15/25/41 bins per dim).
- Unlearnable-baseline artifact: if adult-direct essentially never learns, "growth wins" collapses to "the baseline was pathological"; verify adult-direct
  learns at extended budget and report both a learnable and a hard regime.
- Budget/clock leaks: exploration and lr schedules interacting differently with phases; equalize total steps and audit schedules per condition (DESIGN.md
  matches budgets; schedule clocks need auditing).
- Evaluation asymmetry and reward hacking of the alive bonus (see above); tune hyperparameters only on adult-direct, then freeze for all conditions.
- Overgeneralization from prior art: do not claim to "replicate" Bongard (2011, evolutionary) or Badie (2025, anthropometric deep RL); claim analogy. A
  balance-first null or negative transfer is Adolph-consistent, not a broken experiment.
- Schedule-choice fragility: one growth schedule (e.g., linear over 60%) may not survive schedule variation (Berthouze & Lungarella's alternation result);
  EXP5's schedule sweep (linear/adaptive/jump) partially covers this — scope claims to tested schedules.
- Framing fact-checks: never write "babies have 300 bones" or "short pendulums are easier to balance" — use the verified phrasings.

---

## 7. Methodology: Statistics, Budgets, and Fair Comparison

### What is known

- **Few-run RL results are unreliable — quantified.** Agarwal et al. (NeurIPS 2021 Outstanding Paper): with 100 runs per algorithm on Atari-100k, subsampling
  shows that at the typical 3-10 runs sample medians are so variable that published orderings flip (their reanalysis suggests DER can outperform OTR, contrary
  to published claims), and some published gains (CURL/SUNRISE) were partly artifacts of a non-standard max-over-evaluations protocol. Henderson et al. (AAAI
  2018): splitting 10 runs of the SAME algorithm (TRPO on HalfCheetah) into two groups of 5 seeds produced "statistically different" distributions (t = -9.09, p
  = 0.0016); architecture flipped PPO on Hopper from 2790 +/- 62 to 61 +/- 33; and different codebases of the same algorithm diverge under identical
  hyperparameters.
- **The recommended reporting toolkit.** Agarwal et al.: IQM (mean of the middle 50% of runs, pooled across tasks) as primary aggregate; 95% stratified
  bootstrap CIs; performance profiles; probability of improvement P(X>Y); optimality gap. Precision guidance: aggregate stratified-bootstrap CIs usable from
  5-10 runs, percentile CIs on IQM good from N=10, per-task CIs need 20-30+, plain medians need 50-100. Implemented in the open-source rliable library.
- **Test calibration in RL is not what the textbooks say.** Colas, Sigaud & Oudeyer (2019): measured false-positive rates alpha* of 6 tests on analytic
  distributions and 192 real SAC/TD3 runs — the bootstrap test should never be used below N=50, the permutation test never below N=10, and the ranked t-test is
  degenerate at N=2 and unreliable below N~5-10. The standard advice "use non-parametric tests when data are not normal" FAILS in RL: Mann-Whitney and ranked t
  show alpha* > 0.1, worsening with N (up to >0.3 at N=100), when one distribution is skewed and the other symmetric, or when variances differ (alpha* > 0.1 for
  bimodal data regardless of N) — Mann-Whitney assumes equal shape and spread. Welch's t-test was the most robust overall, though even Welch inflates to alpha*
  ~ 0.1 with one skewed/bimodal distribution at N<10; the fix is to test at lower alpha (e.g., 0.01) so realized alpha* stays below 0.05. They also recommend
  against Kolmogorov-Smirnov for "better than" claims.
- **Power planning.** Colas et al.: power is roughly test-independent; 0.8 power at alpha = 0.05 needs N ~ 50-100 seeds for relative effect epsilon = 0.5, N ~
  20 for epsilon = 1, N ~ 5-10 for epsilon = 2. Their real SAC-vs-TD3 comparison had epsilon = 0.93 (means), needing N = 10-15; at N=5, effects must be 3-4x
  larger to detect. Companion recipe in Colas et al. (2018), "How Many Random Seeds?".
- **Small-environment specifics.** Patterson, Neumann, White & White (JMLR 2024): "in almost all cases 5 runs is insufficient... even 30 runs can be
  insufficient if the distribution is heavily skewed"; their PuddleWorld case study (directly analogous to laptop gridworlds) needed ~20 runs for accurate
  percentile- bootstrap CIs and ~30 to estimate means accurately. Default to percentile- bootstrap CIs; do NOT report standard errors. Hyperparameter
  maximization bias: E[max_h Gbar_h] >= max_h E[Gbar_h] — reported performance rises with the number of configs tried; remedy is the two-stage design (tune on
  ~10 runs, rerun the selected config on FRESH seeds for the reported evaluation).
- **Transfer metrics and budget accounting.** Taylor & Stone (JMLR 2009): jumpstart, asymptotic performance, total reward/AUC, transfer ratio (NOT
  offset-invariant — avoid unless returns have a meaningful zero), and time-to-threshold (threshold is arbitrary — sweep it and report time-vs- threshold
  curves). Their total-time vs target-task-time scenarios become Narvekar et al.'s (2020) weak vs strong transfer; strong transfer (all curriculum time counted,
  curves offset) is the demanding standard.
- **Censored time-to-threshold has no native RL protocol — borrow one.** None of Agarwal/Colas/Henderson/Patterson treat runs that never reach threshold.
  Adjacent literatures do: Kaplan-Meier product-limit estimator (1958), log-rank/ Mantel-Cox test (Mantel 1966), restricted mean survival time as a
  proportional- hazards-free summary (Royston & Parmar 2013); run-time distributions with cutoffs (Hoos & Stuetzle, UAI 1998), performance profiles with
  failures at ratio infinity (Dolan & More 2002), and COCO's ERT = total evaluations (censored included) / number of successes (Hansen et al. 2021). Averaging
  time-to-threshold over only successful runs is a selection-biased estimator.
- **Multi-environment aggregation and protocol hygiene.** Demsar (JMLR 2006): Wilcoxon signed-rank over per-environment scores for two algorithms, Friedman +
  Nemenyi for k algorithms — the right complement when the environment count is small. Machado et al. (JAIR 2018): report online training performance at fixed
  budgets, never best-policy snapshots. Whiteson et al. (2011): evaluate over a distribution of environment instances to avoid environment overfitting. Jordan
  et al. (ICML 2020) give a complete evaluation pipeline.

### Implications for our design

- DESIGN.md's shared statistical protocol already follows the core toolkit: IQM across seeds, 95% percentile bootstrap (10k resamples), Holm-Bonferroni within
  each experiment's primary family, identical eval protocols with eval steps excluded identically, per-seed raw metrics in every JSON, and strict
  budget-matching (the strong-transfer standard) everywhere.
- Censoring is handled non-degenerately: censored seeds are reported explicitly and assigned budget+1 for rank tests (conservative), with the censoring fraction
  always shown. This avoids the successes-only bias; it is simpler than the recommended Kaplan-Meier / log-rank / RMST treatment — adding KM curves and a
  solved-fraction test would strengthen EXP2/EXP5 time-to-threshold claims, and the threshold (90%) should be swept per Taylor & Stone.
- **The sharpest contradiction between DESIGN.md and the literature review:** DESIGN.md prescribes two-sided Mann-Whitney as the primary test, but Colas et al.
  specifically show Mann-Whitney's false-positive rate inflates (up to >0.3, growing with N) exactly when distributions differ in shape or spread — which is the
  expected regime in our experiments (bimodal solved/unsolved seeds in TrapGrid and BalanceBot). The review recommends Welch's t-test at alpha = 0.01 for
  skewed/bimodal data. Options: switch the primary test to Welch, keep Mann-Whitney but lower alpha and verify distribution shapes per comparison, or pair it
  with P(improvement) and report effect sizes — the choice must be made and pre-registered before confirmatory runs.
- Seed counts: 30-40 seeds (EXP1-4) give 0.8 power for epsilon >= ~0.75-0.9; EXP5's 20 seeds only for epsilon >= ~1 — acceptable if the expected effects are
  large (they are, per predictions), but null results must be reported as CIs on differences, not as "no effect," and EXP5 would benefit from 50 seeds at
  trivial cost.
- Not yet specified in DESIGN.md and worth adding: two-stage hyperparameter protocol (tune on disjoint seeds, evaluate fresh; equal tuning budgets across arms —
  the drill/growth arms have extra knobs); exact effect sizes reported alongside p-values; normalization anchors for cross-experiment aggregation (tabular V* is
  exactly computable by value iteration); and layout-distribution evaluation or explicit scoping of claims to the single fixed layouts (Whiteson's
  environment-overfitting warning applies to GridHome, SoccerGrid, TrapGrid as single instances).

### Validity risks

- Skewed/bimodal outcomes inflate false-positive rates for every test; at nominal alpha = 0.05 without shape checks, some "significant" results will be noise.
- Mann-Whitney as primary test is itself a confound in the unequal-shape regime (more seeds make the error worse, not better).
- Successes-only time-to-threshold is selection-biased — the single most likely self-inflicted confound; DESIGN.md's budget+1 assignment avoids it but must be
  applied consistently.
- Log-rank assumptions (non-informative censoring, proportional hazards) break if arms get different budgets or KM curves cross; RMST or solved-fraction tests
  must then carry the claim.
- Threshold choice can reverse conclusions; smoothing windows and hysteresis rules chosen after seeing data are p-hacking — pre-register and sweep.
- Weak-transfer accounting overstates curriculum benefit (DESIGN.md complies with strong transfer; keep it that way through implementation).
- Maximization bias from tuning on reporting seeds; unequal tuning budgets fabricate rankings.
- Multiple-comparison inflation across experiments x conditions x metrics; DESIGN.md's Holm-Bonferroni-within-family plus one primary comparison per experiment
  is the right structure — enforce it.
- Small-M aggregation: stratified bootstrap was designed for 26-57 tasks; with 5 experiments, report per-experiment results as primary and any cross-experiment
  aggregate as illustrative.
- Pseudo-replication: single fixed layouts scope conclusions to those layouts; near-deterministic arms can have (near-)zero variance where t-statistics blow up;
  paired designs must use paired tests and vice versa.

---

## 8. Site Techniques (Interactive Explorable)

The full engineering findings live in `research_site.json`; they are build guidance, not paper citations. Summary of what matters:

- **Pinned, verified stack:** three.js r147 UMD from cdnjs plus the five fat-line addons (Line2 et al.) from jsdelivr at the SAME version — the examples/js
  directory was deleted in r148 and modern three.js is ESM-only (0.185.1 three.min.js 404s), so r147 is the correct pin; GSAP 3.15.0 core + ScrollTrigger +
  DrawSVGPlugin from cdnjs (all GSAP plugins free since 3.13, May 2025). Set `renderer.outputEncoding = THREE.sRGBEncoding` (r147 predates the r152
  color-management flip) and `LineMaterial.resolution` on init and resize.
- **Architecture:** one fixed full-viewport canvas for the whole page (browsers cap WebGL contexts at ~8; resources don't share across contexts); one scene per
  chapter, scissor-split for in-chapter comparisons (H5's two bodies); all text as HTML/SVG overlays; one master timeline per chapter with labeled beats, never
  nested ScrollTriggers; scrub:1; render on demand only (no free-running rAF).
- **Data:** the artifact CSP silently blocks fetch(), so every trajectory, layout, advice set, and body-size schedule from DESIGN.md's `viz` export blocks must
  be inlined as JSON typed-array literals at build time (downsampled to respect the 16MB page budget). This is why the DESIGN.md output contract requires the
  `viz` block — the site build consumes `results/expN.json` directly.
- **UX and accessibility rules:** Bostock's five scroll rules (reversible, scrubbed, no scrolljacking, keyboard intact); The Pudding's no-vh rule (compute step
  heights from window.innerHeight); Distill 2020's finding that steppers slightly beat continuous scroll on comprehension — hence discrete labeled beats inside
  each scrubbed chapter; gsap.matchMedia reduced-motion variants are required (scrubbed camera flights are the classic vestibular trigger), with hidden-text
  narration for every WebGL stage.
- **Per-hypothesis signature scenes** (spotlight blindfold walk for H1, ghost- trajectory accumulation for H2, piano-roll wall for H3, family-tree descent for
  H4, growing-body split-screen for H5) are specified in the JSON with the exact primitives (InstancedMesh for grids/populations — a 20x20 gridworld + 400
  Q-arrows = 2 draw calls; Line2 ribbons for trajectories; DrawSVG for learning curves; orthographic dimetric camera for dioramas).

---

## 9. Positioning: What Is Novel, What Is Replication-in-Miniature

Much of this program is deliberate replication: preregistered developmental- psychology and motor-learning effects (latent learning, part-task curricula, the
Shea & Morgan crossover, cheap infant falls) reproduced in minimal, fully inspectable RL systems where every quantity — value tables, models, beliefs, advice —
can be printed and audited; that is a strength, not a hedge, and the methodology literature (Sections 7) shows exactly how rarely such effects are tested with
matched budgets and adequate seeds even in RL itself. Against that baseline, four elements are genuinely novel to the best of our review. (a) The
blindfold/dead-reckoning evaluation of a learned world model: prior work either trains blind agents end-to-end (Wijmans et al. 2023) or evaluates models by
dream-training transfer (Ha & Schmidhuber 2018); using an already-learned Dyna model as the substrate for belief-filtered dead reckoning, with the home-vs-
stranger and touch/no-touch contrasts as the discriminating controls, is a new evaluation protocol rather than a new algorithm. (b) Iterated distillation
through an episodic-memory bottleneck with plasticity-decaying teachers: the pieces exist separately — iterated in-context/in-weights generation chains (Cook et
al. 2024), zero-shot cultural transmission (Bhoopchand et al. 2023), Reincarnating RL's full-policy reuse (Agarwal et al. 2022), born-again distillation into
fresh same-capacity students (Furlanello et al. 2018), fresh-network distillation within a single lineage (Igl et al. 2021), reset methods whose transmission
channel is the entire unfiltered replay buffer (Nikishin et al. 2022), and the iterated-learning bottleneck theory of Kirby and Griffiths & Kalish — but the
targeted review's verdict is that no published work combines an aging (plasticity-decaying) teacher, fresh tabular students, and a hard, countable
curated-trajectory cap (<=100 (s, a) pairs from the top-3 episodes) with the weight-copy / long-life / no-inheritance causal battery; adding the
reset-with-full-replay baseline (Section 5) is what completes that differentiation. (c) In-silico contextual interference via engineered shared-feature
interference: the RL generalization literature varies levels and measures gaps (Cobbe et al. 2019) but does not implement the Shea & Morgan schedule
manipulation over a fixed variant set with a shared-parameter mechanism hypothesis and an acquisition-retention-transfer battery. (d) Morphological growth
curricula with square-cube-honest physics and damage accounting: prior morphological-development studies are evolutionary (Bongard 2011), neuroevolutionary with
mixed results (Naya-Varela 2023), or deep muscle-actuated (Badie 2025); none runs within-lifetime tabular RL with an explicitly disclosed tau = sqrt(l/g)
difficulty direction, L^4 fall-cost scaling anchored to Han & Adolph's 18.4x infant/adult energy ratio, and cumulative damage as the headline metric. Where the
program replicates, it should say so loudly; where it claims novelty, it should claim exactly these four protocols and no more.

---

## 10. Master Bibliography

All citations from the eight research files, deduplicated (same title + year = one entry), alphabetical by first author. Format: Authors (Year). Title. Venue.
URL (where given).

- Zaheer Abbas, Rosie Zhao, Joseph Modayil, Adam White, Marlos C. Machado (2023). Loss of Plasticity in Continual Deep Reinforcement Learning. CoLLAs 2023. https://arxiv.org/abs/2303.07507
- Karen E. Adolph (2000). Specificity of learning: Why infants fall over a veritable cliff. Psychological Science, 11(4), 290-295. https://journals.sagepub.com/doi/10.1111/1467-9280.00258
- Karen E. Adolph, Whitney G. Cole, Meghana Komati, Jessie S. Garciaguirre, Daryaneh Badaly, Jesse M. Lingeman, Gladys L. Y. Chan, Rachel B. Sotsky (2012). How Do You Learn to Walk? Thousands of Steps and Dozens of Falls per Day. Psychological Science, 23(11), 1387-1394. https://journals.sagepub.com/doi/10.1177/0956797612446346
- Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice. NeurIPS 2021 (Outstanding Paper Award). https://arxiv.org/abs/2108.13264
- Rishabh Agarwal et al. (Google Research) (2021). rliable: Library for Reliable Evaluation on RL Benchmarks (IQM, stratified bootstrap CIs, performance profiles). Open-source library accompanying NeurIPS 2021 paper. https://github.com/google-research/rliable
- Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare (2022). Reincarnating Reinforcement Learning: Reusing Prior Computation to Accelerate Progress. NeurIPS. https://arxiv.org/abs/2206.01626
- Achraf Ammar et al. (2023). The myth of contextual interference learning benefit in sports practice: A systematic review and meta-analysis. Educational Research Review, 39, 100537. https://www.sciencedirect.com/science/article/abs/pii/S1747938X23000301
- P. Anderson, A. Chang, D. S. Chaplot, A. Dosovitskiy, S. Gupta, V. Koltun, J. Kosecka, J. Malik, R. Mottaghi, M. Savva, A. R. Zamir (2018). On Evaluation of Embodied Navigation Agents (SPL metric). arXiv:1807.06757. https://arxiv.org/abs/1807.06757
- Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, Wojciech Zaremba (2017). Hindsight Experience Replay. NeurIPS 2017. https://arxiv.org/abs/1707.01495
- Thomas Anthony, Zheng Tian, David Barber (2017). Thinking Fast and Slow with Deep Learning and Tree Search. NeurIPS 2017. https://arxiv.org/abs/1705.08439
- Jake Archibald (2013). Animated line drawing in SVG. jakearchibald.com. https://jakearchibald.com/2013/animated-line-drawing-svg/
- Minoru Asada, Shoichi Noda, Sukoya Tawaratsumida, Koh Hosoda (1996). Purposive Behavior Acquisition for a Real Robot by Vision-Based Reinforcement Learning. Machine Learning, 23, 279-303. https://link.springer.com/article/10.1023/A:1018237008823
- Jordan T. Ash, Ryan P. Adams (2020). On Warm-Starting Neural Network Training. NeurIPS 2020. https://arxiv.org/abs/1910.08475
- Daniel W. Ash, Dennis H. Holding (1990). Backward versus Forward Chaining in the Acquisition of a Keyboard Skill. Human Factors, 32(2). https://journals.sagepub.com/doi/10.1177/001872089003200202
- Nadine Badie, Firas Al-Hafez, Pierre Schumacher, Daniel F. B. Haeufle, Jan Peters, Syn Schmitt (2025). Bioinspired morphology and task curricula for learning locomotion in bipedal muscle-actuated systems. Communications Engineering (Nature Portfolio), vol. 4. https://www.nature.com/articles/s44172-025-00443-0
- A. Banino, C. Barry, B. Uria, C. Blundell, T. Lillicrap, et al. (2018). Vector-based navigation using grid-like representations in artificial agents. Nature, 557, 429-433. https://www.nature.com/articles/s41586-018-0102-6
- Yoshua Bengio, Jerome Louradour, Ronan Collobert, Jason Weston (2009). Curriculum Learning. ICML 2009, pp. 41-48. https://dl.acm.org/doi/10.1145/1553374.1553380
- Nikolai A. Bernstein (1967). The Co-ordination and Regulation of Movements. Pergamon Press, Oxford.
- Luc Berthouze, Max Lungarella (2004). Motor skill acquisition under environmental perturbations: On the necessity of alternate freezing and freeing of degrees of freedom. Adaptive Behavior, 12(1), 47-64. https://journals.sagepub.com/doi/10.1177/105971230401200104
- Avishkar Bhoopchand, Bethanie Brownfield, Adrian Collister, et al. (Cultural General Intelligence Team, DeepMind) (2023). Learning Few-Shot Imitation as Cultural Transmission. Nature Communications 14, article 7536. https://www.nature.com/articles/s41467-023-42875-2
- Robert A. Bjork (1994). Memory and metamemory considerations in the training of human beings. In J. Metcalfe & A. Shimamura (Eds.), Metacognition: Knowing about knowing (pp. 185-205), MIT Press.
- Elizabeth L. Bjork, Robert A. Bjork (2011). Making things hard on yourself, but in a good way: Creating desirable difficulties to enhance learning. In M. A. Gernsbacher et al. (Eds.), Psychology and the Real World, Worth Publishers. https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/07/RBjork_inpress.pdf
- H. C. Blodgett (1929). The effect of the introduction of reward upon the maze performance of rats. University of California Publications in Psychology, 4, 113-134. https://searchworks.stanford.edu/view/75307
- Josh C. Bongard (2011). Morphological change in machines accelerates the evolution of robust behavior. Proceedings of the National Academy of Sciences, 108(4), 1234-1239. https://www.pnas.org/doi/10.1073/pnas.1015390108
- Mike Bostock (2014). How To Scroll. bost.ocks.org. https://bost.ocks.org/mike/scroll/
- Maya Cakmak, Manuel Lopes (2012). Algorithmic and Human Teaching of Sequential Decision Tasks. AAAI 2012.
- Baptiste Caramiaux, Frederic Bevilacqua, Marcelo M. Wanderley, Caroline Palmer (2018). Dissociable effects of practice variability on learning motor and timing skills. PLoS ONE, 13(3), e0193580. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0193580
- Christine E. Carter, Jessica A. Grahn (2016). Optimizing music learning: Exploring how blocked and interleaved practice schedules affect advanced performance. Frontiers in Psychology, 7:1251. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4989027/
- Roger Chaffin, Gabriela Imreh (2002). Practicing Perfection: Piano Performance as Expert Memory. Psychological Science, 13(4), 342-349.
- Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei (2017). Deep Reinforcement Learning from Human Preferences. NeurIPS. https://arxiv.org/abs/1706.03741
- Cleveland Clinic Health Library (2024). Here's How Many Bones Babies Have (anatomy fact-check: ~270-300 at birth, 206-213 adult). Cleveland Clinic. https://health.clevelandclinic.org/how-many-bones-does-a-baby-have
- Karl Cobbe, Oleg Klimov, Chris Hesse, Taehoon Kim, John Schulman (2019). Quantifying generalization in reinforcement learning. ICML 2019, PMLR 97, 1282-1289. https://arxiv.org/abs/1812.02341
- Karl Cobbe, Christopher Hesse, Jacob Hilton, John Schulman (2020). Leveraging procedural generation to benchmark reinforcement learning (Procgen Benchmark). ICML 2020. https://arxiv.org/abs/1912.01588
- Codrops (2022). How to Code an On-Scroll Folding 3D Cardboard Box Animation with Three.js and GSAP. Codrops (tympanus.net). https://tympanus.net/codrops/2022/12/13/how-to-code-an-on-scroll-folding-3d-cardboard-box-animation-with-three-js-and-gsap/
- Cedric Colas, Olivier Sigaud, Pierre-Yves Oudeyer (2018). How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments. arXiv:1806.08295. https://arxiv.org/abs/1806.08295
- Cedric Colas, Olivier Sigaud, Pierre-Yves Oudeyer (2019). A Hitchhiker's Guide to Statistical Comparisons of Reinforcement Learning Algorithms. arXiv:1904.06979 (v2 2022). https://arxiv.org/abs/1904.06979
- [Authors not listed in source] (2026). Compositionality and systematicity emerge from iterated learning in deep linear networks. PNAS. https://www.pnas.org/doi/abs/10.1073/pnas.2509739123
- Jonathan Cook, Chris Lu, Edward Hughes, Joel Z. Leibo, Jakob Foerster (2024). Artificial Generational Intelligence: Cultural Accumulation in Reinforcement Learning. NeurIPS 2024 (arXiv:2406.00392). https://arxiv.org/abs/2406.00392
- James P. Crutchfield, Sean Whalen (2010). Structural Drift: The Population Dynamics of Sequential Learning. arXiv:1005.2714 (later PLoS Computational Biology). https://arxiv.org/abs/1005.2714
- CSS-Tricks staff (2025). GSAP is Now Completely Free, Even for Commercial Use! CSS-Tricks. https://css-tricks.com/gsap-is-now-completely-free-even-for-commercial-use/
- Cultural General Intelligence Team, Avishkar Bhoopchand, Edward Hughes, et al. (2022). Learning Robust Real-Time Cultural Transmission without Human Data. arXiv preprint (journal version: Nature Communications 2023). https://arxiv.org/abs/2203.00715
- Stanislaw H. Czyz, Aleksandra M. Wojcik, Petra Solarska, Pawel Kiper (2024). High contextual interference improves retention in motor learning: systematic review and meta-analysis. Scientific Reports, 14. https://www.nature.com/articles/s41598-024-65753-3
- Janez Demsar (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. Journal of Machine Learning Research 7:1-30. https://jmlr.org/papers/v7/demsar06a.html
- Michael Dennis, Natasha Jaques, Eugene Vinitsky, Alexandre Bayen, Stuart Russell, Andrew Critch, Sergey Levine (2020). Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design (PAIRED). NeurIPS 2020. https://arxiv.org/abs/2012.02096
- Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton (2024). Loss of plasticity in deep continual learning. Nature 632:768-774. https://www.nature.com/articles/s41586-024-07711-7
- Elizabeth D. Dolan, Jorge J. More (2002). Benchmarking Optimization Software with Performance Profiles. Mathematical Programming 91:201-213. https://link.springer.com/article/10.1007/s101070100263
- Robert A. Duke, Amy L. Simmons, Carla Davis Cash (2009). It's Not How Much; It's How: Characteristics of Practice Behavior and Retention of Performance Skills. Journal of Research in Music Education, 56(4), 310-321. https://journals.sagepub.com/doi/10.1177/0022429408328851
- E. Duvelle, R. M. Grieves (2026). Tolman's Sunburst Maze 80 Years on: A Meta-Analysis Reveals Poor Replicability and Little Evidence for Shortcutting. European Journal of Neuroscience. https://onlinelibrary.wiley.com/doi/10.1111/ejn.70365
- Magnus Enquist, Kimmo Eriksson, Stefano Ghirlanda (2007). Critical Social Learning: A Solution to Rogers's Paradox of Nonadaptive Culture. American Anthropologist 109(4):727-734. https://anthrosource.onlinelibrary.wiley.com/doi/10.1525/aa.2007.109.4.727
- K. Anders Ericsson, Ralf T. Krampe, Clemens Tesch-Romer (1993). The Role of Deliberate Practice in the Acquisition of Expert Performance. Psychological Review 100(3):363-406.
- A. S. Etienne, K. J. Jeffery (2004). Path integration in mammals. Hippocampus, 14(2), 180-192. https://pubmed.ncbi.nlm.nih.gov/15098724/
- Jesse Farebrother, Marlos C. Machado, Michael Bowling (2018). Generalization and regularization in DQN. arXiv:1810.00123 (NeurIPS 2018 Deep RL Workshop). https://arxiv.org/abs/1810.00123
- Fernando Fernandez, Manuela Veloso (2006). Probabilistic Policy Reuse in a Reinforcement Learning Agent. AAMAS, pp. 720-727. https://dl.acm.org/doi/10.1145/1160633.1160762
- Carlos Florensa, David Held, Markus Wulfmeier, Michael Zhang, Pieter Abbeel (2017). Reverse Curriculum Generation for Reinforcement Learning. CoRL 2017, PMLR 78. https://proceedings.mlr.press/v78/florensa17a.html
- Fabio E. Fontana, Ovande Furtado, Oldemar Mazzardo, Jere D. Gallagher (2009). Whole and Part Practice: A Meta-Analysis. Perceptual and Motor Skills, 109(2), 517-530. https://journals.sagepub.com/doi/10.2466/pms.109.2.517-530
- N. Fujita, R. L. Klatzky, J. M. Loomis, R. G. Golledge (1993). The encoding-error model of pathway completion without vision. Geographical Analysis, 25(4), 295-314. https://onlinelibrary.wiley.com/doi/10.1111/j.1538-4632.1993.tb00300.x
- Tommaso Furlanello, Zachary C. Lipton, Michael Tschannen, Laurent Itti, Anima Anandkumar (2018). Born Again Neural Networks. ICML 2018. https://arxiv.org/abs/1805.04770
- Leo Gao, John Schulman, Jacob Hilton (2023). Scaling Laws for Reward Model Overoptimization. ICML. https://arxiv.org/abs/2210.10760
- GreenSock (GSAP Learning Center) (2024). Common ScrollTrigger mistakes. gsap.com resources. https://gsap.com/resources/st-mistakes/
- GreenSock/Webflow (2026). gsap.matchMedia() documentation (prefers-reduced-motion conditions, added in GSAP 3.11). GSAP Docs. https://gsap.com/docs/v3/GSAP/gsap.matchMedia()/
- GreenSock/Webflow (2026). ScrollTrigger documentation (scrub, pin, anticipatePin, fastScrollEnd, normalizeScroll). GSAP Docs. https://gsap.com/docs/v3/Plugins/ScrollTrigger/
- Thomas L. Griffiths, Michael L. Kalish (2007). Language Evolution by Iterated Learning With Bayesian Agents. Cognitive Science 31(3):441-480.
- GSAP team (2025). GSAP 3.13 release — GSAP and all plugins now 100% free via Webflow. gsap.com blog / webflow.com blog. https://gsap.com/blog/3-13/
- Mark A. Guadagnoli, Timothy D. Lee (2004). Challenge point: A framework for conceptualizing the effects of various practice conditions in motor learning. Journal of Motor Behavior, 36(2), 212-224.
- Agrim Gupta, Silvio Savarese, Surya Ganguli, Li Fei-Fei (2021). Embodied intelligence via learning and evolution. Nature Communications, 12:5721. https://www.nature.com/articles/s41467-021-25874-z
- D. Ha, J. Schmidhuber (2018). World Models (also published as: Recurrent World Models Facilitate Policy Evolution, NeurIPS 2018). arXiv:1803.10122 / NeurIPS 2018. https://arxiv.org/abs/1803.10122
- D. Hafner, T. Lillicrap, J. Ba, M. Norouzi (2020). Dream to Control: Learning Behaviors by Latent Imagination (DreamerV1). ICLR 2020. https://arxiv.org/abs/1912.01603
- D. Hafner, T. Lillicrap, M. Norouzi, J. Ba (2021). Mastering Atari with Discrete World Models (DreamerV2). ICLR 2021. https://arxiv.org/abs/2010.02193
- D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap (2023). Mastering Diverse Domains through World Models (DreamerV3; journal version: Mastering diverse control tasks through world models, Nature, 2025). arXiv:2301.04104; Nature (2025). https://arxiv.org/abs/2301.04104
- Danyang Han, Karen E. Adolph (2021). The impact of errors in infant development: Falling like a baby. Developmental Science, 24(5), e13069. https://pmc.ncbi.nlm.nih.gov/articles/PMC8178414/
- Nikolaus Hansen, Anne Auger, Raymond Ros, Olaf Mersmann, Tea Tusar, Dimo Brockhoff (2021). COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting (ERT with censored runs). Optimization Methods and Software 36(1):114-144. https://arxiv.org/abs/1603.08785
- Botao Hao, Rahul Jain, Tor Lattimore, Benjamin Van Roy, Zheng Wen (2023). Leveraging Demonstrations to Improve Online Learning: Quality Matters. ICML, PMLR 202:12527-12545. https://arxiv.org/abs/2302.03319
- Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger (2018). Deep Reinforcement Learning that Matters. AAAI 2018. https://arxiv.org/abs/1709.06560
- Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Gabriel Dulac-Arnold, Ian Osband, John Agapiou, Joel Z. Leibo, Audrunas Gruslys (2018). Deep Q-learning from Demonstrations (DQfD). AAAI. https://arxiv.org/abs/1704.03732
- Geoffrey E. Hinton, Steven J. Nowlan (1987). How Learning Can Guide Evolution. Complex Systems 1:495-502.
- Fred Hohman, Matthew Conlen, Jeffrey Heer, Duen Horng (Polo) Chau (2020). Communicating with Interactive Articles. Distill (DOI 10.23915/distill.00028). https://distill.pub/2020/communicating-with-interactive-articles/
- Holger H. Hoos, Thomas Stuetzle (1998). Evaluating Las Vegas Algorithms — Pitfalls and Remedies (run-time distributions with cutoffs). UAI 1998. https://arxiv.org/abs/1301.7383
- Maximilian Igl, Gregory Farquhar, Jelena Luketina, Wendelin Boehmer, Shimon Whiteson (2021). Transient Non-stationarity and Generalisation in Deep Reinforcement Learning (ITER). ICLR 2021. https://arxiv.org/abs/2006.05826
- Tamas Insperger, John Milton (2014). Sensory uncertainty and stick balancing at the fingertip. Biological Cybernetics, 108(1), 85-101. https://link.springer.com/article/10.1007/s00422-013-0582-2
- M. Janner, J. Fu, M. Zhang, S. Levine (2019). When to Trust Your Model: Model-Based Policy Optimization (MBPO). NeurIPS 2019. https://proceedings.neurips.cc/paper_files/paper/2019/file/5faf461eff3099671ad63c6f3f094f7f-Paper.pdf
- Scott M. Jordan, Yash Chandak, Daniel Cohen, Mengxue Zhang, Philip S. Thomas (2020). Evaluating the Performance of Reinforcement Learning Algorithms. ICML 2020. https://arxiv.org/abs/2006.16958
- L. Kaiser, M. Babaeizadeh, P. Milos, B. Osinski, R. H. Campbell, K. Czechowski, D. Erhan, C. Finn, et al. (2020). Model-Based Reinforcement Learning for Atari (SimPLe). ICLR 2020. https://arxiv.org/abs/1903.00374
- Sham Kakade, John Langford (2002). Approximately Optimal Approximate Reinforcement Learning. ICML 2002.
- E. L. Kaplan, Paul Meier (1958). Nonparametric Estimation from Incomplete Observations (Kaplan-Meier estimator). Journal of the American Statistical Association 53(282):457-481. https://www.jstor.org/stable/2281868
- Andrej Karpathy, Michiel van de Panne (2012). Curriculum learning for motor skills. Canadian Conference on Artificial Intelligence (LNAI 7310), pp. 325-330. https://cs.stanford.edu/people/karpathy/papers/motor-curriculum-full.pdf
- Robert Kerr, Bernard Booth (1978). Specific and varied practice of motor skill. Perceptual and Motor Skills, 46(2), 395-401. https://journals.sagepub.com/doi/10.1177/003151257804600201
- Simon Kirby, Hannah Cornish, Kenny Smith (2008). Cumulative cultural evolution in the laboratory: An experimental approach to the origins of structure in human language. PNAS 105(31):10681-10686. https://www.pnas.org/doi/10.1073/pnas.0707835105
- Simon Kirby, Mike Dowman, Thomas L. Griffiths (2007). Innateness and culture in the evolution of language. PNAS 104(12):5241-5245.
- Simon Kirby, Monica Tamariz, Hannah Cornish, Kenny Smith (2015). Compression and communication in the cultural evolution of linguistic structure. Cognition 141:87-102.
- George Konidaris, Andrew Barto (2009). Skill Discovery in Continuous Reinforcement Learning Domains using Skill Chaining. NeurIPS 22 (NIPS 2009). https://papers.nips.cc/paper/2009/file/e0cf1f47118daebc5b16269099ad7347-Paper.pdf
- Aviral Kumar, Joey Hong, Anikait Singh, Sergey Levine (2022). When Should We Prefer Offline Reinforcement Learning Over Behavioral Cloning? ICLR. https://arxiv.org/abs/2204.05618
- N. Lambert, K. Pister, R. Calandra (2022). Investigating Compounding Prediction Errors in Learned Dynamics Models. arXiv:2203.09637. https://arxiv.org/pdf/2203.09637
- Seunghyun Lee, Younggyo Seo, Kimin Lee, Pieter Abbeel, Jinwoo Shin (2021). Offline-to-Online Reinforcement Learning via Balanced Replay and Pessimistic Q-Ensemble. CoRL. https://arxiv.org/abs/2107.00591
- Xingyu Liu, Deepak Pathak, Kris M. Kitani (2022). REvolveR: Continuous Evolutionary Models for Robot-to-robot Policy Transfer. ICML 2022, PMLR 162 (long oral). https://arxiv.org/abs/2202.05244
- J. M. Loomis, R. L. Klatzky, R. G. Golledge, J. G. Cicinelli, J. W. Pellegrino, P. A. Fry (1993). Nonvisual navigation by blind and sighted: Assessment of path integration ability. Journal of Experimental Psychology: General, 122(1), 73-91. https://people.psych.ucsb.edu/loomis/jack/loomis_klatzky_93.pdf
- Max Lungarella, Giorgio Metta, Rolf Pfeifer, Giulio Sandini (2003). Developmental robotics: a survey. Connection Science, 15(4), 151-190. https://www.tandfonline.com/doi/abs/10.1080/09540090310001655110
- Clare Lyle, Mark Rowland, Will Dabney (2022). Understanding and Preventing Capacity Loss in Reinforcement Learning. ICLR 2022. https://arxiv.org/abs/2204.09560
- Marlos C. Machado, Marc G. Bellemare, Erik Talvitie, Joel Veness, Matthew Hausknecht, Michael Bowling (2018). Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents. Journal of Artificial Intelligence Research 61:523-562. https://arxiv.org/abs/1709.06009
- Brooke N. Macnamara, Megha Maitra (2019). The Role of Deliberate Practice in Expert Performance: Revisiting Ericsson, Krampe & Tesch-Romer (1993). Royal Society Open Science 6(8):190327. https://royalsocietypublishing.org/doi/10.1098/rsos.190327
- Richard A. Magill, Kellie G. Hall (1990). A review of the contextual interference effect in motor skill acquisition. Human Movement Science, 9(3-5), 241-289. https://www.sciencedirect.com/science/article/abs/pii/016794579090005X
- Nathan Mantel (1966). Evaluation of Survival Data and Two New Rank Order Statistics Arising in Its Consideration (log-rank test). Cancer Chemotherapy Reports 50(3):163-170.
- Tambet Matiisen, Avital Oliver, Taco Cohen, John Schulman (2017). Teacher-Student Curriculum Learning. arXiv:1707.00183; journal version in IEEE Transactions on Neural Networks and Learning Systems. https://arxiv.org/abs/1707.00183
- John Milton, Juan Luis Cabrera, Toru Ohira, Shigeru Tajima, Yukinori Tonosaki, Christian W. Eurich, Sue Ann Campbell (2009). The time-delayed inverted pendulum: Implications for human balance control. Chaos, 19(2), 026110. https://pubmed.ncbi.nlm.nih.gov/19566270/
- M. L. Mittelstaedt, H. Mittelstaedt (1980). Homing by path integration in a mammal. Naturwissenschaften, 67, 566-567. https://link.springer.com/article/10.1007/BF00450672
- Hossein Mobahi, Mehrdad Farajtabar, Peter L. Bartlett (2020). Self-Distillation Amplifies Regularization in Hilbert Space. NeurIPS 2020. https://arxiv.org/abs/2002.05715
- I. Momennejad, E. M. Russek, J. H. Cheong, M. M. Botvinick, N. D. Daw, S. J. Gershman (2017). The successor representation in human reinforcement learning. Nature Human Behaviour, 1, 680-692. https://www.nature.com/articles/s41562-017-0180-8
- mrdoob/three.js contributors (2022). InstancedMesh API documentation (r147). three.js docs, r147 source tree. https://github.com/mrdoob/three.js/blob/r147/docs/api/en/objects/InstancedMesh.html
- mrdoob/three.js contributors (2022). LineBasicMaterial API documentation — linewidth always 1 caveat (r147). three.js docs, r147 source tree. https://github.com/mrdoob/three.js/blob/r147/docs/api/en/materials/LineBasicMaterial.html
- M. Muller, R. Wehner (1988). Path integration in desert ants, Cataglyphis fortis. Proceedings of the National Academy of Sciences, 85(14), 5287-5290. https://www.pnas.org/doi/abs/10.1073/pnas.85.14.5287
- Ashvin Nair, Abhishek Gupta, Murtaza Dalal, Sergey Levine (2020). AWAC: Accelerating Online Reinforcement Learning with Offline Datasets. arXiv:2006.09359. https://arxiv.org/abs/2006.09359
- Ashvin Nair, Bob McGrew, Marcin Andrychowicz, Wojciech Zaremba, Pieter Abbeel (2018). Overcoming Exploration in Reinforcement Learning with Demonstrations. ICRA. https://arxiv.org/abs/1709.10089
- Mitsuhiko Nakamoto, Yuexiang Zhai, Anikait Singh, Max Sobol Mark, Yi Ma, Chelsea Finn, Aviral Kumar, Sergey Levine (2023). Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning. NeurIPS. https://arxiv.org/abs/2303.05479
- Sanmit Narvekar, Jivko Sinapov, Matteo Leonetti, Peter Stone (2016). Source Task Creation for Curriculum Learning. AAMAS 2016. https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/AAMAS16-Narvekar.pdf
- Sanmit Narvekar, Bei Peng, Matteo Leonetti, Jivko Sinapov, Matthew E. Taylor, Peter Stone (2020). Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey. Journal of Machine Learning Research 21(181):1-50. https://jmlr.org/papers/v21/20-212.html
- Martin Naya-Varela, Andres Faina, Richard J. Duro (2023). Engineering morphological development in a robotic bipedal walking problem: An empirical study. Neurocomputing, 527, 83-99. https://www.sciencedirect.com/science/article/pii/S0925231223000115
- James C. Naylor, George E. Briggs (1963). Effects of Task Complexity and Task Organization on the Relative Efficiency of Part and Whole Training Methods. Journal of Experimental Psychology 65(3). https://pubmed.ncbi.nlm.nih.gov/13937802/
- Andrew Y. Ng, Daishi Harada, Stuart Russell (1999). Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping. ICML.
- Evgenii Nikishin, Junhyuk Oh, Georg Ostrovski, Clare Lyle, Razvan Pascanu, Will Dabney, Andre Barreto (2023). Deep Reinforcement Learning with Plasticity Injection. NeurIPS 2023. https://arxiv.org/abs/2305.15555
- Evgenii Nikishin, Max Schwarzer, Pierluca D'Oro, Pierre-Luc Bacon, Aaron Courville (2022). The Primacy Bias in Deep Reinforcement Learning. ICML 2022. https://arxiv.org/abs/2205.07802
- Junhyuk Oh, Yijie Guo, Satinder Singh, Honglak Lee (2018). Self-Imitation Learning. ICML 2018. https://arxiv.org/abs/1806.05635
- Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, et al. (2022). Training Language Models to Follow Instructions with Human Feedback (InstructGPT). NeurIPS. https://arxiv.org/abs/2203.02155
- Andrew Patterson, Samuel Neumann, Martha White, Adam White (2024). Empirical Design in Reinforcement Learning. Journal of Machine Learning Research 25. https://arxiv.org/abs/2304.01315
- Jette Randlov, Preben Alstrom (1998). Learning to Drive a Bicycle using Reinforcement Learning and Shaping. ICML.
- Limor Raviv, Gary Lupyan, C. Shawn Green (2022). How variability shapes learning and generalization. Trends in Cognitive Sciences, 26(6), 462-483. https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(22)00065-1
- Yi Ren, Shangmin Guo, Matthieu Labeau, Shay B. Cohen, Simon Kirby (2020). Compositional Languages Emerge in a Neural Iterated Learning Model. ICLR 2020. https://arxiv.org/abs/2002.01365
- Alan R. Rogers (1988). Does Biology Constrain Culture? American Anthropologist 90(4):819-831. https://www.cognitionandculture.net/wp-content/uploads/Rogers-AA-90-819.pdf
- Stephane Ross, J. Andrew Bagnell (2010). Efficient Reductions for Imitation Learning. AISTATS. https://proceedings.mlr.press/v9/ross10a.html
- Stephane Ross, Geoffrey J. Gordon, J. Andrew Bagnell (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger). AISTATS. https://arxiv.org/abs/1011.0686
- Patrick Royston, Mahesh K. B. Parmar (2013). Restricted Mean Survival Time: An Alternative to the Hazard Ratio for the Design and Analysis of Randomized Trials with a Time-to-Event Outcome. BMC Medical Research Methodology 13:152. https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-13-152
- Andrei A. Rusu, Sergio Gomez Colmenarejo, Caglar Gulcehre, Guillaume Desjardins, James Kirkpatrick, Razvan Pascanu, Volodymyr Mnih, Koray Kavukcuoglu, Raia Hadsell (2016). Policy Distillation. ICLR 2016. https://arxiv.org/abs/1511.06295
- Russell Samora (2017). Responsive scrollytelling best practices. The Pudding (pudding.cool/process). https://pudding.cool/process/responsive-scrollytelling/
- Joseph Santamaria (2025). How to Build Cinematic 3D Scroll Experiences with GSAP. Codrops (tympanus.net). https://tympanus.net/codrops/2025/11/19/how-to-build-cinematic-3d-scroll-experiences-with-gsap/
- Stefan Schaal (1997). Learning from Demonstration. NIPS 9. https://papers.nips.cc/paper_files/paper/1996/hash/68d13cf26c4b4f4f932e3eff990093ba-Abstract.html
- Richard A. Schmidt (1975). A schema theory of discrete motor skill learning. Psychological Review, 82(4), 225-260. https://www.semanticscholar.org/paper/2a1332efbef8d0a67fd78ce0cfa69fc5117a933a
- Simon Schmitt, Jonathan J. Hudson, Augustin Zidek, Simon Osindero, Carl Doersch, Wojciech M. Czarnecki, Joel Z. Leibo, Heinrich Kuttler, Andrew Zisserman, Karen Simonyan, S. M. Ali Eslami (2018). Kickstarting Deep Reinforcement Learning. arXiv:1803.03835. https://arxiv.org/abs/1803.03835
- J. Schrittwieser, I. Antonoglou, T. Hubert, K. Simonyan, L. Sifre, S. Schmitt, A. Guez, E. Lockhart, D. Hassabis, T. Graepel, T. Lillicrap, D. Silver (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero). Nature, 588, 604-609. https://arxiv.org/abs/1911.08265
- Oliver G. Selfridge, Richard S. Sutton, Andrew G. Barto (1985). Training and Tracking in Robotics. IJCAI 1985, pp. 670-672. https://www.ijcai.org/Proceedings/85-1/Papers/129a.pdf
- John B. Shea, Robyn L. Morgan (1979). Contextual interference effects on the acquisition, retention, and transfer of a motor skill. Journal of Experimental Psychology: Human Learning and Memory, 5(2), 179-187. https://gwern.net/doc/psychology/spaced-repetition/1979-shea.pdf
- David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, et al. (2017). Mastering the Game of Go without Human Knowledge (AlphaGo Zero). Nature 550:354-359. https://www.nature.com/articles/nature24270
- Bruno Simon (2022). Crafting Scroll Based Animations in Three.js. Codrops (tympanus.net). https://tympanus.net/codrops/2022/01/05/crafting-scroll-based-animations-in-three-js/
- Kenny Smith, Simon Kirby, Henry Brighton (2003). Iterated learning: a framework for the emergence of language. Artificial Life 9(4):371-386.
- Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, Utku Evci (2023). The Dormant Neuron Phenomenon in Deep Reinforcement Learning. ICML 2023 (oral). https://arxiv.org/abs/2302.12902
- Yuda Song, Yifei Zhou, Ayush Sekhari, J. Andrew Bagnell, Akshay Krishnamurthy, Wen Sun (2023). Hybrid RL: Using Both Offline and Online Data Can Make RL Efficient. ICLR. https://arxiv.org/abs/2210.06718
- Peter Stone, Richard S. Sutton, Gregory Kuhlmann (2005). Reinforcement Learning for RoboCup-Soccer Keepaway. Adaptive Behavior 13(3):165-188.
- R. S. Sutton (1990). Integrated architectures for learning, planning, and reacting based on approximating dynamic programming. Proceedings of the Seventh International Conference on Machine Learning (ICML), 216-224. https://www.sciencedirect.com/science/chapter/edited-volume/abs/pii/B9781558601413500304
- R. S. Sutton (1991). Dyna, an integrated architecture for learning, planning, and reacting. ACM SIGART Bulletin, 2(4), 160-163. https://dl.acm.org/doi/10.1145/122344.122377
- R. S. Sutton, A. G. Barto (2018). Reinforcement Learning: An Introduction (2nd ed.), Ch. 8: Planning and Learning with Tabular Methods (Dyna Maze Example 8.1; blocking and shortcut mazes, Dyna-Q+). MIT Press. http://incompleteideas.net/book/the-book-2nd.html
- Richard S. Sutton, Doina Precup, Satinder Singh (1999). Between MDPs and semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning. Artificial Intelligence 112(1-2):181-211. https://www.sciencedirect.com/science/article/pii/S0004370299000521
- E. Talvitie (2017). Self-Correcting Models for Model-Based Reinforcement Learning. AAAI 2017 (arXiv:1612.06018); see also Talvitie, Model Regularization for Stable Sample Rollouts ("hallucinated replay"), UAI 2014. https://arxiv.org/abs/1612.06018
- Matthew E. Taylor, Peter Stone (2009). Transfer Learning for Reinforcement Learning Domains: A Survey. Journal of Machine Learning Research 10:1633-1685. https://jmlr.org/papers/v10/taylor09a.html
- Michael Henry Tessler, Jason Madeano, Pedro A. Tsividis, Brin Harper, Noah D. Goodman, Joshua B. Tenenbaum (2021). Learning to solve complex tasks by growing knowledge culturally across generations. CogSci 2021 / NeurIPS 2021 Cooperative AI Workshop. https://arxiv.org/abs/2107.13377
- D. Thistlethwaite (1951). A critical review of latent learning and related experiments. Psychological Bulletin, 48(2), 97-129. https://psycnet.apa.org/doi/10.1037/h0055171
- three.js manual (2023). Cameras (OrthographicCamera box projection, 2D/diorama use). threejs.org manual. https://threejs.org/manual/en/cameras.html
- three.js manual (threejsfundamentals) (2023). Multiple Canvases, Multiple Scenes (single-canvas scissor technique, ~8-context limit). threejs.org manual. https://threejs.org/manual/en/multiple-scenes.html
- three.js manual (2023). Rendering on Demand (requestRenderIfNotRequested pattern). threejs.org manual. https://threejs.org/manual/en/rendering-on-demand.html
- Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, Pieter Abbeel (2017). Domain randomization for transferring deep neural networks from simulation to the real world. IEEE/RSJ IROS. https://arxiv.org/abs/1703.06907
- E. C. Tolman (1948). Cognitive maps in rats and men. Psychological Review, 55(4), 189-208. https://psycnet.apa.org/doi/10.1037/h0061626
- E. C. Tolman, C. H. Honzik (1930). Introduction and removal of reward, and maze performance in rats. University of California Publications in Psychology, 4, 257-275. https://www.researchgate.net/figure/Maze-used-by-Tolman-and-Honzik-1930-to-study-latent-learning-in-rats-From-Tolman-EC_fig11_285777770
- E. C. Tolman, B. F. Ritchie, D. Kalish (1946). Studies in spatial learning. I. Orientation and the short-cut. Journal of Experimental Psychology, 36(1), 13-24. https://psycnet.apa.org/doi/10.1037/h0053944
- Lisa Torrey, Matthew E. Taylor (2013). Teaching on a Budget: Agents Advising Agents in Reinforcement Learning. AAMAS. https://www.ifaamas.org/Proceedings/aamas2013/docs/p1053.pdf
- Ikechukwu Uchendu, Ted Xiao, Yao Lu, Banghua Zhu, Mengyuan Yan, Josephine Simon, Matthew Bennice, Chuyuan Fu, Cong Ma, Jiantao Jiao, Sergey Levine, Karol Hausman (2023). Jump-Start Reinforcement Learning (JSRL). ICML, PMLR 202:34556-34583. https://proceedings.mlr.press/v202/uchendu23a.html
- H. van Hasselt, M. Hessel, J. Aslanides (2019). When to use parametric models in reinforcement learning? NeurIPS 2019. https://proceedings.neurips.cc/paper/2019/hash/1b742ae215adf18b75449c6e272fd92d-Abstract.html
- Ankit Vani, Max Schwarzer, Yuchen Lu, Eeshan Dhekane, Aaron Courville (2021). Iterated learning for emergent systematicity in VQA. ICLR 2021. https://arxiv.org/abs/2105.01119
- Jacques H. A. van Rossum (1990). Schmidt's schema theory: The empirical base of the variability of practice hypothesis. A critical analysis. Human Movement Science, 9(3-5), 387-435. https://www.sciencedirect.com/science/article/abs/pii/016794579090010B
- Mel Vecerik, Todd Hester, Jonathan Scholz, Fumin Wang, Olivier Pietquin, Bilal Piot, Nicolas Heess, Thomas Rothorl, Thomas Lampe, Martin Riedmiller (2017). Leveraging Demonstrations for Deep Reinforcement Learning on Robotics Problems with Sparse Rewards (DDPGfD). arXiv:1707.08817. https://arxiv.org/abs/1707.08817
- Beatrix Vereijken, Richard E. A. van Emmerik, H. T. A. Whiting, Karl M. Newell (1992). Free(z)ing degrees of freedom in skill acquisition. Journal of Motor Behavior, 24(1), 133-142.
- Rui Wang, Joel Lehman, Jeff Clune, Kenneth O. Stanley (2019). Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions. arXiv:1901.01753. https://arxiv.org/abs/1901.01753
- R. Wehner, M. V. Srinivasan (1981). Searching behaviour of desert ants, genus Cataglyphis (Formicidae, Hymenoptera). Journal of Comparative Physiology A, 142, 315-338. https://link.springer.com/article/10.1007/BF00605445
- Shimon Whiteson, Brian Tanner, Matthew E. Taylor, Peter Stone (2011). Protecting Against Evaluation Overfitting in Empirical Reinforcement Learning. IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning (ADPRL). https://www.cs.utexas.edu/~pstone/Papers/bib2html/b2hd-ADPRL11-shimon.html
- WHO Multicentre Growth Reference Study Group (2006). WHO Motor Development Study: Windows of achievement for six gross motor development milestones. Acta Paediatrica, Suppl 450:86-95. https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1651-2227.2006.tb02379.x
- Eric Wiewiora (2003). Potential-Based Shaping and Q-Value Initialization are Equivalent. Journal of Artificial Intelligence Research 19:205-208. https://arxiv.org/abs/1106.5267
- Dennis C. Wightman, Gavan Lintern (1985). Part-Task Training for Tracking and Manual Control. Human Factors 27(3):267-283. https://journals.sagepub.com/doi/10.1177/001872088502700304
- E. Wijmans, M. Savva, I. Essa, S. Lee, A. S. Morcos, D. Batra (2023). Emergence of Maps in the Memories of Blind Navigation Agents. ICLR 2023 (Outstanding Paper). https://arxiv.org/abs/2301.13261
- Jin Wu, Weiyi Cai, Derek Watkins, James Glanz (2020). How the Virus Got Out. The New York Times (interactive). https://www.nytimes.com/interactive/2020/03/22/world/coronavirus-spread.html
- Xiaoxia Wu, Ethan Dyer, Behnam Neyshabur (2021). When Do Curricula Work? ICLR 2021 (oral). https://arxiv.org/abs/2012.03107
- Tengyang Xie, Nan Jiang, Huan Wang, Caiming Xiong, Yu Bai (2021). Policy Finetuning: Bridging Sample-Efficient Offline and Online Reinforcement Learning. NeurIPS. https://arxiv.org/abs/2106.04895
- Wenhao Yu, Greg Turk, C. Karen Liu (2018). Learning symmetric and low-energy locomotion. ACM Transactions on Graphics (SIGGRAPH), 37(4):144. https://arxiv.org/abs/1801.08093
- Chiyuan Zhang, Oriol Vinyals, Remi Munos, Samy Bengio (2018). A study on overfitting in deep reinforcement learning. arXiv:1804.06893. https://arxiv.org/abs/1804.06893





