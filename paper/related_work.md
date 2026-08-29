# Related Work

Each of our five experiments sits on an established literature: the spectrum of channels between
imitation and pure reinforcement (EXP4's advice bottleneck and EXP2's boundary condition), latent
learning and world models (EXP1), curricula and part-whole training (EXP2), variability of practice
and contextual interference (EXP3), generational transmission (EXP4), and morphological development
(EXP5). We review each thread, then locate our contribution: mostly deliberate
replication-in-miniature, plus four protocols we believe are new.

## Between imitation and evaluation

Static demonstrations cannot be trusted at states the teacher never visited: behavioral cloning
with per-step error epsilon incurs up to O(T^2 epsilon) regret as the learner's own mistakes shift
the state distribution, and DAgger restores O(T epsilon)-order guarantees by reducing interactive
imitation to no-regret online learning (Ross et al. 2011). Folding demonstrations into value-based
RL buys a head start but can anchor: DQfD beat prioritized dueling DQN on 41 of 42 Atari games in
the first million steps, yet its persistent margin loss left it above its own demonstrator on only
14 (Hester et al. 2018). Annealing the teacher away removes the tether — kickstarted students reach
teacher level roughly 10x faster and end 42% above the from-scratch baseline precisely because the
distillation weight is driven to zero (Schmitt et al. 2018) — and Jump-Start RL moves instruction
entirely into the exploration distribution: a guide policy rolls in for a receding horizon while
the student's objective stays pure RL (Uchendu et al. 2023). The offline-to-online literature shows
why the handoff is delicate: distribution shift at the handoff can destroy a good initial policy
(Lee et al. 2021), conservative pretraining produces a characteristic dip that value calibration
removes (Nakamoto et al. 2023), while simply mixing offline data into the online buffer is provably
efficient under coverage (Song et al. 2023). Teacher quality caps the value of imitation, not of RL
(Kumar et al. 2022), and RLHF chains the stages — supervised imitation, then optimization against a
learned proxy under a KL tether (Ouyang et al. 2022).

Theory bounds when any of this helps. A guide policy covering the optimal policy's states turns the
exponential-in-horizon sample complexity of naive epsilon-greedy sparse-reward exploration into a
polynomial one (Uchendu et al. 2023), but the matching lower bound of Xie et al. (2021) shows that
in the worst case the optimal algorithm either reduces purely to the reference policy or ignores
it: instruction is worth exponential factors to naive explorers and approximately nothing to
optimistic ones. That asymmetry is load-bearing here — it predicts the boundary result of EXP2,
where the drill curriculum's advantage disappears against an optimistically initialized whole-game
learner. The channel matters independently of the content: priming values or policies from a
demonstration gave no speedup where priming a dynamics model enabled one-trial pole balancing
(Schaal 1997); Q-value initialization is update-for-update equivalent to potential-based shaping
(Wiewiora 2003), itself the only reward modification guaranteed policy-invariant (Ng et al. 1999);
and a fixed advice budget is best spent at high-importance states rather than early (Torrey &
Taylor 2013). The organizing lesson we take: acceleration comes from an artifact reshaping the
exploration distribution, lock-in from an artifact persisting in the objective. EXP4's advice prime
— a value artifact with no persistent loss term, freely overwritten by TD updates — sits on the
benign side of that split by design.

## World models and latent learning

Latent learning is the original evidence that agents acquire structure without reward: rats that
explored mazes unrewarded collapsed their error curves within a day or two of the first reward to
the level of always-rewarded controls (Blodgett 1929; Tolman & Honzik 1930), which Tolman (1948)
synthesized into the cognitive-map hypothesis. The cautionary half of that history matters equally:
the famous sunburst shortcut probe (Tolman et al. 1946) replicates poorly, exceeding chance in only
17% of 47 experiments in a recent meta-analysis (Duvelle & Grieves 2026), so we rest conclusions on
repeated probes over many seeds, never one dramatic display. Dead reckoning — the competence our
blindfold test isolates — is biologically real and lawful: gerbil mothers retrieve displaced pups
in total darkness on direct vectors from self-motion cues (Mittelstaedt & Mittelstaedt 1980);
blind and blindfolded-sighted humans path-integrate equally well, with error growing in path length
and complexity (Loomis et al. 1993); and mammalian path integration accumulates drift that landmark
fixes reset (Etienne & Jeffery 2004) — the pattern our slip noise, distance effects, and
bump-driven belief re-anchoring are built to reproduce.

In RL, Dyna made planning "RL applied to simulated experience" (Sutton 1990), and world models
scaled the idea: policies trained inside a learned dream transfer to reality, though at low dream
temperature the policy exploited model flaws, scoring ~2086 in the dream but ~193 in reality (Ha &
Schmidhuber 2018); one-step model errors compound with rollout horizon, which many short branched
rollouts mitigate (Janner et al. 2019); latent models predicting only reward, value, and policy
plan at superhuman level (Schrittwieser et al. 2020); and one fixed Dreamer configuration now spans
150+ tasks (Hafner et al. 2023). Two findings discipline our claims. Blind navigation success alone
does not require an explicit model — agents sensing only egomotion reach ~95% success and
spontaneously develop decodable metric maps (Wijmans et al. 2023) — so our inference rides on the
home-versus-stranger and touch/no-touch contrasts, not on blind success per se. And experience
replay is effectively a non-parametric model, so claimed model-based sample-efficiency gains can be
pure update-count artifacts (van Hasselt et al. 2019) — the reason EXP1 adds an update-matched
replay baseline before crediting Dyna's speedup to the model.

## Curricula and part-whole practice

Curriculum learning entered ML as easy-examples-first (Bengio et al. 2009); curriculum RL was
formalized as task generation, sequencing, and transfer (Narvekar et al. 2020) using the transfer
metrics of Taylor & Stone (2009). That framework's sharpest lesson is budget accounting: most
surveyed work demonstrates only "weak transfer," with drill time treated as sunk cost. The earliest
RL curriculum already won under strong accounting — easy-pole-first balancing took 73 total
failures against 119 direct (Selfridge et al. 1985) — as did the direct ancestors of our soccer
drills: vision-based shooting learned by starting episodes near the goal and moving them back
(Asada et al. 1996), and auto-generated shoot and dribble microtasks in Half Field Offense that
beat from-scratch learning with curves offset by all source-task time (Narvekar et al. 2016). Start
states alone are a powerful curriculum: reverse curricula grow the start distribution backward from
the goal (Florensa et al. 2017), with theory supplied by restart distributions that cover a good
policy's state visitation (Kakade & Langford 2002); hindsight relabeling is the main non-curriculum
alternative for sparse goals (Andrychowicz et al. 2017). EXP2's drills change only the start-state
distribution inside the same MDP, so we attribute any win to this restart mechanism — not to "task
decomposition" as such — and charge every drill step to the shared budget.

The human part-task literature conditions the prediction rather than guaranteeing it: part training
is most effective when subtask complexity is high and inter-subtask organization low (Naylor &
Briggs 1963); the segmentation wins used backward chaining, training the final segment first
(Wightman & Lintern 1985), yet forward chaining beat backward on a keyboard skill (Ash & Holding
1990), so our shoot-first ordering is one point in a task-dependent space. Curricula also have a
budget regime: across thousands of orderings, curriculum conferred no benefit in standard
supervised settings, helping only under limited budget or label noise (Wu et al. 2021) — so we
separate time-to-threshold claims from asymptotic ones and scope the effect to the tested budget.

## Variable practice and contextual interference

Shea & Morgan (1979) is EXP3's target result: blocked practice looked better during acquisition
(1.32 vs 1.69 s), then the ordering reversed at retention (1.31 vs 1.73 s favoring random
acquisition) and transfer, with the blocked deficit largest under changed test contexts. Schema
theory predicted the related variability-of-practice effect (Schmidt 1975), most strikingly when
varied practice that never included the criterion distance beat practice at the criterion itself
(Kerr & Booth 1978); the umbrella concept is desirable difficulties — conditions that impair
acquisition while enhancing retention and transfer (Bjork 1994). The boundary conditions sit
exactly where our task lives: contextual interference is most reliable when variants require
different motor programs and weaker for parameter-only variation (Magill & Hall 1990);
meta-analytically it is large in laboratory tasks (SMD 0.92) yet null in applied sport (Czyz et al.
2024); and in the closest analog of our paradigm — piano-like sequences practiced under tempo
variability — lower variability and non-random schedules transferred better, reversing the textbook
prediction (Caramiaux et al. 2018). We therefore pre-committed to publishing whichever direction
the in-silico effect takes.

Across ~80 years of motor learning, categorization, perception, language, and ML the synthesis is
consistent: low-variability input is learned fast and generalizes poorly; high-variability input,
the reverse (Raviv et al. 2022). RL operationalizes practice variation as environment diversity:
domain randomization transfers from non-realistic simulation to the real world (Tobin et al. 2017),
CoinRun and Procgen exhibit an explicit train-worse/test-better crossover as level diversity grows,
with severe memorization below thousands of variants (Cobbe et al. 2019; Cobbe et al. 2020), and
small-scale agents memorize fixed configurations by default (Zhang et al. 2018). What that
literature does not do — and EXP3 does — is hold the variant set fixed and manipulate only the
practice schedule, over an engineered shared-feature substrate for interference, with the full
acquisition/retention/transfer battery.

## Generational transmission

Distillation lineages show students exceeding teachers: policy-distillation students match or beat
their teachers at a fraction of the size (Rusu et al. 2016), and born-again networks —
equal-capacity students trained on teacher outputs, generation after generation — outperform their
teachers, with gains arriving in the first few generations and then saturating (Furlanello et al.
2018); self-distillation theory predicts that continued rounds eventually underfit, an inverted-U
over generations (Mobahi et al. 2020). Reincarnating RL formalizes reuse of prior computation but
transfers the full teacher, with no curated bottleneck and no teacher aging (Agarwal et al. 2022);
distilling into a freshly initialized network within one lineage repairs damage from early
non-stationarity (Igl et al. 2021); and imitating one's own top-return episodes helps within a
single lifetime (Oh et al. 2018) — a confound any generational claim must separate from the
fresh-student step.

The iterated-learning tradition supplies the theory of transmission itself: human diffusion chains
learning artificial languages through a bottleneck become more learnable and more structured
without anyone intending it (Kirby et al. 2008); for posterior-sampling learners the chain
converges to the learner's prior, bottleneck size setting the rate, not the endpoint (Griffiths &
Kalish 2007); and structure requires joint pressure for compressibility and expressivity —
transmission-only chains collapse to degenerate systems (Kirby et al. 2015). In our chains the
student's post-imitation reward-driven lifetime is that expressivity pressure. Cultural
transmission has reached deep RL as a single expert-to-novice step acquired zero-shot (Bhoopchand
et al. 2023) and as generational accumulation that beats one long life at matched cumulative
experience (Cook et al. 2024); human chains given two lives per generation accumulate by passing
distilled written messages (Tessler et al. 2021) — the nearest relative of our 100-pair advice
artifact. The null model is structural drift: resampling chains can wander into degenerate
absorbing states, so cumulative improvement is not guaranteed (Crutchfield & Whalen 2010).

The aging half of EXP4 is motivated by plasticity loss: deep continual learners progressively lose
the ability to learn at all (Dohare et al. 2024). But the same literature raises the
identifiability question our controls exist to answer: periodically resetting network parameters
while keeping the replay buffer already improves deep RL by curing primacy bias (Nikishin et al.
2022), so an apparent generational benefit could be "just resets" — restored plasticity rather than
transmitted content. EXP4's battery targets exactly this: no-inheritance lineages (fresh agents, no
advice) isolate the reset effect alone; weight-copy separates inherited content from inherited
freshness; and one-long-life controls with matched and slowed plasticity decay price the cost of
aging. We present teacher aging as a model inspired by deep-network plasticity loss, not a
phenomenon intrinsic to tabular learners — which is precisely what keeps the testbed clean.

## Morphological development

Bongard (2011) found that robots whose body plans changed during evolution found successful
controllers faster and ended more robust — but by evolutionary search over controllers, not
within-lifetime RL, which is exactly the transfer our experiment tests. Developmental robotics
articulates the rationale of immaturity as a dimensionality-reducing scaffold (Lungarella et al.
2003), with the complication that a single monotonic freeze-then-free schedule can be insufficient
(Berthouze & Lungarella 2004). The empirical record is honestly mixed: across five
morphological-development strategies on a bipedal walker, development helped in some setups and
hurt in others (Naya-Varela et al. 2023), while the strongest recent positive result required
realistic child-to-adult anthropometry — uniformly scaled morphology failed (Badie et al. 2025).
Morphology causally modulates learnability (Gupta et al. 2021), and transferring policies through a
continuous sequence of intermediate bodies beats abrupt transfer (Liu et al. 2022) — the prior for
grow-linear over grow-jump.

The infant data license our cost model: walking infants average ~2,400 steps and 17 falls per hour
(Adolph et al. 2012), and falling is cheap — over 90% of spontaneous infant falls are uneventful,
and measured fall energy is 18.4x lower than the same falls at adult size and speed (Han & Adolph
2021), the anchor for our (s/s_adult)^4 damage law. The counter-evidence is equally real: balance
knowledge can fail to transfer across postures, with infants avoiding risky gaps while sitting yet
falling into them while crawling (Adolph 2000), so a balance-first null would be developmentally
consistent. We also model the physics honestly: the inverted pendulum's time constant sqrt(l/g)
means the small body falls faster and is harder for any delay-limited controller (Milton et al.
2009); smallness's true advantages are the L^4 fall cost and relative actuation authority, and we
disclose that direction rather than rig it. Standing precedes walking in the WHO milestone windows
(WHO Multicentre Growth Reference Study Group 2006), the empirical basis for balance-first; and
assistive forces annealed to zero (Yu et al. 2018) are the established non-morphological easy-start
against which "growing the body specifically helps" would ultimately have to be tested.

## Positioning

Much of this program is deliberate replication: preregistered developmental-psychology and
motor-learning effects — latent learning, part-task curricula, the Shea & Morgan crossover, cheap
infant falls — reproduced in minimal, fully inspectable RL systems where every value table, model,
belief, and advice set can be printed and audited. Against that baseline we claim four protocols as
new. (a) The blindfold evaluation of a learned world model: prior work trains blind agents
end-to-end (Wijmans et al. 2023) or evaluates models by dream-training transfer (Ha & Schmidhuber
2018); using an already-learned Dyna model as the substrate for belief-filtered dead reckoning,
with home-versus-stranger and touch/no-touch contrasts as the discriminating controls, is a new
evaluation protocol rather than a new algorithm. (b) Iterated distillation through an
episodic-memory bottleneck with plasticity-decaying teachers: the pieces exist separately —
generational accumulation (Cook et al. 2024), zero-shot cultural transmission (Bhoopchand et al.
2023), full-policy reuse (Agarwal et al. 2022), born-again distillation (Furlanello et al. 2018),
fresh-network distillation within a lineage (Igl et al. 2021), resets whose transmission channel is
the entire unfiltered buffer (Nikishin et al. 2022), and the bottleneck theory of Kirby et al.
(2008) and Griffiths & Kalish (2007) — but to the best of our review no published work combines an
aging teacher, fresh tabular students, and a hard countable trajectory cap with the weight-copy /
long-life / no-inheritance causal battery. (c) In-silico contextual interference via engineered
shared-feature interference: the RL generalization literature varies environment diversity and
measures gaps (Cobbe et al. 2019) but does not run the Shea & Morgan schedule manipulation over a
fixed variant set with a mechanism hypothesis and an acquisition-retention-transfer battery. (d)
Morphological growth curricula with square-cube-honest physics: prior art is evolutionary (Bongard
2011), neuroevolutionary with mixed results (Naya-Varela et al. 2023), or deep and muscle-actuated
(Badie et al. 2025); none runs within-lifetime tabular RL with a disclosed sqrt(l/g) difficulty
direction, L^4 fall-cost scaling anchored to Han & Adolph (2021), and cumulative damage as the
headline metric. Where we replicate, we say so loudly; where we claim novelty, we claim exactly
these four protocols and no more.
