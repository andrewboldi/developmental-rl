# Growing Up to Learn: Developmental Scaffolds for Reinforcement Learning

An experimental research program testing five hypotheses about how RL should sit
**between imitation and evaluation** — the way humans learn: from instruction,
world models, drills, teachers, and growing bodies.

**Interactive results:** https://andrewboldi.github.io/developmental-rl/ ·
**Paper:** [`paper/paper.pdf`](paper/paper.pdf) ·
**Literature synthesis:** [`RESEARCH.md`](RESEARCH.md) ·
**Versioned protocols:** [`DESIGN.md`](DESIGN.md)

## The five hypotheses — as the data left them

| # | Name | What survived fresh-seed, Holm-corrected, dual-statistic verification |
|---|------|------|
| H1 | **The Blindfold Test** | With a learned touch filter, blind navigation at "home" matches sighted (0.975 vs 0.973) and collapses in a "stranger's home" (0.290, floor 0.033). **Refuted honestly:** the famous Dyna speedup — an update-matched replay control closes 100% of it. The model's unique value is acting blind, not learning fast. |
| H2 | **Microtasks** | Start-state drills reach full-game mastery in 15,125 steps vs 40,375 (p≤4e-9). Uniform start diversity is worthless (p=0.39 vs whole) — targeted structure carries it. **Boundary:** optimistic initialization reproduces the whole effect; drills pay for undirected explorers. |
| H3 | **Variation Practice** | The Shea–Morgan crossover, counterbalanced: interleaved wins retention (0.910 vs 0.793, p=5e-10) and transfer (0.779 vs 0.621). **Mechanism proven by ablation** — remove shared features and the crossover vanishes (p=0.92). Acquisition edge: boundary. |
| H4 | **Generational Teaching** | Episodic-bottleneck distillation ratchets to 10.0 IQM (48/60 lineages) while weight-copy, long lives, and random optimistic advice all fail (p≤2e-20; random advice *poisons* to 0.0). **Boundary:** global optimism solves this small world outright — teaching pays where optimism is unaffordable. |
| H5 | **Growing Bodies** | Confound-free pairing: growth reaches adult competence with 9–42× less fall damage AND 30–40% fewer steps, robust to physics variants. **Reversed:** gradualism (abrupt growth is cheaper) and balance-first staging (walking-from-the-start wins). |
| E6 | **The Self-Coach** | With no teacher, restarting from moments in its own best episodes makes the agent 28% faster than raw play (Holm p=.02). **Refuted:** self matches teacher — the designed curriculum is another ~2× faster (p≤2.5e-6): the measured value of instruction. |
| E7 | **Layout robustness** | Both headline effects replicate on freshly generated worlds: blindfold 5/5 home pairs (pooled Welch p=2e-30), generational teaching 4/5 TrapGrids (miss direction-consistent). Plus the decisive EXP4 control: fresh students replaying entire goal-bearing teacher lifetimes consolidate **0/35** vs distill's **129/129** — curation is causal. |

Every claim above is scored by a pre-registered rule on fresh seeds disjoint
from all tuning, with both Mann-Whitney and Welch tests, Holm-corrected. Seven
registered claims were refuted, two landed on boundaries — all reported, none
deleted. The program went through a 16-agent adversarial verification pass and
a mandated hardening round; the audit trail lives in the PR history.

## Layout

```
src/devrl/        core library: environments, agents, harness, stats
experiments/      one runnable script per hypothesis (v2, with controls)
tests/            pytest suite (321 tests)
results/          per-seed JSON produced by the committed scripts
paper/            manuscript (markdown -> 17pp PDF), figures, bibliography
docs/             Three.js + GSAP interactive site (GitHub Pages), real data
```

## Reproduce

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest                                  # 321 tests
python -m devrl.run_all                 # rerun every experiment, all seeds
python docs/build_data.py               # regenerate the site's data
python paper/make_figures.py            # regenerate the paper's figures
(cd paper && python assemble.py && ./build.sh)   # rebuild the PDF
```

Single experiment, custom scale:

```bash
python experiments/exp4_generations.py --seeds 60 --seed-offset 100 --out results/exp4.json
```
