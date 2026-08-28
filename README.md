# Growing Up to Learn: Developmental Scaffolds for Reinforcement Learning

An experimental research program testing five hypotheses about how RL should sit
**between imitation and evaluation** — the way humans learn: from instruction,
world models, drills, teachers, and growing bodies.

## The five hypotheses

| # | Name | Claim |
|---|------|-------|
| H1 | **The Blindfold Test** | An agent with a learned world model can act without observations in a *familiar* environment but not a *strange* one; model-based learning (Dyna) dominates model-free in sample efficiency. |
| H2 | **Microtasks** | Drill-based part-training (shoot, then dribble, then play) reaches whole-task mastery in fewer environment steps than whole-task-only training. |
| H3 | **Variation Practice** | Varied/interleaved practice looks *worse during training* but *wins at test* under novel perturbation — the contextual-interference effect, in RL form. |
| H4 | **Generational Teaching** | Iterated teach-and-relearn — an aging agent distills principles to a fresh, plastic student — beats both weight-copying and one agent trained for the combined budget. |
| H5 | **Growing Bodies** | Starting small (cheap falls, quick dynamics) and growing toward the adult body, balance-first, beats training the adult body directly. |

Every hypothesis is validated with multi-seed runs, confidence intervals, and
significance tests. Results feed a paper (`paper/`) and an interactive
Three.js + GSAP scroll experience (`site/`).

## Layout

```
src/devrl/        core library: environments, agents, harness, stats
experiments/      one runnable script per hypothesis
tests/            pytest suite
results/          JSON + figures produced by real runs
paper/            manuscript (markdown -> PDF)
site/             Three.js + GSAP visualization, driven by results/ data
```

## Reproduce

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
python -m devrl.run_all          # runs every experiment, all seeds
```
