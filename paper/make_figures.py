"""Render paper figures from results/*.json into paper/figs/*.pdf."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(exist_ok=True)

C = {"blue": "#4C86D8", "orange": "#C77B2F", "green": "#4F9E6E",
     "violet": "#AA64C8", "gold": "#AD8F2E", "red": "#C75F6B", "ink": "#222"}
ORDER = [C["blue"], C["orange"], C["green"], C["violet"], C["gold"], C["red"]]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
    "grid.linewidth": .5, "legend.frameon": False, "figure.dpi": 150,
})

R = lambda n: json.load(open(ROOT / "results" / f"exp{n}.json"))


def curveband(ax, x, y, lo, hi, color, label):
    ax.plot(x, y, color=color, lw=1.6, label=label)
    if lo and hi:
        ax.fill_between(x, lo, hi, color=color, alpha=.15, lw=0)


def fig1():
    j = R(1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    cur = j["curves"]["conditions"]
    names = {"dynaq-A": ("Dyna-Q", C["blue"]), "qlearning-A": ("Q-learning", C["orange"]),
             "replayq-A": ("Q + replay (update-matched)", C["green"])}
    for k, (nm, col) in names.items():
        if k in cur:
            c = cur[k]
            curveband(a, j["curves"]["checkpoints"], c["iqm"], c.get("ci_lo"), c.get("ci_hi"), col, nm)
    a.axhline(.9, color=C["gold"], ls="--", lw=.8)
    a.set(xlabel="training steps", ylabel="eval success", title="(a) Sample efficiency, HOME A")
    a.legend(fontsize=7.5, loc="lower right")
    summ = j["viz"]["blind_summary"]
    names2 = [s["name"] for s in summ]
    vals = [s["success_iqm"] for s in summ]
    err = [[max(0, s["success_iqm"] - s["success_ci"][0]) for s in summ],
           [max(0, s["success_ci"][1] - s["success_iqm"]) for s in summ]]
    cols = [C["green"] if "touch" in n else C["blue"] if "blind" in n else C["gold"] if "sighted" in n else C["red"] for n in names2]
    b.bar(range(len(vals)), vals, yerr=err, color=cols, width=.7, capsize=2, error_kw={"lw": .7})
    b.set_xticks(range(len(vals)), names2, fontsize=5.6, rotation=35, ha="right")
    b.set(ylabel="success rate", title="(b) The blindfold test")
    fig.tight_layout(); fig.savefig(FIGS / "fig1_blindfold.pdf"); plt.close(fig)


def fig2():
    j = R(2)
    ec = j["viz"]["eval_curves"]
    fig, a = plt.subplots(figsize=(4.6, 2.7))
    for i, (k, c) in enumerate(ec["conditions"].items()):
        curveband(a, ec["checkpoints"], c["iqm"], c.get("ci_lo"), c.get("ci_hi"), ORDER[i % 6], k)
    a.axhline(ec.get("threshold", .9), color=C["gold"], ls="--", lw=.8)
    a.set(xlabel="training steps", ylabel="full-game success", title="Reverse start-state curriculum")
    a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "fig2_drills.pdf"); plt.close(fig)


def fig3():
    j = R(3)
    v = j["viz"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [3, 2]})
    pc = v.get("practice_curves", {})
    eps = pc.get("episodes", [])
    for nm, col in [("blocked", C["orange"]), ("interleaved", C["blue"])]:
        if nm in pc:
            y = pc[nm]["iqm"] if isinstance(pc[nm], dict) else pc[nm]
            a.plot(eps[:len(y)], y, color=col, lw=1.5, label=nm)
    a.set(xlabel="practice episodes", ylabel="score on practiced passage", title="(a) Acquisition")
    a.legend(fontsize=7.5)
    rb, tb = v["retention_bars"], v["transfer_bars"]
    groups = ["retention", "transfer"]
    for i, (nm, col) in enumerate([("blocked", C["orange"]), ("interleaved", C["blue"])]):
        rv = rb[nm]["iqm"][-1] if isinstance(rb[nm], dict) else rb[nm]
        tv = tb[nm]["iqm"] if isinstance(tb[nm], dict) and "iqm" in tb[nm] else tb[nm]
        tv = tv if isinstance(tv, (int, float)) else tv[-1]
        b.bar([x + (i - .5) * .32 for x in range(2)], [rv, tv], width=.3, color=col, label=nm)
    b.set_xticks(range(2), groups)
    b.set(ylabel="test score", title="(b) Test, learning off", ylim=(0, 1))
    b.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIGS / "fig3_variation.pdf"); plt.close(fig)


def fig4():
    j = R(4)
    pg = j["viz"]["per_generation_curves"]["greedy_return"]
    fig, a = plt.subplots(figsize=(4.6, 2.7))
    picks = [("generational-distill", C["blue"]), ("optimistic-init", C["red"]),
             ("weight-copy", C["orange"]), ("random-advice", C["violet"]),
             ("reset-replay-full", C["gold"]), ("no-inheritance", C["green"])]
    for k, col in picks:
        c = pg["conditions"].get(k)
        if not c:
            continue
        gens = list(range(1, len(c["iqm"]) + 1))
        curveband(a, gens, c["iqm"], c.get("ci_lo"), c.get("ci_hi"), col, k)
    a.set(xlabel="generation", ylabel="greedy return at generation end",
          title="Generational teaching", xticks=list(range(1, 6)))
    a.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(FIGS / "fig4_generations.pdf"); plt.close(fig)


def fig5():
    j = R(5)
    v = j["viz"]
    fig, a = plt.subplots(figsize=(4.9, 2.7))
    for i, (k, d) in enumerate(v["damage_vs_competence"].items()):
        a.plot(d["steps"], d["damage_iqm"], color=ORDER[i % 6], lw=1.5, label=k)
    a.set(xlabel="training steps", ylabel="cumulative fall damage (IQM)",
          title="The price of practice, by body plan")
    a.legend(fontsize=6.2)
    fig.tight_layout(); fig.savefig(FIGS / "fig5_growing.pdf"); plt.close(fig)


def fig6():
    j = R(6)
    ec = j["viz"]["eval_curves"]
    fig, a = plt.subplots(figsize=(4.6, 2.7))
    picks = [("teacher-drills", C["blue"]), ("self-drills", C["green"]),
             ("self-drills-late", C["gold"]), ("whole", C["orange"])]
    for k, col in picks:
        c = ec["conditions"].get(k)
        if not c:
            continue
        curveband(a, ec["checkpoints"], c["iqm"], c.get("ci_lo"), c.get("ci_hi"), col, k)
    a.axhline(.9, color=C["gold"], ls="--", lw=.8)
    a.set(xlabel="training steps", ylabel="full-game success", title="The Self-Coach")
    a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "fig6_selfcoach.pdf"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        try:
            f()
            print(f"{f.__name__} ok")
        except Exception as e:
            print(f"{f.__name__} FAILED: {e}")
