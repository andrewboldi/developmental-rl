"""Build site/data.js from results/*.json — downsampled, site-shaped, inline.

The website only embeds what it draws. Curves keep every checkpoint (they are
small); trajectories/traces are trimmed; floats are rounded to keep the
payload lean. Output: `window.DATA = {...}` as a JS literal.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = lambda n: json.load(open(ROOT / "results" / f"exp{n}.json"))


def rnd(x, p=4):
    if isinstance(x, float):
        return round(x, p)
    if isinstance(x, list):
        return [rnd(v, p) for v in x]
    if isinstance(x, dict):
        return {k: rnd(v, p) for k, v in x.items()}
    return x


def curve(c, keys=("iqm", "ci_lo", "ci_hi")):
    return {k: rnd(c[k], 3) for k in keys if k in c}


def exp1(j):
    v = j["viz"]
    trajs = {}
    for cond in ("blind-A", "blind-A-touch", "blind-B", "blind-B-touch"):
        # one success example if any, else the first; trim path to 60 steps
        cands = v["blind_trajectories"][cond]
        pick = next((t for t in cands if t["success"]), cands[0])
        trajs[cond] = {
            "success": pick["success"],
            "steps": pick["steps"],
            "path": [
                {"t": s["t"], "true": s["true"], "bel": s["believed"],
                 "H": rnd(s["entropy_bits"], 2), "bump": s["bump"]}
                for s in pick["path"][:60]
            ],
        }
    prim = [t for t in j["tests"] if t.get("family") == "primary"]
    return {
        "homes": v["homes"],
        "slip": v["slip"],
        "summary": rnd(v["blind_summary"], 3),
        "curves": {
            "checkpoints": j["curves"]["checkpoints"],
            "dyna": curve(j["curves"]["conditions"]["dynaq-A"]),
            "q": curve(j["curves"]["conditions"]["qlearning-A"]),
        },
        "t90": rnd(v["sample_efficiency"]["t90"], 1),
        "trajs": trajs,
        "tests": rnd([{k: t[k] for k in ("name", "iqm_a", "iqm_b", "p_holm")} for t in prim], 6),
        "conclusion": j["conclusion"],
    }


def exp2(j):
    v = j["viz"]
    ec = v["eval_curves"]
    trajs = {}
    for cond, stages in v["trajectories"].items():
        trajs[cond] = {}
        for stage, t in stages.items():
            trajs[cond][stage] = {
                "scored": t["scored"],
                "steps": [{"agent": s["agent"], "ball": s["ball"], "a": s["a"]}
                          for s in t["steps"][:70]],
            }
    prim = j["tests"]["primary"] if isinstance(j["tests"], dict) else j["tests"]
    return {
        "pitch": v["pitch"],
        "curves": {"checkpoints": ec["checkpoints"], "threshold": ec["threshold"],
                   "conditions": {k: curve(c) for k, c in ec["conditions"].items()}},
        "trajs": trajs,
        "tests": rnd(prim, 6),
        "conclusion": j["conclusion"],
    }


def exp3(j):
    v = j["viz"]
    tests = {k: rnd({kk: t[kk] for kk in ("iqm", "p_holm", "significant") if kk in t}, 6)
             for k, t in j["tests"].items() if isinstance(t, dict)}
    return {
        "structure": {k: v["example_structure"][k] for k in ("motif_table", "passages", "exceptions", "correct_keys")},
        "n_keys": v["n_keys"], "passage_names": v["passage_names"],
        "phase_boundaries": v["phase_boundaries_episodes"],
        "practice": rnd(v["practice_curves"], 3),
        "retention": rnd(v["retention_bars"], 3),
        "transfer": rnd(v["transfer_bars"], 3),
        "eval_curves": {"episodes": j["curves"]["episodes"],
                        "acquisition": {c: curve(j["curves"]["metrics"]["acquisition"][c]) for c in ("blocked", "interleaved")}
                        if "acquisition" in j["curves"]["metrics"] else None},
        "tests": tests,
        "conclusion": j["conclusion"],
    }


def exp4(j):
    v = j["viz"]
    pg = v["per_generation_curves"]
    conds = list(pg["greedy_return"]["conditions"].keys())
    return {
        "grid": v["grid"],
        "gens": pg["greedy_return"]["steps"],
        "greedy": {c: rnd(pg["greedy_return"]["conditions"][c], 3) for c in conds},
        "big_goal": {c: rnd(pg["big_goal"]["conditions"][c], 3) for c in conds},
        "gap": {c: rnd(pg["gap"]["conditions"][c], 3) for c in conds},
        "advice": [{"generation": a["generation"], "n_pairs": a["n_pairs"],
                    "pairs": [{"rc": p["rc"], "action": p["action"]} for p in a["pairs"]]}
                   for a in v["advice_by_generation"]],
        "final_paths": v["final_greedy_paths"],
        "tests": rnd([t for t in j["tests"] if (t.get("family") == "primary" if isinstance(t, dict) else False)] or j["tests"], 6),
        "conclusion": j["conclusion"],
    }


def exp5(j):
    v = j["viz"]
    conds = v["meta"]["conditions"] if "conditions" in v.get("meta", {}) else list(v["size_schedules"].keys())
    out = {"conds": conds, "meta": v.get("meta", {}), "sizes": {}, "damage": {}, "theta": {}, "falls": {}, "eval": {}}
    for c in conds:
        ss = v["size_schedules"][c]
        out["sizes"][c] = {"steps": ss["steps"], "iqm": rnd(ss["iqm"], 3),
                           "example": rnd(ss.get("example_s", []), 3)}
        d = v["damage_vs_competence"][c]
        out["damage"][c] = {"steps": d["steps"], "iqm": rnd(d["damage_iqm"], 2),
                            "lo": rnd(d["damage_lo"], 2), "hi": rnd(d["damage_hi"], 2)}
        th = v["theta_traces"][c]["theta"][:200]
        out["theta"][c] = rnd(th, 3)
        fe = v["fall_events"][c]
        stride = max(1, len(fe) // 150)
        out["falls"][c] = [{"step": f["step"], "size": rnd(f["size"], 2), "dmg": rnd(f["damage"], 3)}
                           for f in fe[::stride]][:150]
        cc = j["curves"][c]
        out["eval"][c] = {"steps": cc["steps"], "iqm": rnd(cc["iqm"], 1),
                          "lo": rnd(cc["lo"], 1), "hi": rnd(cc["hi"], 1)}
    out["tests"] = rnd(j["tests"], 6)
    out["conclusion"] = j["conclusion"]
    return out


def main():
    data = {
        "exp1": exp1(R(1)), "exp2": exp2(R(2)), "exp3": exp3(R(3)),
        "exp4": exp4(R(4)), "exp5": exp5(R(5)),
    }
    js = "window.DATA = " + json.dumps(data, separators=(",", ":")) + ";\n"
    out = ROOT / "site" / "data.js"
    out.write_text(js)
    print(f"wrote {out} ({len(js)/1024:.0f} KB)")
    for k, v in data.items():
        print(f"  {k}: {len(json.dumps(v))/1024:.0f} KB")


if __name__ == "__main__":
    main()
