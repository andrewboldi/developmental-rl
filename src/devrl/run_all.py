"""Run every experiment in sequence: python -m devrl.run_all [--smoke]

Extra arguments are passed through to every experiment script.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    scripts = sorted((ROOT / "experiments").glob("exp*.py"))
    extra = sys.argv[1:]
    for s in scripts:
        out = ROOT / "results" / (s.stem.split("_")[0] + (".smoke.json" if "--smoke" in extra else ".json"))
        cmd = [sys.executable, str(s), "--out", str(out), *extra]
        print(f"\n=== {s.name} ===", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
