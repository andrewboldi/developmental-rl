"""Layout generators for EXP7 — layout-resampling robustness (critic P7).

The two headline results were measured on single fixed layouts (TRAP_MAP,
HOME_A/HOME_B), which scopes their conclusions to those instances (Whiteson
et al. 2011 environment overfitting). These generators resample layouts from
the same design family so the headline pairings can be replicated across
fresh instances:

- ``gen_trapgrid``: the original 15x11 open shell with the standard S (left)
  and G (far right), but the 3 terminal candy cells placed at randomized
  left-third floor cells (never on or Chebyshev-adjacent to S). A candidate
  is accepted only if it is BFS-connected, S->G solvable dodging candy, and
  passes the calibration guard: a uniform random walk is absorbed into candy
  in [80%, 98%] of 2000 probe episodes — the trap must throttle exploration
  the way the original does (its rate is ~0.93). Reject+resample otherwise,
  cap 50 tries.

- ``gen_home_pair``: two 13x11 homes with the original shell and the
  original S/G coordinates, interior walls from a random room partition —
  recursive splits with door openings, parity-constrained (walls on even
  rows/cols, doors on odd) so no door can ever be plugged or approach-blocked
  by a later perpendicular wall; full floor connectivity holds by
  construction and is verified anyway. Both maps must be BFS-solvable with
  shortest path in [10, 25] (the originals are 17 and 11) and the pair must
  differ in >= 10 wall cells.

Both generators draw only from the rng passed in, so a fixed seed fully
determines the instance.
"""

from collections import deque

import numpy as np

from devrl.envs.gridhome import GridHome
from devrl.envs.trapgrid import TrapGrid

# --- TrapGrid family (matches devrl.envs.trapgrid.TRAP_MAP) ---------------
TRAP_SHAPE = (11, 15)          # (rows, cols)
TRAP_START = (5, 1)            # standard left S
TRAP_GOAL = (5, 13)            # standard far-right G
LEFT_THIRD_MAX_COL = 5         # 15 // 3: candy columns 1..5 (original: 3..5)
ABSORPTION_BAND = (0.80, 0.98)

# --- GridHome family (matches HOME_A / HOME_B) ----------------------------
HOME_SHAPE = (11, 13)
HOME_START = (1, 1)
HOME_GOAL = (2, 7)
HOME_SP_BAND = (10, 25)        # originals: HOME_A 17, HOME_B 11
MIN_WALL_DIFF = 10

_MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))


def wall_cells(map_str):
    """{(row, col)} of '#' cells in a map string."""
    return {(r, c) for r, row in enumerate(map_str.strip().splitlines())
            for c, ch in enumerate(row) if ch == "#"}


def floor_connected(map_str):
    """True iff every non-wall cell is BFS-reachable from S (pure adjacency;
    candy/goal termination is ignored — this checks the graph, not play)."""
    rows = map_str.strip().splitlines()
    floors = {(r, c) for r, row in enumerate(rows)
              for c, ch in enumerate(row) if ch != "#"}
    start = next((r, c) for r, row in enumerate(rows)
                 for c, ch in enumerate(row) if ch == "S")
    seen, q = {start}, deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in _MOVES:
            nxt = (r + dr, c + dc)
            if nxt in floors and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen == floors


def candy_absorption(map_str, rng, episodes=2000, cap=120):
    """Fraction of uniform-random-walk episodes absorbed by a candy cell.

    An episode ends on any terminal (candy or goal) or truncates at ``cap``
    steps (the standard TrapGrid cap); only candy endings count."""
    env = TrapGrid(map_str, cap=cap)
    hits = 0
    for _ in range(episodes):
        env.reset()
        while True:
            _, _, done, info = env.step(int(rng.integers(env.n_actions)))
            if done:
                hits += env.pos in env.candies
                break
            if info["truncated"]:
                break
    return hits / episodes


def _trap_map_from(candies):
    h, w = TRAP_SHAPE
    rows = []
    for r in range(h):
        row = []
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                row.append("#")
            elif (r, c) == TRAP_START:
                row.append("S")
            elif (r, c) == TRAP_GOAL:
                row.append("G")
            elif (r, c) in candies:
                row.append("C")
            else:
                row.append(".")
        rows.append("".join(row))
    return "\n".join(rows)


def gen_trapgrid(rng, probe_episodes=2000, max_tries=50, band=ABSORPTION_BAND):
    """One resampled TrapGrid map string (see module docstring)."""
    h, _ = TRAP_SHAPE
    sr, sc = TRAP_START
    eligible = [(r, c) for r in range(1, h - 1)
                for c in range(1, LEFT_THIRD_MAX_COL + 1)
                if max(abs(r - sr), abs(c - sc)) > 1]
    lo, hi = band
    for _ in range(max_tries):
        idx = rng.choice(len(eligible), size=3, replace=False)
        map_str = _trap_map_from({eligible[int(i)] for i in idx})
        if not floor_connected(map_str):
            continue
        if TrapGrid(map_str).shortest_path_len() is None:
            continue
        if lo <= candy_absorption(map_str, rng, probe_episodes) <= hi:
            return map_str
    raise RuntimeError(
        "gen_trapgrid: no candidate passed the calibration guard "
        f"(absorption in [{lo}, {hi}]) within {max_tries} tries")


def _carve_home(rng):
    """One parity-constrained recursive-division home map.

    Walls go on even rows/cols, doors on odd ones: a door cell's off-wall
    neighbours are odd-odd cells no wall can ever cover, so doors are never
    plugged and the floor stays fully connected by construction. S is odd-odd
    (never walled); G sits on an even row, so carving it out of a horizontal
    wall punches an extra odd-column opening — a legal door."""
    h, w = HOME_SHAPE
    grid = [["#" if r in (0, h - 1) or c in (0, w - 1) else "."
             for c in range(w)] for r in range(h)]

    def split(r0, r1, c0, c1):
        wall_rows = [r for r in range(r0 + 1, r1) if r % 2 == 0]
        wall_cols = [c for c in range(c0 + 1, c1) if c % 2 == 0]
        if not wall_rows and not wall_cols:
            return
        if wall_rows and wall_cols:
            height, width = r1 - r0 + 1, c1 - c0 + 1
            if height > width:
                horizontal = True
            elif width > height:
                horizontal = False
            else:
                horizontal = bool(rng.integers(2))
        else:
            horizontal = bool(wall_rows)
        if horizontal:
            wr = wall_rows[int(rng.integers(len(wall_rows)))]
            doors = [c for c in range(c0, c1 + 1) if c % 2 == 1]
            dc = doors[int(rng.integers(len(doors)))]
            for c in range(c0, c1 + 1):
                if c != dc:
                    grid[wr][c] = "#"
            split(r0, wr - 1, c0, c1)
            split(wr + 1, r1, c0, c1)
        else:
            wc = wall_cols[int(rng.integers(len(wall_cols)))]
            doors = [r for r in range(r0, r1 + 1) if r % 2 == 1]
            dr = doors[int(rng.integers(len(doors)))]
            for r in range(r0, r1 + 1):
                if r != dr:
                    grid[r][wc] = "#"
            split(r0, r1, c0, wc - 1)
            split(r0, r1, wc + 1, c1)

    split(1, h - 2, 1, w - 2)
    grid[HOME_START[0]][HOME_START[1]] = "S"
    grid[HOME_GOAL[0]][HOME_GOAL[1]] = "G"
    return "\n".join("".join(row) for row in grid)


def _gen_home(rng, max_tries, sp_band):
    lo, hi = sp_band
    for _ in range(max_tries):
        map_str = _carve_home(rng)
        if not floor_connected(map_str):
            continue
        env = GridHome(map_str, rng=np.random.default_rng(0))
        sp = env.shortest_path_len()
        if sp is not None and lo <= sp <= hi:
            return map_str
    raise RuntimeError(
        f"gen_home: no layout with shortest path in [{lo}, {hi}] "
        f"within {max_tries} tries")


def gen_home_pair(rng, max_tries=50, sp_band=HOME_SP_BAND,
                  min_diff=MIN_WALL_DIFF):
    """Two resampled home map strings (see module docstring)."""
    map_a = _gen_home(rng, max_tries, sp_band)
    for _ in range(max_tries):
        map_b = _gen_home(rng, max_tries, sp_band)
        if len(wall_cells(map_a) ^ wall_cells(map_b)) >= min_diff:
            return map_a, map_b
    raise RuntimeError(
        f"gen_home_pair: no second layout differing in >= {min_diff} wall "
        f"cells within {max_tries} tries")
