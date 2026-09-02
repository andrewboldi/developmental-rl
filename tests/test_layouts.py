"""Layout-generator tests for EXP7 (layout-resampling robustness, critic P7).

Pins the registered generation contract from DESIGN.md "Amendments (v3)":

- gen_trapgrid: 15x11 shell, standard S/G, 3 candies at randomized left-third
  floor cells (not on or Chebyshev-adjacent to S), BFS-connected, and a
  calibration guard — random-walk absorption into candy within [0.80, 0.98]
  over 2k probe episodes (reject+resample, cap 50 tries).
- gen_home_pair: two 13x11 maps, original shell and S/G coordinates, interior
  walls from a random room partition (parity-constrained recursive split with
  door openings), both BFS-solvable with shortest path in [10, 25], differing
  in >= 10 wall cells.
"""

import numpy as np
import pytest

from devrl.envs.gridhome import HOME_A, HOME_B, GridHome
from devrl.envs.layouts import (ABSORPTION_BAND, HOME_GOAL, HOME_SHAPE,
                                HOME_SP_BAND, HOME_START, LEFT_THIRD_MAX_COL,
                                MIN_WALL_DIFF, TRAP_GOAL, TRAP_SHAPE,
                                TRAP_START, candy_absorption, floor_connected,
                                gen_home_pair, gen_trapgrid, wall_cells)
from devrl.envs.trapgrid import TRAP_MAP, TrapGrid


def _border_and_interior_walls(grid):
    H, W = grid.shape
    border_ok = (all(grid[0, c] == "#" and grid[H - 1, c] == "#"
                     for c in range(W))
                 and all(grid[r, 0] == "#" and grid[r, W - 1] == "#"
                         for r in range(H)))
    interior_walls = {(r, c) for r in range(1, H - 1) for c in range(1, W - 1)
                      if grid[r, c] == "#"}
    return border_ok, interior_walls


# ------------------------------------------------------------- gen_trapgrid

def test_trapgrid_constants_match_original_env():
    env = TrapGrid()
    assert TRAP_SHAPE == env.shape == (11, 15)
    assert TRAP_START == env.start
    assert TRAP_GOAL == env.goal
    assert LEFT_THIRD_MAX_COL == 15 // 3  # original candies live in cols 3..5


def test_gen_trapgrid_matches_shell_start_goal_and_candy_count():
    env = TrapGrid(gen_trapgrid(np.random.default_rng(0)))
    assert env.shape == (11, 15)
    assert env.start == TRAP_START and env.goal == TRAP_GOAL
    assert len(env.candies) == 3
    border_ok, interior_walls = _border_and_interior_walls(env.grid)
    assert border_ok
    assert interior_walls == set()  # open shell: walls only on the border


def test_gen_trapgrid_candies_left_third_not_adjacent_to_start():
    for seed in range(3):
        env = TrapGrid(gen_trapgrid(np.random.default_rng(seed)))
        for r, c in env.candies:
            assert 1 <= c <= LEFT_THIRD_MAX_COL
            assert 1 <= r <= env.H - 2
            cheb = max(abs(r - TRAP_START[0]), abs(c - TRAP_START[1]))
            assert cheb > 1  # not on S, not (even diagonally) adjacent


def test_gen_trapgrid_connected_and_solvable():
    env = TrapGrid(gen_trapgrid(np.random.default_rng(1)))
    assert env.shortest_path_len() is not None  # S->G dodging candy
    assert floor_connected(gen_trapgrid(np.random.default_rng(1)))


def test_gen_trapgrid_deterministic_in_rng_and_varies_across_seeds():
    a = gen_trapgrid(np.random.default_rng(5))
    b = gen_trapgrid(np.random.default_rng(5))
    c = gen_trapgrid(np.random.default_rng(6))
    assert a == b
    assert TrapGrid(a).candies != TrapGrid(c).candies


def test_gen_trapgrid_absorption_guard_holds_on_accepted_maps():
    lo, hi = ABSORPTION_BAND
    assert (lo, hi) == (0.80, 0.98)
    m = gen_trapgrid(np.random.default_rng(2))
    # independent re-measure with a fresh rng; 2k episodes -> sd ~ 0.007,
    # so +-0.02 slack around the registered band covers probe noise
    rate = candy_absorption(m, np.random.default_rng(999), episodes=2000)
    assert lo - 0.02 <= rate <= hi + 0.02


def test_candy_absorption_brackets_the_original_map():
    # the guard band must contain the original TRAP_MAP's absorption rate
    rate = candy_absorption(TRAP_MAP, np.random.default_rng(3), episodes=2000)
    assert ABSORPTION_BAND[0] <= rate <= ABSORPTION_BAND[1]


def test_gen_trapgrid_rejects_until_cap_then_raises():
    with pytest.raises(RuntimeError, match="tries"):
        gen_trapgrid(np.random.default_rng(0), probe_episodes=50,
                     max_tries=3, band=(1.01, 1.02))  # unsatisfiable band


# ----------------------------------------------------------- gen_home_pair

def test_home_constants_match_original_envs():
    a = GridHome(HOME_A, rng=np.random.default_rng(0))
    b = GridHome(HOME_B, rng=np.random.default_rng(0))
    assert HOME_SHAPE == a.shape == b.shape == (11, 13)
    assert HOME_START == tuple(map(int, a.start)) == tuple(map(int, b.start))
    assert HOME_GOAL == tuple(map(int, a.goal)) == tuple(map(int, b.goal))
    # the registered sp band brackets both originals (17 and 11)
    assert HOME_SP_BAND[0] <= a.shortest_path_len() <= HOME_SP_BAND[1]
    assert HOME_SP_BAND[0] <= b.shortest_path_len() <= HOME_SP_BAND[1]


def test_gen_home_pair_shell_and_anchor_coordinates():
    ma, mb = gen_home_pair(np.random.default_rng(0))
    for ms in (ma, mb):
        env = GridHome(ms, rng=np.random.default_rng(0))
        assert env.shape == (11, 13)
        assert tuple(map(int, env.start)) == HOME_START
        assert tuple(map(int, env.goal)) == HOME_GOAL
        border_ok, _ = _border_and_interior_walls(env.grid)
        assert border_ok


def test_gen_home_pair_solvable_within_band_and_connected():
    for seed in range(3):
        ma, mb = gen_home_pair(np.random.default_rng(seed))
        for ms in (ma, mb):
            sp = GridHome(ms, rng=np.random.default_rng(0)).shortest_path_len()
            assert HOME_SP_BAND[0] <= sp <= HOME_SP_BAND[1]
            assert floor_connected(ms)


def test_gen_home_pair_differs_in_enough_wall_cells():
    assert MIN_WALL_DIFF == 10
    for seed in range(3):
        ma, mb = gen_home_pair(np.random.default_rng(seed))
        assert len(wall_cells(ma) ^ wall_cells(mb)) >= MIN_WALL_DIFF


def test_gen_home_pair_deterministic_in_rng_and_varies_across_seeds():
    p1 = gen_home_pair(np.random.default_rng(9))
    p2 = gen_home_pair(np.random.default_rng(9))
    p3 = gen_home_pair(np.random.default_rng(10))
    assert p1 == p2
    assert p1 != p3


def test_gen_home_pair_raises_on_unsatisfiable_band():
    with pytest.raises(RuntimeError, match="tries"):
        gen_home_pair(np.random.default_rng(0), max_tries=3, sp_band=(1, 2))


# ----------------------------------------------------------------- helpers

def test_wall_cells_reads_hash_cells():
    assert wall_cells("###\n#S#\n###") == {(0, 0), (0, 1), (0, 2), (1, 0),
                                           (1, 2), (2, 0), (2, 1), (2, 2)}


def test_floor_connected_detects_disconnection():
    connected = "#####\n#S.G#\n#####"
    split = "#####\n#S#G#\n#####"
    assert floor_connected(connected)
    assert not floor_connected(split)
