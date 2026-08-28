import numpy as np
import pytest

from devrl.envs.gridhome import HOME_A, HOME_B, GridHome

TINY = """
#####
#S.G#
#####
"""


def test_parses_map_and_state_indexing():
    env = GridHome(TINY, slip=0.0, rng=np.random.default_rng(0))
    assert env.n_actions == 4
    s = env.reset()
    assert env.pos_of(s) == (1, 1)


def test_deterministic_move_right_reaches_goal_with_reward():
    env = GridHome(TINY, slip=0.0, rng=np.random.default_rng(0))
    env.reset()
    s2, r, done, info = env.step(GridHome.RIGHT)
    assert env.pos_of(s2) == (1, 2) and r == 0.0 and not done
    s3, r, done, info = env.step(GridHome.RIGHT)
    assert env.pos_of(s3) == (1, 3) and r == 1.0 and done


def test_wall_blocks_and_reports_bump():
    env = GridHome(TINY, slip=0.0, rng=np.random.default_rng(0))
    s = env.reset()
    s2, r, done, info = env.step(GridHome.UP)
    assert s2 == s and info["bump"] is True


def test_slip_moves_perpendicular_sometimes():
    env = GridHome(HOME_A, slip=0.5, rng=np.random.default_rng(0))
    env.reset()
    outcomes = set()
    for _ in range(200):
        env.reset()
        s2, *_ = env.step(GridHome.RIGHT)
        outcomes.add(env.pos_of(s2))
    assert len(outcomes) > 1  # slip produced at least one non-right outcome


def test_homes_are_same_size_different_layout():
    a = GridHome(HOME_A, rng=np.random.default_rng(0))
    b = GridHome(HOME_B, rng=np.random.default_rng(0))
    assert a.shape == b.shape
    assert a.walls_rc() != b.walls_rc()
    # both are solvable
    for env in (a, b):
        assert env.shortest_path_len() is not None


def test_bad_map_raises():
    with pytest.raises(ValueError):
        GridHome("###\n#S#\n###")  # no goal
