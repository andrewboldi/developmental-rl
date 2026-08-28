import numpy as np
import pytest

from devrl.envs.soccer import CARRIED, SCORED, SoccerGrid


def make_env(seed=0):
    return SoccerGrid(rng=np.random.default_rng(seed))


def test_pitch_dimensions_and_state_space():
    env = make_env()
    assert (env.W, env.H) == (11, 7)
    assert env.n_actions == 5
    assert env.n_cells == 77
    # ball status: at any of 77 cells, carried, or scored
    assert env.n_states == 77 * 79


def test_action_ids_extend_gridhome_convention():
    assert (SoccerGrid.UP, SoccerGrid.RIGHT, SoccerGrid.DOWN, SoccerGrid.LEFT,
            SoccerGrid.SHOOT) == (0, 1, 2, 3, 4)


def test_goal_geometry():
    env = make_env()
    assert set(env.GOAL_CELLS) == {(2, 10), (3, 10), (4, 10)}  # middle of right edge
    assert env.GOAL_CENTER == (3, 10)


def test_state_encoding_roundtrip():
    env = make_env()
    seen = set()
    for agent in [(0, 0), (3, 5), (6, 10)]:
        for ball in [(0, 0), (5, 7), CARRIED, SCORED]:
            s = env.state_of(agent, ball)
            assert 0 <= s < env.n_states
            assert env.decode(s) == (agent, ball)
            seen.add(s)
    assert len(seen) == 12  # all combinations distinct


def test_kickoff_reset():
    env = make_env()
    s = env.reset()
    assert env.decode(s) == ((3, 0), (3, 5))  # agent at own edge, ball at center


def test_reset_to_custom_start_states():
    env = make_env()
    assert env.decode(env.reset(agent=(2, 9), ball=CARRIED)) == ((2, 9), CARRIED)
    assert env.decode(env.reset(agent=(5, 1), ball=(3, 5))) == ((5, 1), (3, 5))


def test_reset_on_ball_cell_picks_it_up():
    env = make_env()
    s = env.reset(agent=(3, 5), ball=(3, 5))
    assert env.decode(s) == ((3, 5), CARRIED)


def test_moves_are_deterministic():
    env = make_env()
    env.reset(agent=(3, 5), ball=(0, 0))
    for a, rc in [(SoccerGrid.UP, (2, 5)), (SoccerGrid.RIGHT, (2, 6)),
                  (SoccerGrid.DOWN, (3, 6)), (SoccerGrid.LEFT, (3, 5))]:
        s, r, done, _ = env.step(a)
        assert env.decode(s)[0] == rc and r == 0.0 and not done


def test_edges_block_movement():
    env = make_env()
    for start, a in [((0, 5), SoccerGrid.UP), ((6, 5), SoccerGrid.DOWN),
                     ((3, 0), SoccerGrid.LEFT), ((3, 10), SoccerGrid.RIGHT)]:
        env.reset(agent=start, ball=(0, 0))
        s, r, done, _ = env.step(a)
        assert env.decode(s)[0] == start and not done


def test_pickup_by_entering_ball_cell():
    env = make_env()
    env.reset()  # agent (3,0), ball (3,5)
    for _ in range(4):
        s, *_ = env.step(SoccerGrid.RIGHT)
        assert env.decode(s)[1] == (3, 5)  # ball waits until reached
    s, r, done, _ = env.step(SoccerGrid.RIGHT)  # step onto (3,5)
    assert env.decode(s) == ((3, 5), CARRIED)
    assert r == 0.0 and not done
    s, *_ = env.step(SoccerGrid.RIGHT)  # carried ball travels with the agent
    assert env.decode(s) == ((3, 6), CARRIED)


def test_walking_into_goal_never_scores():
    env = make_env()
    env.reset(agent=(3, 9), ball=CARRIED)
    s, r, done, info = env.step(SoccerGrid.RIGHT)
    assert env.decode(s) == ((3, 10), CARRIED)
    assert r == 0.0 and not done and not info["scored"]


def test_shoot_without_ball_is_noop():
    env = make_env()
    s0 = env.reset()  # kickoff: not carrying
    s, r, done, info = env.step(SoccerGrid.SHOOT)
    assert s == s0 and r == 0.0 and not done and not info["shot"]


def test_shot_p_is_chebyshev_ramp():
    env = make_env()
    assert env.shot_p((3, 10)) == 1.0                 # d=0: point blank
    assert env.shot_p((3, 9)) == pytest.approx(0.8)   # d=1
    assert env.shot_p((0, 10)) == pytest.approx(0.4)  # d=3 (row offset counts)
    assert env.shot_p((3, 5)) == 0.0                  # d=5: out of range
    assert env.shot_p((0, 0)) == 0.0                  # d=10


def test_shot_from_goal_center_always_scores():
    env = make_env()
    for _ in range(50):
        env.reset(agent=(3, 10), ball=CARRIED)
        s, r, done, info = env.step(SoccerGrid.SHOOT)
        assert r == 1.0 and done and info["scored"]
        assert env.decode(s)[1] == SCORED


def test_shot_out_of_range_always_misses_and_ends_episode():
    env = make_env()
    for _ in range(50):
        env.reset(agent=(3, 5), ball=CARRIED)  # d=5 -> p=0
        _, r, done, info = env.step(SoccerGrid.SHOOT)
        assert r == 0.0 and done and not info["scored"]


def test_any_shot_while_carrying_ends_the_episode():
    env = make_env(seed=1)
    misses = 0
    for _ in range(200):
        env.reset(agent=(3, 7), ball=CARRIED)  # d=3, p=0.4
        _, r, done, info = env.step(SoccerGrid.SHOOT)
        assert done  # goal or miss, the ball is gone
        if not info["scored"]:
            misses += 1
            assert r == 0.0
    assert misses > 0


def test_shot_success_probability_matches_formula():
    # Statistical pin of the one stochastic rule: empirical success rate
    # within ~4 sigma of p = 1 - d/5 at several distances.
    env = make_env(seed=2)
    n = 4000
    for cell, p in [((3, 9), 0.8), ((3, 8), 0.6), ((3, 6), 0.2)]:
        wins = 0
        for _ in range(n):
            env.reset(agent=cell, ball=CARRIED)
            _, r, _, _ = env.step(SoccerGrid.SHOOT)
            wins += int(r)
        se = (p * (1 - p) / n) ** 0.5
        assert abs(wins / n - p) < 4 * se
