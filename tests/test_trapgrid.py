import numpy as np

from devrl.envs.trapgrid import TRAP_MAP, TrapGrid


def test_map_dimensions_and_landmarks():
    env = TrapGrid()
    assert env.shape == (11, 15)  # 15 wide x 11 tall per DESIGN
    assert env.n_states == 165 and env.n_actions == 4
    assert env.start == (5, 1)  # left-center
    assert env.goal == (5, 13)  # far right
    assert env.candies == {(2, 3), (5, 5), (8, 3)}  # near the start


def test_state_indexing_roundtrip():
    env = TrapGrid()
    s = env.reset()
    assert env.pos_of(s) == (5, 1)
    assert env.state_of((5, 1)) == s


def test_moves_are_deterministic():
    outcomes = set()
    for _ in range(20):
        env = TrapGrid()
        env.reset()
        s2, r, done, info = env.step(TrapGrid.RIGHT)
        outcomes.add((env.pos_of(s2), r, done))
    assert outcomes == {((5, 2), 0.0, False)}


def test_wall_blocks_and_reports_bump():
    env = TrapGrid()
    s = env.reset()
    s2, r, done, info = env.step(TrapGrid.LEFT)  # column 0 is wall
    assert s2 == s and info["bump"] is True and r == 0.0 and not done


def test_candy_is_terminal_with_small_reward():
    env = TrapGrid()
    env.reset()
    for _ in range(3):
        _, r, done, _ = env.step(TrapGrid.RIGHT)
        assert r == 0.0 and not done
    s2, r, done, _ = env.step(TrapGrid.RIGHT)  # onto candy at (5, 5)
    assert env.pos_of(s2) == (5, 5) and r == 0.3 and done


def test_big_goal_is_terminal_with_reward_10():
    env = TrapGrid()
    env.reset()
    env.pos = (5, 12)
    s2, r, done, _ = env.step(TrapGrid.RIGHT)
    assert env.pos_of(s2) == (5, 13) and r == 10.0 and done


def test_cap_truncates_without_terminating():
    env = TrapGrid(cap=120)
    env.reset()
    for _ in range(119):
        _, r, done, info = env.step(TrapGrid.LEFT)  # bump forever
        assert not done and not info["truncated"]
    _, r, done, info = env.step(TrapGrid.LEFT)
    assert not done and info["truncated"] and r == 0.0
    env.reset()
    _, _, _, info = env.step(TrapGrid.LEFT)
    assert not info["truncated"]  # reset restarts the episode clock


def test_reset_restores_start():
    env = TrapGrid()
    env.reset()
    env.step(TrapGrid.RIGHT)
    s = env.reset()
    assert env.pos_of(s) == (5, 1)


def test_shortest_safe_path_dodges_candy():
    # the straight line (12 steps) is blocked by the middle candy, which would
    # end the episode; the cheapest viable route detours around it (+2 steps)
    assert TrapGrid().shortest_path_len() == 14


def test_random_walks_end_on_candy_far_more_than_goal():
    """The trap statistics that make H4 possible: candy screens the start, so
    undirected exploration is absorbed a few steps from home and deep runs to
    the big goal are rare (but not impossible)."""
    rng = np.random.default_rng(0)
    env = TrapGrid()
    ends = {"candy": 0, "goal": 0, "timeout": 0}
    n = 2000
    for _ in range(n):
        env.reset()
        while True:
            _, r, done, info = env.step(int(rng.integers(4)))
            if done or info["truncated"]:
                which = "goal" if env.pos == env.goal else ("candy" if done else "timeout")
                ends[which] += 1
                break
    assert ends["candy"] + ends["goal"] + ends["timeout"] == n
    assert ends["candy"] / n > 0.5  # candy absorbs most random walks
    assert ends["goal"] / n < 0.05  # the mountain is rarely reached blindly
    assert ends["candy"] > 10 * ends["goal"]  # trap dominates by an order
