import numpy as np
import pytest

from devrl.envs.balance import BalanceBot


def make(s=1.0, mode="balance", seed=0, noise=0.0):
    return BalanceBot(s=s, mode=mode, rng=np.random.default_rng(seed),
                      noise_std_frac=noise)


def pin(env, theta=0.0, theta_dot=0.0):
    env.theta, env.theta_dot = theta, theta_dot


# ---------------------------------------------------------------- morphology

def test_square_cube_scaling_of_body_parameters():
    env = make()
    for s in (0.5, 0.75, 1.0):
        env.set_size(s)
        assert env.s == s
        assert env.l == pytest.approx(s)
        assert env.m == pytest.approx(15 * s ** 3)
        assert env.tau_max == pytest.approx(40 * s ** 2)


def test_square_cube_control_authority_ordering():
    # max angular accel from torque: tau_max / (m l^2) = 40 / (15 s^3):
    # strictly DEcreasing in s — small bodies are relatively stronger.
    env = make()
    auth = []
    for s in (0.5, 0.75, 1.0):
        env.set_size(s)
        auth.append(env.tau_max / (env.m * env.l ** 2))
    assert auth[0] > auth[1] > auth[2]
    assert auth[0] == pytest.approx(40 / (15 * 0.5 ** 3))
    assert auth[2] == pytest.approx(40 / 15)


def test_set_size_changes_dynamics_live_mid_episode():
    env = make(s=0.5)
    env.reset()
    env.step(2)
    env.set_size(1.0)
    assert env.tau_max == pytest.approx(40.0) and env.m == pytest.approx(15.0)
    # a fall NOW is charged at the current size
    pin(env, theta=0.7)
    _, r, done, info = env.step(2)
    assert done and info["fall"] and info["damage"] == pytest.approx(1.0)


# ------------------------------------------------------------- physics sanity

def test_uncontrolled_pendulum_falls_from_small_tilt():
    env = make(s=1.0, noise=0.0)
    env.reset()
    pin(env, theta=0.05, theta_dot=0.0)
    thetas = []
    for t in range(400):
        _, r, done, info = env.step(2)  # zero torque
        thetas.append(env.theta)
        if done:
            break
    assert done and info["fall"], "gravity must topple an uncontrolled pendulum"
    assert t < 399  # well before the cap
    assert all(np.diff(thetas) >= 0)  # tilt grows monotonically until the fall


def test_strong_counter_torque_recovers_small_tilt_at_adult_size():
    env = make(s=1.0, noise=0.0)
    env.reset()
    pin(env, theta=0.1, theta_dot=0.0)
    for _ in range(10):
        _, _, done, info = env.step(0)  # -tau_max
        assert not done
    assert env.theta < 0.1  # torque beat gravity: tilt is shrinking


def test_square_cube_behavioral_ordering_same_controller_same_tilt():
    # bang-bang controller from tilt 0.45 (< fall angle pi/5 ~ 0.628):
    # adult max torque accel 40/15=2.67 < gravity 9.8*sin(0.45)=4.26 -> doomed;
    # child (s=0.5) 21.3 > 19.6*sin(0.45)=8.5 -> recoverable.
    def rollout(s):
        env = make(s=s, noise=0.0)
        env.reset()
        pin(env, theta=0.45, theta_dot=0.0)
        for t in range(300):
            a = 0 if env.theta + 0.3 * env.theta_dot > 0 else 4
            _, _, done, info = env.step(a)
            if done:
                return t + 1, info["fall"]
        return 300, False

    steps_child, fell_child = rollout(0.5)
    steps_adult, fell_adult = rollout(1.0)
    assert fell_adult and steps_adult < 50
    assert not fell_child and steps_child == 300


# --------------------------------------------------------- rewards and damage

def test_fall_reward_fixed_but_damage_scales_as_s_fourth():
    for s in (0.5, 1.0):
        env = make(s=s, noise=0.0)
        env.reset()
        pin(env, theta=0.7)  # beyond pi/5: next integration step keeps it out
        s2, r, done, info = env.step(2)
        assert done and info["fall"]
        assert r == -5.0                                   # reward is FIXED
        assert info["damage"] == pytest.approx(s ** 4)     # damage is not
        assert 0 <= s2 < env.n_states


def test_no_damage_on_ordinary_step():
    env = make(noise=0.0)
    env.reset()
    pin(env)
    _, _, done, info = env.step(2)
    assert not done and not info["fall"] and info["damage"] == 0.0


def test_reward_one_upright_plus_two_on_target():
    env = make(mode="balance", noise=0.0)
    env.reset()
    pin(env, theta=0.0, theta_dot=0.0)  # stays exactly at 0, target is 0
    _, r, _, _ = env.step(2)
    assert r == 3.0
    pin(env, theta=0.2, theta_dot=0.0)  # upright but off target
    _, r, _, _ = env.step(2)
    assert r == 1.0


def test_on_target_bonus_tracks_walking_target():
    env = make(mode="walk", noise=0.0)
    env.reset()
    env.target_idx = 0  # target -0.15
    pin(env, theta=-0.15, theta_dot=0.0)
    _, r, _, info = env.step(2)
    assert info["target"] == pytest.approx(-0.15)
    assert r == 3.0
    pin(env, theta=0.0, theta_dot=0.0)  # |0 - (-0.15)| > 0.05
    _, r, _, _ = env.step(2)
    assert r == 1.0


def test_balance_mode_target_fixed_at_zero_and_cap_400():
    env = make(mode="balance", noise=0.0, seed=5)
    env.reset()
    total, n = 0.0, 0
    done = False
    while not done:
        pin(env, 0.0, 0.0)
        _, r, done, info = env.step(2)
        total += r
        n += 1
        assert info["target"] == 0.0
    assert n == 400 and not info["fall"]      # cap, not a fall
    assert total == pytest.approx(1200.0)     # 400 * (+1 upright +2 on-target)


# ------------------------------------------------------------ walking targets

def test_walking_targets_switch_exactly_every_40_steps():
    env = make(mode="walk", noise=0.0, seed=3)
    env.reset()
    seq = []
    for _ in range(400):
        pin(env, 0.0, 0.0)
        env.step(2)
        seq.append(env.target)
    changes = [i for i in range(1, 400) if seq[i] != seq[i - 1]]
    assert changes == list(range(39, 399, 40))  # after steps 40, 80, ..., 360
    assert set(seq) <= {-0.15, 0.0, 0.15}
    assert len(set(seq)) >= 2


def test_target_switch_is_random_uniform_over_other_two():
    # statistical: a switch never repeats the current target, and picks each
    # of the other two about half the time.
    env = make(mode="walk", noise=0.0, seed=11)
    env.reset()
    pairs = []
    for _ in range(24000):
        pin(env, 0.0, 0.0)
        before = env.target
        _, _, done, _ = env.step(2)
        if not done and env.target != before:
            pairs.append((before, env.target))
        if done:
            env.reset()
    assert len(pairs) == 60 * 9  # 9 switches per 400-step episode, 60 episodes
    assert all(a != b for a, b in pairs)
    nxt = np.array([b for a, b in pairs if a == 0.0])
    assert len(nxt) > 100
    frac = np.mean(nxt == -0.15)
    assert 0.35 < frac < 0.65


# ------------------------------------------------------------- stochasticity

def test_torque_noise_statistics_match_5pct_of_tau_max():
    # at theta=0 with zero commanded torque the only accel is the noise:
    # theta_dot after one step ~ N(0, dt * 0.05*tau_max / (m l^2)).
    env = make(s=1.0, mode="balance", seed=7, noise=0.05)
    expect_std = 0.02 * (0.05 * 40.0) / 15.0
    samples = []
    for _ in range(4000):
        env.reset()
        pin(env, 0.0, 0.0)
        env.step(2)
        samples.append(env.theta_dot)
    samples = np.array(samples)
    assert abs(samples.mean()) < 3 * expect_std / np.sqrt(len(samples))
    assert 0.9 * expect_std < samples.std() < 1.1 * expect_std


def test_zero_noise_is_exactly_deterministic():
    env = make(noise=0.0)
    env.reset()
    pin(env, 0.0, 0.0)
    env.step(2)
    assert env.theta == 0.0 and env.theta_dot == 0.0


def test_reset_tilt_distribution_small_and_bounded():
    env = make(seed=13)
    thetas, thdots = [], []
    for _ in range(1000):
        s = env.reset()
        assert 0 <= s < env.n_states
        thetas.append(env.theta)
        thdots.append(env.theta_dot)
        assert env.t == 0 and env.target == 0.0
    thetas, thdots = np.array(thetas), np.array(thdots)
    assert np.all(np.abs(thetas) <= 0.05) and np.all(np.abs(thdots) <= 0.1)
    assert 0.02 < thetas.std() < 0.038   # U(-0.05, 0.05) has std 0.029
    assert abs(thetas.mean()) < 0.005


# ------------------------------------------------------------- discretization

def test_state_space_dimensions():
    env = make()
    assert env.n_states == 17 * 15 * 3 == 765
    assert env.n_actions == 5
    assert np.allclose(env.ACTIONS, [-1.0, -1 / 3, 0.0, 1 / 3, 1.0])


def test_discretize_roundtrip_at_bin_centers():
    env = make()
    w_th = (2 * np.pi / 5) / 17
    w_td = 6.0 / 15
    for ti in (0, 8, 16):
        for vi in (0, 7, 14):
            for ki in (0, 1, 2):
                theta = -np.pi / 5 + (ti + 0.5) * w_th
                thdot = -3.0 + (vi + 0.5) * w_td
                s = env.discretize(theta, thdot, ki)
                assert s == (ti * 15 + vi) * 3 + ki
                assert env.bins_of(s) == (ti, vi, ki)


def test_discretize_edges_and_clipping():
    env = make()
    assert env.bins_of(env.discretize(-np.pi / 5, 0.0, 1))[0] == 0
    assert env.bins_of(env.discretize(np.pi / 5, 0.0, 1))[0] == 16
    assert env.bins_of(env.discretize(0.0, -9.9, 1))[1] == 0    # clipped
    assert env.bins_of(env.discretize(0.0, 9.9, 1))[1] == 14    # clipped
    for theta in np.linspace(-1.0, 1.0, 41):        # even past the fall angle
        for td in (-5.0, 0.0, 5.0):
            for ki in range(3):
                assert 0 <= env.discretize(theta, td, ki) < 765


def test_state_reflects_current_target_index():
    env = make(mode="walk")
    env.reset()
    pin(env, 0.0, 0.0)
    states = []
    for idx in range(3):
        env.target_idx = idx
        states.append(env.state())
    assert len(set(states)) == 3
    assert [s % 3 for s in states] == [0, 1, 2]


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        BalanceBot(mode="fly")
