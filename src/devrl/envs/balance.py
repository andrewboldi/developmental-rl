"""BalanceBot: a torque-limited inverted pendulum with a growable body (H5).

The body has one scale parameter s (adult s=1): length l = s, mass m = 15 s^3,
motor limit tau_max = 40 s^2 — the square-cube law. Peak angular acceleration
from torque is tau_max / (m l^2) = 40 / (15 s^3), so a small body is
*relatively* stronger, but its dynamics are faster per fixed control step
(natural frequency sqrt(g/l) ~ s^-1/2) — the scaling is not rigged in either
direction. A fall (|theta| > pi/5) costs a fixed -5 reward, but the *physical
damage* scales as (s / s_adult)^4 (impact energy ~ m g l ~ s^4) and is
recorded separately from reward: the reward landscape is identical at every
size; only the wear on the body differs.

Damping (default, disclosed): joint damping torque is b * theta_dot with
b = B = 1.0 CONSTANT across sizes. Because the inertia is m l^2 = 15 s^5,
the *relative* damping b / (m l^2) = 1 / (15 s^5) is ~32x stronger for the
s=0.5 body than for the adult — an extra stabilization aid for small bodies
that is NOT part of the square-cube story. This was an undeclared v1 choice
(flagged in adversarial verification); it is now documented here and probed
by the `damp_exp` robustness variant below.

Physics-robustness variants (v2): `tau_exp` sets the torque law
tau_max = 40 s^tau_exp (2 = DESIGN default; 3 = muscle-torque ~ L^3, force
~ L^2 x lever arm ~ L, per RESEARCH.md), and `damp_exp` sets the damping law
b = B s^damp_exp (0 = constant default; 2 = size-scaled damping b = s^2,
which shrinks the small-body relative-damping advantage from s^-5 to s^-3).
All variants coincide exactly at s = 1.0, so adult-body evaluation and
adult-only conditions are variant-independent.

"Walking" = tracking a lean target in {-0.15, 0, +0.15} rad that switches
every 40 steps (weight transfer); "balance" fixes the target at 0. Reward is
+1 per upright step, +2 extra within 0.05 rad of the target. Episodes cap at
400 steps. Semi-implicit Euler at dt=0.02; Gaussian torque noise with
sigma = 5% of tau_max. `set_size` changes the dynamics live, mid-episode.

State = (theta in 17 bins over [-pi/5, pi/5]) x (theta_dot in 15 bins over
[-3, 3], clipped) x (target index) = 765 discrete states; 5 torque actions
{-1, -1/3, 0, 1/3, 1} * tau_max.
"""

import numpy as np

DT = 0.02
G = 9.8
B = 1.0  # base damping coefficient (b = B * s^damp_exp; default damp_exp=0)
FALL_ANGLE = np.pi / 5
CAP = 400
TARGETS = (-0.15, 0.0, 0.15)
SWITCH_EVERY = 40
N_THETA, N_THDOT = 17, 15
THDOT_MAX = 3.0


class BalanceBot:
    ACTIONS = np.array([-1.0, -1 / 3, 0.0, 1 / 3, 1.0])  # fractions of tau_max
    n_actions = 5
    n_states = N_THETA * N_THDOT * len(TARGETS)

    def __init__(self, s=1.0, mode="walk", rng=None, noise_std_frac=0.05,
                 tau_exp=2, damp_exp=0):
        if mode not in ("walk", "balance"):
            raise ValueError(f"mode must be 'walk' or 'balance', got {mode!r}")
        self.mode = mode
        self.noise_std_frac = noise_std_frac
        self.rng = rng if rng is not None else np.random.default_rng()
        self.tau_exp = int(tau_exp)
        self.damp_exp = int(damp_exp)
        self.set_size(s)
        self.theta = 0.0
        self.theta_dot = 0.0
        self.target_idx = 1  # target 0
        self.t = 0

    def set_size(self, s):
        """Morphology laws: l = s, m = 15 s^3, tau_max = 40 s^tau_exp,
        b = B s^damp_exp (defaults tau_exp=2, damp_exp=0 — see module doc)."""
        self.s = float(s)
        self.l = self.s
        self.m = 15.0 * self.s ** 3
        self.tau_max = 40.0 * self.s ** self.tau_exp
        self.b = B * self.s ** self.damp_exp

    def set_mode(self, mode):
        if mode not in ("walk", "balance"):
            raise ValueError(f"mode must be 'walk' or 'balance', got {mode!r}")
        self.mode = mode

    @property
    def target(self):
        return TARGETS[self.target_idx]

    # ------------------------------------------------------- discretization

    def discretize(self, theta, theta_dot, target_idx):
        ti = int((theta + FALL_ANGLE) / (2 * FALL_ANGLE) * N_THETA)
        ti = min(max(ti, 0), N_THETA - 1)
        vi = int((np.clip(theta_dot, -THDOT_MAX, THDOT_MAX) + THDOT_MAX)
                 / (2 * THDOT_MAX) * N_THDOT)
        vi = min(max(vi, 0), N_THDOT - 1)
        return (ti * N_THDOT + vi) * len(TARGETS) + target_idx

    def bins_of(self, state):
        rest, ki = divmod(int(state), len(TARGETS))
        ti, vi = divmod(rest, N_THDOT)
        return ti, vi, ki

    def state(self):
        return self.discretize(self.theta, self.theta_dot, self.target_idx)

    # --------------------------------------------------------------- dynamics

    def reset(self):
        self.theta = self.rng.uniform(-0.05, 0.05)
        self.theta_dot = self.rng.uniform(-0.1, 0.1)
        self.target_idx = 1
        self.t = 0
        return self.state()

    def step(self, a):
        tau = self.ACTIONS[a] * self.tau_max
        if self.noise_std_frac > 0:
            tau += self.rng.normal(0.0, self.noise_std_frac * self.tau_max)
        ml2 = self.m * self.l ** 2
        theta_dd = (G / self.l) * np.sin(self.theta) + tau / ml2 \
            - self.b * self.theta_dot / ml2
        self.theta_dot += DT * theta_dd          # semi-implicit Euler
        self.theta += DT * self.theta_dot
        self.t += 1

        info = {"theta": self.theta, "theta_dot": self.theta_dot,
                "target": self.target, "s": self.s}
        if abs(self.theta) > FALL_ANGLE:
            info.update(fall=True, damage=self.s ** 4)
            return self.state(), -5.0, True, info

        r = 1.0 + (2.0 if abs(self.theta - self.target) < 0.05 else 0.0)
        done = self.t >= CAP
        info.update(fall=False, damage=0.0)
        if self.mode == "walk" and not done and self.t % SWITCH_EVERY == 0:
            others = [i for i in range(len(TARGETS)) if i != self.target_idx]
            self.target_idx = int(self.rng.choice(others))
        return self.state(), r, done, info
