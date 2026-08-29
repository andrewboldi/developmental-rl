"""ReplayQ: the update-matched Q-learning + experience-replay baseline
for EXP1 (van Hasselt, Hessel & Aslanides 2019). One real Q update per
env step plus `replay_steps` replayed updates drawn uniformly from the
agent's own transition buffer — DynaQ's update budget with remembered
experience in place of a learned model."""

import numpy as np

from devrl.agents.dyna import DynaQ
from devrl.agents.qlearning import QLearner
from devrl.agents.replayq import ReplayQ


def test_replayq_is_a_qlearner_without_a_model():
    ag = ReplayQ(n_states=4, n_actions=2, replay_steps=20,
                 rng=np.random.default_rng(0))
    assert isinstance(ag, QLearner)
    assert not isinstance(ag, DynaQ)
    assert not hasattr(ag, "trans")  # no learned model — remembered data only


def test_age_counts_real_env_steps_only_and_buffer_grows():
    # budget accounting must mirror DynaQ: age == env steps, replayed
    # updates are extra compute, not extra data
    ag = ReplayQ(n_states=4, n_actions=2, replay_steps=20,
                 rng=np.random.default_rng(0))
    for _ in range(7):
        ag.update(0, 1, 0.0, 1, False)
    assert ag.age == 7
    assert len(ag.buffer) == 7


def test_zero_replay_steps_equals_plain_qlearner():
    kw = dict(n_states=5, n_actions=3, lr=0.3, gamma=0.9, optimistic_init=1.0)
    a = ReplayQ(replay_steps=0, rng=np.random.default_rng(1), **kw)
    b = QLearner(rng=np.random.default_rng(1), **kw)
    for s, act, r, s2, d in [(0, 1, 0.0, 1, False), (1, 2, 1.0, 4, True),
                             (0, 0, 0.0, 2, False)]:
        a.update(s, act, r, s2, d)
        b.update(s, act, r, s2, d)
    assert np.array_equal(a.Q, b.Q)


def test_replay_does_exactly_k_extra_updates_on_singleton_buffer():
    # one terminal transition, lr=0.5: the real update then k replays of
    # the same transition give Q = 1 - 0.5**(k+1) exactly
    k = 20
    ag = ReplayQ(n_states=3, n_actions=1, lr=0.5, gamma=0.9, replay_steps=k,
                 rng=np.random.default_rng(0))
    ag.update(0, 0, 1.0, 2, True)
    assert np.isclose(ag.Q[0, 0], 1.0 - 0.5 ** (k + 1))


def test_replay_propagates_value_backward_within_one_pass():
    # feed a single left-to-right corridor walk: plain Q-learning moves
    # value only one cell back from the reward, replay chains it toward
    # the start within the same data
    walk = [(s, 0, 1.0 if s == 5 else 0.0, s + 1, s == 5) for s in range(6)]
    plain = QLearner(n_states=7, n_actions=1, lr=0.5, gamma=0.9,
                     rng=np.random.default_rng(2))
    rep = ReplayQ(n_states=7, n_actions=1, lr=0.5, gamma=0.9,
                  replay_steps=200, rng=np.random.default_rng(2))
    for tr in walk:
        plain.update(*tr)
        rep.update(*tr)
    assert plain.Q[0, 0] == 0.0  # reward never reaches the start
    assert rep.Q[0, 0] > 0.01  # replay chains it back
    assert rep.Q[4, 0] > plain.Q[4, 0]
