import numpy as np
import pytest

from devrl.agents.dyna import DynaQ
from devrl.agents.touchnav import TouchDynaQ, TouchNavigator


def make_agent(**kw):
    defaults = dict(n_states=4, n_actions=2, planning_steps=0,
                    rng=np.random.default_rng(0))
    defaults.update(kw)
    return TouchDynaQ(**defaults)


def test_is_a_dynaq_and_records_bump_counts():
    ag = make_agent()
    assert isinstance(ag, DynaQ)
    for _ in range(3):
        ag.observe_touch(1, 0, True)
    ag.observe_touch(1, 0, False)
    # Laplace-smoothed: (3 bumps + 1) / (4 visits + 2)
    assert ag.bump_prob(1, 0) == pytest.approx(4 / 6)


def test_bump_prob_unseen_pair_is_uninformative():
    ag = make_agent()
    assert ag.bump_prob(0, 1) == pytest.approx(0.5)


def test_bump_likelihood_is_complementary():
    ag = make_agent()
    ag.observe_touch(2, 1, True)
    ag.observe_touch(2, 1, True)
    p = ag.bump_prob(2, 1)
    assert ag.bump_likelihood(2, 1, True) == pytest.approx(p)
    assert ag.bump_likelihood(2, 1, False) == pytest.approx(1 - p)
    assert (ag.bump_likelihood(2, 1, True)
            + ag.bump_likelihood(2, 1, False)) == pytest.approx(1.0)


def _train_on_chain(agent_cls):
    """Identical scripted experience for DynaQ vs TouchDynaQ."""
    ag = agent_cls(n_states=6, n_actions=2, planning_steps=5, lr=0.2, eps=0.3,
                   rng=np.random.default_rng(1))
    s = 0
    for _ in range(400):
        a = ag.act(s)
        s2 = min(5, s + 1) if a == 1 else max(0, s - 1)
        done = s2 == 5
        if isinstance(ag, TouchDynaQ):
            ag.observe_touch(s, a, bumped=(s2 == s))
        ag.update(s, a, float(done), s2, done)
        s = 0 if done else s2
    return ag.Q


def test_touch_recording_does_not_change_learning():
    # bump bookkeeping must not consume rng or alter Q/model updates,
    # so the DynaQ sample-efficiency curve is legitimate DynaQ
    assert np.array_equal(_train_on_chain(DynaQ), _train_on_chain(TouchDynaQ))


def test_posterior_matches_bayes_formula_by_hand():
    # posterior[s2] ∝ sum_s belief[s] * P(s2|s,a) * P(bump=b|s,a)
    ag = make_agent(n_states=3, n_actions=1)
    for _ in range(3):
        ag.update(0, 0, 0.0, 1, False)  # P(1|0,0)=0.75
    ag.update(0, 0, 0.0, 2, False)      # P(2|0,0)=0.25
    ag.update(1, 0, 0.0, 2, False)      # P(2|1,0)=1.0
    ag.update(1, 0, 0.0, 2, False)
    for _ in range(4):
        ag.observe_touch(0, 0, True)
    for _ in range(4):
        ag.observe_touch(0, 0, False)   # P(bump|0,0) = 5/10
    for _ in range(6):
        ag.observe_touch(1, 0, True)    # P(bump|1,0) = 7/8
    nav = TouchNavigator(model=ag, q=ag.Q, start_state=0, touch=ag)
    nav.belief = np.array([0.5, 0.5, 0.0])
    nav.advance(0, True)
    un = 0.5 * 0.5 * np.array([0.0, 0.75, 0.25]) \
        + 0.5 * (7 / 8) * np.array([0.0, 0.0, 1.0])
    assert np.allclose(nav.belief, un / un.sum())


def test_bump_observation_disambiguates_wall_adjacent_state():
    # two states, both self-looping; s0 bumps, s1 does not: feeling the wall
    # should concentrate an ambiguous belief onto the wall-adjacent state
    ag = make_agent(n_states=2, n_actions=1)
    ag.update(0, 0, 0.0, 0, False)
    ag.update(1, 0, 0.0, 1, False)
    for _ in range(18):
        ag.observe_touch(0, 0, True)    # P(bump|0,0) = 19/20
    for _ in range(18):
        ag.observe_touch(1, 0, False)   # P(bump|1,0) = 1/20
    nav = TouchNavigator(ag, ag.Q, 0, touch=ag)
    nav.belief = np.array([0.5, 0.5])
    nav.advance(0, True)
    assert int(np.argmax(nav.belief)) == 0
    assert nav.belief[0] > 0.9


def test_touchnavigator_act_is_expected_q_greedy():
    ag = make_agent(n_states=2, n_actions=2)
    ag.Q = np.array([[0.0, 1.0], [1.0, 0.0]])
    nav = TouchNavigator(ag, ag.Q, 0, touch=ag)
    nav.belief = np.array([0.9, 0.1])
    assert nav.act() == 1  # E[Q(.,1)] = 0.9 > E[Q(.,0)] = 0.1
