import numpy as np

from devrl.agents.qlearning import QLearner


def make_agent(**kw):
    defaults = dict(n_states=2, n_actions=2, lr=0.5, gamma=0.9, eps=0.0,
                    rng=np.random.default_rng(0))
    defaults.update(kw)
    return QLearner(**defaults)


def test_update_moves_q_toward_target():
    ag = make_agent()
    ag.update(0, 1, 1.0, 1, done=True)  # terminal: target = r
    assert ag.Q[0, 1] == 0.5  # 0 + 0.5*(1 - 0)
    ag.update(0, 1, 1.0, 1, done=True)
    assert ag.Q[0, 1] == 0.75


def test_update_bootstraps_from_next_state_when_not_done():
    ag = make_agent()
    ag.Q[1, 0] = 1.0
    ag.update(0, 0, 0.0, 1, done=False)  # target = 0 + 0.9 * 1.0
    assert np.isclose(ag.Q[0, 0], 0.5 * 0.9)


def test_greedy_prefers_higher_q():
    ag = make_agent()
    ag.Q[0] = [0.2, 0.8]
    assert ag.greedy(0) == 1


def test_eps_one_explores_uniformly():
    ag = make_agent(eps=1.0, rng=np.random.default_rng(0))
    ag.Q[0] = [100.0, 0.0]
    acts = [ag.act(0) for _ in range(500)]
    assert 0.4 < np.mean(acts) < 0.6  # both actions taken about equally


def test_plasticity_decay_reduces_lr_and_eps_with_age():
    # lr/eps given as callables of age (number of updates so far)
    ag = make_agent(lr=lambda age: 0.5 / (1 + age), eps=lambda age: 1.0 / (1 + age))
    assert ag.current_lr() == 0.5
    ag.update(0, 0, 1.0, 1, done=True)
    assert ag.Q[0, 0] == 0.5
    assert ag.current_lr() == 0.25  # age is now 1
    assert ag.current_eps() == 0.5


def test_optimistic_init_fills_q():
    ag = make_agent(optimistic_init=2.0)
    assert np.all(ag.Q == 2.0)


def test_learns_two_armed_bandit():
    ag = make_agent(n_states=1, eps=0.2, lr=0.1, rng=np.random.default_rng(3))
    rng = np.random.default_rng(4)
    for _ in range(300):
        a = ag.act(0)
        r = float(rng.random() < (0.9 if a == 1 else 0.1))
        ag.update(0, a, r, 0, done=True)
    assert ag.greedy(0) == 1
