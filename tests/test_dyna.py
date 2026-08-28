import numpy as np

from devrl.agents.dyna import BlindNavigator, DynaQ


class Chain:
    """Deterministic 5-state chain: action 1 moves right, action 0 moves left.

    Reward 1.0 upon entering state 4 (terminal).
    """

    n_states, n_actions = 5, 2

    def __init__(self):
        self.s = 0

    def reset(self):
        self.s = 0
        return self.s

    def step(self, a):
        self.s = min(4, self.s + 1) if a == 1 else max(0, self.s - 1)
        done = self.s == 4
        return self.s, float(done), done, {}


def run_episodes(agent, env, n_ep, cap=50):
    total_steps = 0
    for _ in range(n_ep):
        s = env.reset()
        for _ in range(cap):
            a = agent.act(s)
            s2, r, done, _ = env.step(a)
            agent.update(s, a, r, s2, done)
            total_steps += 1
            s = s2
            if done:
                break
    return total_steps


def test_model_learns_transition_probabilities():
    ag = DynaQ(n_states=5, n_actions=2, planning_steps=0, eps=1.0,
               rng=np.random.default_rng(0))
    run_episodes(ag, Chain(), n_ep=30)
    probs = ag.model_next_probs(1, 1)  # from state 1, right -> state 2 always
    assert probs[2] > 0.99


def test_planning_accelerates_value_propagation():
    # After identical single experience of the goal transition, the planner
    # with many replay steps should have propagated value further back.
    def experience(ag):
        rng = np.random.default_rng(1)
        env = Chain()
        for _ in range(8):  # a handful of random-walk episodes
            s = env.reset()
            for _ in range(30):
                a = int(rng.random() < 0.6)
                s2, r, done, _ = env.step(a)
                ag.update(s, a, r, s2, done)
                s = s2
                if done:
                    break

    plain = DynaQ(n_states=5, n_actions=2, planning_steps=0, lr=0.2,
                  rng=np.random.default_rng(2))
    planner = DynaQ(n_states=5, n_actions=2, planning_steps=30, lr=0.2,
                    rng=np.random.default_rng(2))
    experience(plain)
    experience(planner)
    # value at the start state, best action, should be larger for the planner
    assert planner.Q[0].max() > plain.Q[0].max() + 0.05


def test_blind_navigator_tracks_belief_in_deterministic_world():
    ag = DynaQ(n_states=5, n_actions=2, planning_steps=10, eps=1.0,
               rng=np.random.default_rng(0))
    run_episodes(ag, Chain(), n_ep=50)
    nav = BlindNavigator(model=ag, q=ag.Q, start_state=0)
    a = nav.act()  # greedy under belief: should move right
    assert a == 1
    nav.advance(a)  # dead-reckon: belief should now be on state 1
    assert np.argmax(nav.belief) == 1


def test_blind_navigator_reaches_goal_open_loop():
    ag = DynaQ(n_states=5, n_actions=2, planning_steps=20, eps=0.3,
               rng=np.random.default_rng(0))
    run_episodes(ag, Chain(), n_ep=200)
    env = Chain()
    s = env.reset()
    nav = BlindNavigator(model=ag, q=ag.Q, start_state=s)
    for _ in range(10):
        a = nav.act()
        s, r, done, _ = env.step(a)  # note: s is NOT given to nav
        nav.advance(a)
        if done:
            break
    assert done  # blind agent walked its own home
