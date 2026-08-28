"""Dyna-Q: model-based tabular RL, plus blind navigation from the model.

The learned model doubles as the agent's "imagination": `BlindNavigator`
tracks a belief over states using only the actions taken (dead reckoning),
which is how an agent walks through its own home with its eyes closed.
"""

from collections import defaultdict

import numpy as np

from devrl.agents.qlearning import QLearner


class DynaQ(QLearner):
    def __init__(self, *args, planning_steps=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.planning_steps = planning_steps
        # model: (s, a) -> {s2: count}, running mean reward, terminal count
        self.trans = defaultdict(lambda: defaultdict(int))
        self.rew = {}
        self.term = defaultdict(lambda: defaultdict(int))
        self.seen = []  # list of (s, a) pairs, for uniform planning replay

    def update(self, s, a, r, s2, done):
        key = (s, a)
        if key not in self.rew:
            self.seen.append(key)
            self.rew[key] = r
        else:
            n = sum(self.trans[key].values())
            self.rew[key] += (r - self.rew[key]) / (n + 1)
        self.trans[key][s2] += 1
        self.term[key][s2] += int(done)
        super().update(s, a, r, s2, done)
        self._plan()

    def _plan(self):
        for _ in range(self.planning_steps):
            s, a = self.seen[self.rng.integers(len(self.seen))]
            nexts = self.trans[(s, a)]
            states = list(nexts)
            counts = np.array([nexts[x] for x in states], dtype=float)
            s2 = states[self.rng.choice(len(states), p=counts / counts.sum())]
            done = self.term[(s, a)][s2] / nexts[s2] > 0.5
            r = self.rew[(s, a)]
            target = r if done else r + self.gamma * self.Q[s2].max()
            self.Q[s, a] += self.current_lr() * (target - self.Q[s, a])

    def model_next_probs(self, s, a):
        """P(s' | s, a) under the learned model; assume 'stay' if never seen."""
        probs = np.zeros(self.n_states)
        nexts = self.trans.get((s, a))
        if not nexts:
            probs[s] = 1.0
            return probs
        total = sum(nexts.values())
        for s2, c in nexts.items():
            probs[s2] = c / total
        return probs


class BlindNavigator:
    """Act without observations: dead-reckon a belief using a learned model.

    belief starts one-hot at the (known) start state; each action advances it
    through the model's transition probabilities. Actions are chosen greedily
    by expected Q under the belief.
    """

    def __init__(self, model, q, start_state):
        self.model = model
        self.q = q
        self.belief = np.zeros(model.n_states)
        self.belief[start_state] = 1.0

    def act(self):
        expected_q = self.belief @ self.q
        return int(np.argmax(expected_q))

    def advance(self, a):
        new_belief = np.zeros_like(self.belief)
        for s in np.flatnonzero(self.belief > 1e-12):
            new_belief += self.belief[s] * self.model.model_next_probs(s, a)
        self.belief = new_belief / max(new_belief.sum(), 1e-12)
