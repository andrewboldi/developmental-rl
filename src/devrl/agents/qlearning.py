"""Tabular Q-learning with optional plasticity schedules.

`lr` and `eps` may be floats or callables of the agent's age (number of
updates performed), which models plasticity that declines over a lifetime.
"""

import numpy as np


class QLearner:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, eps=0.1,
                 optimistic_init=0.0, rng=None):
        self.n_states, self.n_actions = n_states, n_actions
        self.gamma = gamma
        self._lr, self._eps = lr, eps
        self.Q = np.full((n_states, n_actions), float(optimistic_init))
        self.rng = rng if rng is not None else np.random.default_rng()
        self.age = 0

    def current_lr(self):
        return self._lr(self.age) if callable(self._lr) else self._lr

    def current_eps(self):
        return self._eps(self.age) if callable(self._eps) else self._eps

    def greedy(self, s):
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    def act(self, s):
        if self.rng.random() < self.current_eps():
            return int(self.rng.integers(self.n_actions))
        return self.greedy(s)

    def update(self, s, a, r, s2, done):
        target = r if done else r + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.current_lr() * (target - self.Q[s, a])
        self.age += 1
