"""Linear Q-learning: Q(s, a) = w_a . phi(s), SGD on the TD(0) error.

Unlike a table, actions share value across every state that activates the
same features, so learning one context reshapes others — the substrate of
contextual interference (H3). The update is a plain stochastic gradient step
on the squared TD error with the bootstrap target held fixed
(semi-gradient): w_a += lr * td * phi.
"""

import numpy as np


class LinearQ:
    def __init__(self, n_features, n_actions, lr=0.1, gamma=0.99, eps=0.1,
                 rng=None):
        self.n_features, self.n_actions = n_features, n_actions
        self.lr, self.gamma, self.eps = lr, gamma, eps
        self.W = np.zeros((n_actions, n_features))
        self.rng = rng if rng is not None else np.random.default_rng()

    def q(self, phi):
        """Q-values for all actions at features phi."""
        return self.W @ phi

    def greedy(self, phi):
        q = self.q(phi)
        best = np.flatnonzero(q == q.max())
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    def act(self, phi):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(phi)

    def update(self, phi, a, r, phi2, done):
        """TD(0) semi-gradient step; phi2 is ignored (may be None) when done."""
        target = r if done else r + self.gamma * float(self.q(phi2).max())
        td = target - float(self.W[a] @ phi)
        self.W[a] += self.lr * td * phi
