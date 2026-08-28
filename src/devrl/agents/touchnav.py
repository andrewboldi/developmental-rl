"""Touch: bump statistics learned during training, and a blind navigator
that filters its dead-reckoned belief with them.

`TouchDynaQ` is a DynaQ that additionally counts bump signals per (s, a) —
the tactile memory of where the walls are. Recording is a side channel
(`observe_touch`): it consumes no randomness and never alters Q, the model,
or the update sequence, so the agent's learning is exactly DynaQ's.

`TouchNavigator` extends dead reckoning with a Bayes filter: after taking
action a and feeling bump b, the belief over the next state is

    posterior[s2] ∝ sum_s belief[s] * P(s2 | s, a) * P(bump=b | s, a)

Bump probabilities are Laplace-smoothed, so a never-visited (s, a) is
uninformative (P = 0.5) rather than impossible.
"""

import numpy as np

from devrl.agents.dyna import BlindNavigator, DynaQ


class TouchDynaQ(DynaQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bump_visits = np.zeros((self.n_states, self.n_actions))
        self.bump_hits = np.zeros((self.n_states, self.n_actions))

    def observe_touch(self, s, a, bumped):
        self.bump_visits[s, a] += 1
        self.bump_hits[s, a] += bool(bumped)

    def bump_prob(self, s, a):
        """Laplace-smoothed P(bump | s, a)."""
        return float((self.bump_hits[s, a] + 1.0)
                     / (self.bump_visits[s, a] + 2.0))

    def bump_likelihood(self, s, a, bumped):
        p = self.bump_prob(s, a)
        return p if bumped else 1.0 - p


class TouchNavigator(BlindNavigator):
    """Blind navigation with touch: dead reckoning plus bump filtering."""

    def __init__(self, model, q, start_state, touch):
        super().__init__(model, q, start_state)
        self.touch = touch

    def advance(self, a, bumped):
        new_belief = np.zeros_like(self.belief)
        for s in np.flatnonzero(self.belief > 1e-12):
            w = self.belief[s] * self.touch.bump_likelihood(s, a, bumped)
            new_belief += w * self.model.model_next_probs(s, a)
        self.belief = new_belief / max(new_belief.sum(), 1e-12)
