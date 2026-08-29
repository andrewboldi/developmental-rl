"""Q-learning + uniform experience replay: the update-matched baseline.

Van Hasselt, Hessel & Aslanides (NeurIPS 2019) point out that experience
replay is a non-parametric model — in a near-deterministic tabular world,
Dyna's learned model IS a replay buffer — so Dyna's sample-efficiency
edge over 1-update-per-step Q-learning may be a pure update-count
artifact. `ReplayQ` grants Q-learning the same update budget as
DynaQ(planning=k): one real update per env step plus `replay_steps`
replayed updates drawn uniformly (with replacement) from its own
transition buffer. Whatever gap remains between DynaQ and ReplayQ is
attributable to the parametric model, not to doing more updates.

Deliberately mirrors DynaQ's accounting: replayed updates are applied
inline (they do not advance `age`), so `age` counts real env steps only.
"""

from devrl.agents.qlearning import QLearner


class ReplayQ(QLearner):
    """QLearner plus a transition buffer replayed uniformly each step."""

    def __init__(self, *args, replay_steps=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_steps = replay_steps
        self.buffer = []  # (s, a, r, s2, done) real transitions, in order

    def update(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, bool(done)))
        super().update(s, a, r, s2, done)
        self._replay()

    def _replay(self):
        for _ in range(self.replay_steps):
            s, a, r, s2, done = self.buffer[
                self.rng.integers(len(self.buffer))]
            target = r if done else r + self.gamma * self.Q[s2].max()
            self.Q[s, a] += self.current_lr() * (target - self.Q[s, a])
