"""Distillation machinery for generational teaching (H4).

A dying teacher does not hand over its Q table — it hands over a story: the
(s, a) sequences of its few best episodes, deduplicated and capped to a
narrow advice bottleneck. The student is pretrained by setting an optimistic
value on each advised pair, so its greedy policy walks the remembered path
from birth and re-earns consolidated values with young plasticity. The
bottleneck transmits the rare peak experience ("the one time I reached the
mountain") without transmitting trap-shaped mediocrity or rigid habits.
"""


class EpisodicMemory:
    """Top-k episodes by return, with their (s, a) sequences.

    Ranking is by return first; `tie_break` decides among tied returns:

    - "earliest" (default, the registered primary rule): ties keep the
      EARLIER episode. A trapped teacher's tied-return episodes are its
      earliest ones — long exploratory meanders — so even mediocre advice
      transmits breadth rather than a distilled habit.
    - "shortest": ties keep the SHORTEST episode (equal lengths: earlier
      wins) — the rejected alternative the adversarial verification found
      the v1 headline sensitive to. EXP4 v2 runs BOTH rules as separate
      conditions so the choice is reported, not hidden.

    Both rules are deterministic and stable.
    """

    TIE_BREAKS = ("earliest", "shortest")

    def __init__(self, k=3, tie_break="earliest"):
        if tie_break not in self.TIE_BREAKS:
            raise ValueError("unknown tie_break %r (want one of %s)"
                             % (tie_break, ", ".join(self.TIE_BREAKS)))
        self.k = k
        self.tie_break = tie_break
        self._eps = []  # (-return, arrival index, [(s, a), ...])
        self._n = 0

    def _key(self, e):
        negr, arrival, sa = e
        if self.tie_break == "shortest":
            return (negr, len(sa), arrival)
        return (negr, arrival)

    def add(self, ep_return, sa_pairs):
        self._eps.append((-float(ep_return), self._n, list(sa_pairs)))
        self._n += 1
        self._eps.sort(key=self._key)
        del self._eps[self.k:]

    @property
    def episodes(self):
        """[(return, [(s, a), ...]), ...], best first."""
        return [(-negr, sa) for negr, _, sa in self._eps]

    def best_return(self):
        return self.episodes[0][0] if self._eps else 0.0


def extract_advice(memory, cap=100):
    """Deduplicated (s, a) pairs from the top episodes, best episode first
    and order preserved within an episode; at most `cap` pairs survive the
    bottleneck, so the best episode's pairs are kept preferentially."""
    seen, advice = set(), []
    for _, sa in memory.episodes:
        for pair in sa:
            if len(advice) == cap:
                return advice
            if pair not in seen:
                seen.add(pair)
                advice.append(pair)
    return advice


def apply_advice(Q, advice, value=5.0):
    """Pretrain a fresh student in place: bless each advised (s, a) with an
    optimistic value — far above any candy habit (0.3), below the true value
    near the big goal, and self-sustaining under gamma~1 bootstrapping."""
    for s, a in advice:
        Q[s, a] = value
    return Q


class TransitionLog:
    """Append-only record of a lifetime's (s, a, r, s2, done) transitions.

    The raw-experience inheritance channel (EXP4 v3): where the advice
    bottleneck hands a fresh student 100 curated (s, a) pairs, the
    reset-with-replay control hands it this log — the teacher's ENTIRE
    unfiltered experience, the Nikishin et al. (2022) reset channel — or a
    uniform sample of it, dose-matched to the bottleneck. `done` is stored
    exactly as it was fed to the teacher's Q-update, so a replay reproduces
    the update semantics (bootstrapping through cap truncations included).
    """

    def __init__(self):
        self.transitions = []

    def add(self, s, a, r, s2, done):
        self.transitions.append((int(s), int(a), float(r), int(s2), bool(done)))

    def __len__(self):
        return len(self.transitions)

    def shuffled(self, rng):
        """Every transition, in rng-shuffled order (the log is left intact)."""
        return [self.transitions[int(i)]
                for i in rng.permutation(len(self.transitions))]

    def sample(self, rng, n):
        """`n` distinct log entries drawn uniformly WITHOUT replacement, in
        rng order — matching the advice bottleneck's n distinct pairs."""
        idx = rng.choice(len(self.transitions), size=n, replace=False)
        return [self.transitions[int(i)] for i in idx]


def replay_pretrain(Q, transitions, lr, gamma):
    """One pass of standard Q-updates over `transitions` in the given order,
    at a FIXED lr, mutating Q in place. The update rule mirrors
    QLearner.update exactly (target = r, or r + gamma * max Q[s2] when not
    done) but touches no agent state: the student's age — hence its lr/eps
    schedules — is untouched, so replay pretraining, like advice priming,
    costs no env steps and no plasticity."""
    for s, a, r, s2, done in transitions:
        target = r if done else r + gamma * Q[s2].max()
        Q[s, a] += lr * (target - Q[s, a])
    return Q


def halflife_schedule(base, halflife):
    """base * 2^(-age/halflife): plasticity that halves every `halflife`
    updates — young agents learn and explore, old agents are rigid."""
    def schedule(age):
        return base * 2.0 ** (-age / halflife)
    return schedule
