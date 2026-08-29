"""PianoPiece: motif-structured sequence tasks for Variation Practice (H3).

A piece has 3 training passages (A, B, C) plus one novel transfer passage,
each a sequence of 4 motifs from a shared library of 6 motifs x 3 notes.
The correct key (of 8) at a position is given by a (motif, pos-in-motif)
table — the same motif demands the same fingers wherever it appears, which
is why practice on one passage transfers to another. Each TRAINING passage
additionally has exactly one exception position whose key overrides the
table (context-dependent fingering): learnable only from the passage/position
context. The novel passage is a fresh arrangement of the same motifs with no
exception, isolating what the motif features alone have stored.

An episode plays one passage start to finish: 12 key presses, +1 per correct
note, errors do not end the episode. Score = fraction correct. Structure is
generated from the constructor rng; the dynamics themselves are deterministic.

Feature maps (v2): `feature_map="motif"` (default) is the shared-feature
representation above. `feature_map="local"` is the mechanism control —
phi = onehot((passage, position)) pair, one indicator per state, NO shared
slots (a passage-local tabular equivalent): interference and transfer are
both impossible by construction. The piece structure is identical either way.
"""

import numpy as np


class PianoPiece:
    n_keys = 8
    n_actions = 8  # actions are key presses
    n_motifs = 6
    motif_len = 3
    n_slots = 4  # motifs per passage
    passage_len = n_slots * motif_len
    n_train_passages = 3
    n_passages = 4
    NOVEL = 3  # passage index of the novel transfer passage
    # "motif" map: onehot(motif, pos-in-motif) ++ onehot(passage) ++
    # onehot(position); "local" map: onehot((passage, position)) pair only
    n_features = n_motifs * motif_len + n_passages + passage_len
    FEATURE_MAPS = ("motif", "local")

    def __init__(self, rng=None, feature_map="motif"):
        if feature_map not in self.FEATURE_MAPS:
            raise ValueError(f"unknown feature_map {feature_map!r}")
        self.feature_map = feature_map
        self.n_features = (self.n_passages * self.passage_len
                           if feature_map == "local"
                           else type(self).n_features)
        rng = rng if rng is not None else np.random.default_rng()
        # 6 distinct motifs: rows of 3 keys
        while True:
            self.motif_table = rng.integers(0, self.n_keys,
                                            size=(self.n_motifs, self.motif_len))
            if len({tuple(r) for r in self.motif_table.tolist()}) == self.n_motifs:
                break
        # 4 pairwise-distinct passages of 4 distinct motifs; the training
        # passages must jointly cover the whole library (so the novel
        # passage recombines only practiced material)
        while True:
            self.passages = [rng.permutation(self.n_motifs)[:self.n_slots]
                             for _ in range(self.n_passages)]
            seqs = [tuple(p.tolist()) for p in self.passages]
            if (len(set(seqs)) == self.n_passages
                    and set().union(*seqs[:3]) == set(range(self.n_motifs))):
                break
        # one exception per training passage: a key that contradicts the table
        self.exceptions = {}
        for p in range(self.n_train_passages):
            pos = int(rng.integers(self.passage_len))
            m, j = self.motif_at(p, pos)
            while True:
                key = int(rng.integers(self.n_keys))
                if key != self.motif_table[m, j]:
                    break
            self.exceptions[p] = (pos, key)
        self.passage = 0
        self.position = 0

    def motif_at(self, passage, position):
        """(motif id, position within motif) sounding at this position."""
        return int(self.passages[passage][position // self.motif_len]), \
            position % self.motif_len

    def correct_key(self, passage, position):
        exc = self.exceptions.get(passage)
        if exc is not None and exc[0] == position:
            return exc[1]
        m, j = self.motif_at(passage, position)
        return int(self.motif_table[m, j])

    def features(self, passage, position):
        """Feature vector for a state; only valid for position < passage_len.

        "motif" map: onehot(motif, pos-in-motif) ++ onehot(passage) ++
        onehot(position). The motif and position blocks are shared across
        passages — that is where contextual interference lives.

        "local" map (mechanism control): onehot((passage, position)) — one
        indicator per state, NO shared slots, so cross-passage interference
        and transfer are impossible by construction.
        """
        phi = np.zeros(self.n_features)
        if self.feature_map == "local":
            phi[passage * self.passage_len + position] = 1.0
            return phi
        m, j = self.motif_at(passage, position)
        phi[m * self.motif_len + j] = 1.0
        phi[self.n_motifs * self.motif_len + passage] = 1.0
        phi[self.n_motifs * self.motif_len + self.n_passages + position] = 1.0
        return phi

    def reset(self, passage):
        self.passage = passage
        self.position = 0
        return (passage, 0)

    def step(self, a):
        correct = int(a) == self.correct_key(self.passage, self.position)
        self.position += 1
        done = self.position == self.passage_len
        return ((self.passage, self.position), float(correct), done,
                {"correct": bool(correct)})

    def structure(self):
        """Plain-python description of the piece (for JSON export / viz)."""
        exc = {}
        for p, (pos, key) in self.exceptions.items():
            m, j = self.motif_at(p, pos)
            exc[p] = {"position": pos, "key": key,
                      "table_key": int(self.motif_table[m, j])}
        return {
            "motif_table": self.motif_table.tolist(),
            "passages": [p.tolist() for p in self.passages],
            "exceptions": exc,
            "correct_keys": [[self.correct_key(p, i)
                              for i in range(self.passage_len)]
                             for p in range(self.n_passages)],
        }
