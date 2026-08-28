import numpy as np
import pytest

from devrl.envs.piano import PianoPiece


def make_env(seed=0):
    return PianoPiece(rng=np.random.default_rng(seed))


def test_constants_and_structure_shapes():
    env = make_env()
    assert env.n_keys == 8 and env.n_actions == 8
    assert env.n_motifs == 6 and env.motif_len == 3
    assert env.passage_len == 12 and env.n_passages == 4 and env.NOVEL == 3
    assert env.n_features == 6 * 3 + 4 + 12
    assert env.motif_table.shape == (6, 3)
    assert env.motif_table.min() >= 0 and env.motif_table.max() < 8
    assert len(env.passages) == 4
    for p in env.passages:
        assert len(p) == 4 and len(set(p.tolist())) == 4  # 4 distinct motifs
        assert all(0 <= m < 6 for m in p.tolist())


def test_structure_invariants_across_generations():
    for seed in range(30):
        env = make_env(seed)
        seqs = [tuple(p.tolist()) for p in env.passages]
        assert len(set(seqs)) == 4  # all passages distinct arrangements
        # training passages jointly cover the whole motif library
        assert set().union(*seqs[:3]) == set(range(6))
        # no two motifs are identical note-triples
        assert len({tuple(r) for r in env.motif_table.tolist()}) == 6
        # exactly one exception per TRAINING passage, none for the novel one
        assert set(env.exceptions) == {0, 1, 2}
        for p, (pos, key) in env.exceptions.items():
            assert 0 <= pos < 12 and 0 <= key < 8
            m, j = env.motif_at(p, pos)
            assert key != env.motif_table[m, j]  # override actually overrides


def test_same_seed_same_piece_different_seed_differs():
    a, b = make_env(7), make_env(7)
    assert np.array_equal(a.motif_table, b.motif_table)
    assert all(np.array_equal(x, y) for x, y in zip(a.passages, b.passages))
    assert a.exceptions == b.exceptions
    c = make_env(8)
    same = (np.array_equal(a.motif_table, c.motif_table)
            and all(np.array_equal(x, y) for x, y in zip(a.passages, c.passages))
            and a.exceptions == c.exceptions)
    assert not same


def test_motif_at_maps_position_to_slot_and_offset():
    env = make_env(0)
    for p in range(4):
        for i in range(12):
            m, j = env.motif_at(p, i)
            assert m == env.passages[p][i // 3] and j == i % 3


def test_correct_key_is_motif_table_except_at_exception():
    env = make_env(3)
    for p in range(4):
        exc = env.exceptions.get(p)
        for i in range(12):
            m, j = env.motif_at(p, i)
            want = env.motif_table[m, j]
            if exc is not None and exc[0] == i:
                want = exc[1]
            assert env.correct_key(p, i) == want
    assert 3 not in env.exceptions  # the novel passage plays pure motif table


def test_perfect_play_scores_12_and_episode_lasts_12_steps():
    env = make_env(0)
    assert env.reset(0) == (0, 0)
    total = 0.0
    for i in range(12):
        s2, r, done, info = env.step(env.correct_key(0, i))
        total += r
        assert r == 1.0 and info["correct"] is True
        assert s2 == (0, i + 1)
        assert done is (i == 11)
    assert total == 12.0


def test_wrong_key_scores_zero_and_episode_continues():
    env = make_env(0)
    env.reset(0)
    wrong = (env.correct_key(0, 0) + 1) % 8
    s2, r, done, info = env.step(wrong)
    assert r == 0.0 and done is False and info["correct"] is False
    assert s2 == (0, 1)  # an error does not stop the piece


def test_reset_switches_passage_and_rewinds():
    env = make_env(0)
    env.reset(0)
    env.step(0)
    env.step(1)
    assert env.reset(2) == (2, 0)


def test_features_onehot_blocks():
    env = make_env(0)
    phi = env.features(2, 7)  # passage 2, position 7 -> motif slot 2, offset 1
    m = int(env.passages[2][2])
    assert phi.shape == (34,) and phi.sum() == 3.0
    assert phi[m * 3 + 1] == 1.0        # (motif, pos-in-motif) block
    assert phi[18 + 2] == 1.0           # passage block
    assert phi[18 + 4 + 7] == 1.0       # position block
    # the novel passage activates its own (never-trained) passage slot
    assert env.features(3, 0)[18 + 3] == 1.0


def test_shared_motif_features_are_identical_across_passages():
    # Interference channel: the same motif in two different passages must
    # activate the exact same (motif, pos-in-motif) features.
    env = make_env(1)
    p0, p1 = [set(p.tolist()) for p in env.passages[:2]]
    m = min(p0 & p1)  # any two 4-of-6 passages share >= 2 motifs
    s0 = env.passages[0].tolist().index(m)
    s1 = env.passages[1].tolist().index(m)
    a = env.features(0, 3 * s0)
    b = env.features(1, 3 * s1)
    assert np.array_equal(a[:18], b[:18])
    assert a[m * 3] == 1.0


def test_random_play_scores_at_chance_one_eighth():
    # statistical: uniform key presses earn 1/8 per note in expectation
    env = make_env(0)
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(400):
        env.reset(int(rng.integers(4)))
        done, tot = False, 0.0
        while not done:
            _, r, done, _ = env.step(int(rng.integers(8)))
            tot += r
        scores.append(tot / env.passage_len)
    assert abs(np.mean(scores) - 1 / 8) < 0.02  # ~4 sd tolerance


def test_exception_positions_uniform_over_generations():
    # statistical: exception positions are drawn uniformly over the 12 slots
    counts = np.zeros(12)
    n = 400
    for seed in range(n):
        for pos, _ in make_env(seed).exceptions.values():
            counts[pos] += 1
    freqs = counts / (3 * n)
    assert np.all(np.abs(freqs - 1 / 12) < 0.03)  # ~4 sd tolerance per bin
