"""GridHome: apartment-shaped gridworlds for the Blindfold Test (H1).

Two homes with identical outer dimensions and identical bed (S) and fridge
(G) coordinates, but different floor plans. Walls block movement (with a
`bump` signal, like touch). Motor noise: with probability `slip` the agent
moves perpendicular to its intent.
"""

import numpy as np

HOME_A = """
#############
#S...#......#
#....#.G....#
#....#......#
#.##.####.###
#...........#
#.#######.#.#
#.#.....#.#.#
#.#.###.#.#.#
#...#.......#
#############
"""

HOME_B = """
#############
#S.#........#
#..#...G#...#
#..#.###....#
#...........#
###.#####.###
#....#......#
#.##.#.####.#
#.#..#....#.#
#...........#
#############
"""


class GridHome:
    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
    MOVES = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
    n_actions = 4

    def __init__(self, map_str, slip=0.0, rng=None):
        rows = [r for r in map_str.strip().splitlines()]
        if len({len(r) for r in rows}) != 1:
            raise ValueError("ragged map")
        self.grid = np.array([list(r) for r in rows])
        self.H, self.W = self.grid.shape
        starts = np.argwhere(self.grid == "S")
        goals = np.argwhere(self.grid == "G")
        if len(starts) != 1 or len(goals) != 1:
            raise ValueError("map must contain exactly one S and one G")
        self.start = tuple(starts[0])
        self.goal = tuple(goals[0])
        self.slip = slip
        self.rng = rng if rng is not None else np.random.default_rng()
        self.pos = self.start

    @property
    def shape(self):
        return (self.H, self.W)

    @property
    def n_states(self):
        return self.H * self.W

    def state_of(self, rc):
        return rc[0] * self.W + rc[1]

    def pos_of(self, s):
        return divmod(int(s), self.W)

    def walls_rc(self):
        return {tuple(rc) for rc in np.argwhere(self.grid == "#")}

    def reset(self):
        self.pos = self.start
        return self.state_of(self.pos)

    def step(self, a):
        if self.rng.random() < self.slip:
            a = (a + self.rng.choice([1, 3])) % 4  # perpendicular slip
        dr, dc = self.MOVES[a]
        target = (self.pos[0] + dr, self.pos[1] + dc)
        bump = self.grid[target] == "#"
        if not bump:
            self.pos = target
        done = self.pos == self.goal
        return self.state_of(self.pos), float(done), done, {"bump": bool(bump)}

    def shortest_path_len(self):
        """BFS distance from S to G; None if unreachable."""
        from collections import deque

        dist = {self.start: 0}
        q = deque([self.start])
        while q:
            rc = q.popleft()
            if rc == self.goal:
                return dist[rc]
            for dr, dc in self.MOVES.values():
                nxt = (rc[0] + dr, rc[1] + dc)
                if self.grid[nxt] != "#" and nxt not in dist:
                    dist[nxt] = dist[rc] + 1
                    q.append(nxt)
        return None
