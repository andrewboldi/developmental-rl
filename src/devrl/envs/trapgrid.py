"""TrapGrid: candy-trap gridworld for generational teaching (H4).

A big terminal reward (10) waits at the far right; three candy cells
(reward 0.3, also terminal) sit a few steps from the left-center start.
Candy ends the episode, so undirected exploration keeps being absorbed near
home — a local optimum that throttles deep exploration toward the mountain.
Moves are deterministic; episodes truncate (info["truncated"], NOT terminal,
so learners still bootstrap through the cutoff) at `cap` steps.
"""

from collections import deque

import numpy as np

TRAP_MAP = """
###############
#.............#
#..C..........#
#.............#
#.............#
#S...C.......G#
#.............#
#.............#
#..C..........#
#.............#
###############
"""


class TrapGrid:
    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
    MOVES = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
    n_actions = 4
    CANDY_REWARD, GOAL_REWARD = 0.3, 10.0

    def __init__(self, map_str=TRAP_MAP, cap=120):
        rows = map_str.strip().splitlines()
        if len({len(r) for r in rows}) != 1:
            raise ValueError("ragged map")
        self.grid = np.array([list(r) for r in rows])
        self.H, self.W = self.grid.shape
        starts = np.argwhere(self.grid == "S")
        goals = np.argwhere(self.grid == "G")
        if len(starts) != 1 or len(goals) != 1:
            raise ValueError("map must contain exactly one S and one G")
        self.start = (int(starts[0][0]), int(starts[0][1]))
        self.goal = (int(goals[0][0]), int(goals[0][1]))
        self.candies = {(int(r), int(c)) for r, c in np.argwhere(self.grid == "C")}
        self.cap = cap
        self.pos = self.start
        self.t = 0

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

    def reset(self):
        self.pos = self.start
        self.t = 0
        return self.state_of(self.pos)

    def step(self, a):
        dr, dc = self.MOVES[a]
        target = (self.pos[0] + dr, self.pos[1] + dc)
        bump = self.grid[target] == "#"
        if not bump:
            self.pos = target
        self.t += 1
        if self.pos == self.goal:
            r, done = self.GOAL_REWARD, True
        elif self.pos in self.candies:
            r, done = self.CANDY_REWARD, True
        else:
            r, done = 0.0, False
        info = {"bump": bool(bump), "truncated": not done and self.t >= self.cap}
        return self.state_of(self.pos), r, done, info

    def shortest_path_len(self):
        """BFS S -> G dodging walls AND candy (stepping on candy ends the
        episode, so a viable route must avoid it); None if unreachable."""
        dist = {self.start: 0}
        q = deque([self.start])
        while q:
            rc = q.popleft()
            if rc == self.goal:
                return dist[rc]
            for dr, dc in self.MOVES.values():
                nxt = (rc[0] + dr, rc[1] + dc)
                if self.grid[nxt] != "#" and nxt not in self.candies and nxt not in dist:
                    dist[nxt] = dist[rc] + 1
                    q.append(nxt)
        return None
