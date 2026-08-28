"""SoccerGrid: a minimal soccer pitch for the microtask curriculum (H2).

An 11x7 pitch (W x H); the goal is the middle 3 cells of the right edge,
goal center (3, 10). The full game: fetch the ball, dribble it (the ball
travels with the agent once picked up), and shoot. A shot while carrying
scores with p = max(0, 1 - d/5), d = Chebyshev distance to the goal center —
point-blank shots are sure goals, 5+ cells out never scores — and goal or
miss, the ball is gone: every shot ends the episode, so shot selection is
part of the skill. Reward is 1 on a goal, 0 otherwise; no shaping anywhere,
which lets curricula be expressed purely as start-state distributions.

State = (agent cell, ball status), ball status in {at cell k, carried,
scored}: 77 * 79 = 6083 discrete states. Actions: 4 moves + shoot. Moves are
deterministic; walking off the pitch leaves the agent in place; stepping onto
the ball's cell picks it up (spawning on it does too). Shooting without the
ball is a wasted step. The 100-step episode cap is enforced by the training
loop, as with GridHome.
"""

import numpy as np

CARRIED = "carried"
SCORED = "scored"


class SoccerGrid:
    UP, RIGHT, DOWN, LEFT, SHOOT = 0, 1, 2, 3, 4
    MOVES = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
    n_actions = 5
    W, H = 11, 7
    n_cells = W * H
    _BALL_CARRIED, _BALL_SCORED = n_cells, n_cells + 1
    n_ball = n_cells + 2
    n_states = n_cells * n_ball
    GOAL_CELLS = ((2, 10), (3, 10), (4, 10))
    GOAL_CENTER = (3, 10)
    KICKOFF_AGENT = (3, 0)
    KICKOFF_BALL = (3, 5)

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def cell_of(self, rc):
        return int(rc[0]) * self.W + int(rc[1])

    def state_of(self, agent_rc, ball):
        """Encode (agent cell, ball status); ball is a cell, CARRIED or SCORED."""
        if ball == CARRIED:
            code = self._BALL_CARRIED
        elif ball == SCORED:
            code = self._BALL_SCORED
        else:
            code = self.cell_of(ball)
        return self.cell_of(agent_rc) * self.n_ball + code

    def decode(self, s):
        cell, code = divmod(int(s), self.n_ball)
        if code == self._BALL_CARRIED:
            ball = CARRIED
        elif code == self._BALL_SCORED:
            ball = SCORED
        else:
            ball = divmod(code, self.W)
        return divmod(cell, self.W), ball

    def shot_p(self, rc):
        """P(goal) for a shot from rc: max(0, 1 - d/5), d Chebyshev to goal center."""
        d = max(abs(rc[0] - self.GOAL_CENTER[0]), abs(rc[1] - self.GOAL_CENTER[1]))
        return max(0.0, 1.0 - d / 5.0)

    def _s(self):
        return self.state_of(self.pos, self.ball)

    def reset(self, agent=None, ball=None):
        """Start an episode; defaults to kickoff. Curricula differ ONLY here."""
        self.pos = tuple(agent) if agent is not None else self.KICKOFF_AGENT
        if ball == CARRIED:
            self.ball = CARRIED
        else:
            self.ball = tuple(ball) if ball is not None else self.KICKOFF_BALL
            if self.pos == self.ball:
                self.ball = CARRIED
        return self._s()

    def step(self, a):
        if a == self.SHOOT:
            if self.ball != CARRIED:
                return self._s(), 0.0, False, {"shot": False, "scored": False}
            p = self.shot_p(self.pos)
            scored = bool(self.rng.random() < p)
            if scored:
                self.ball = SCORED
            return (self._s(), float(scored), True,
                    {"shot": True, "scored": scored, "shot_p": p})
        dr, dc = self.MOVES[a]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        if 0 <= r < self.H and 0 <= c < self.W:
            self.pos = (r, c)
        if self.ball != CARRIED and self.pos == self.ball:
            self.ball = CARRIED
        return self._s(), 0.0, False, {"shot": False, "scored": False}
