from gym import Env
import numpy as np
from enum import Enum


class Action(Enum):
    ACTION_NOOP = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2


class Pong(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.game_over = False
        self.winner = None
        self.cycle = 0
        self.player_turn = 0
        self.bat_pos = [0, 0]
        self.ball_pos = np.array([5, 5])
        self.ball_velocity = np.asarray([1, 1])

    def step(self, action):
        if self.game_over:
            return

        if action == Action.ACTION_LEFT.value:
            self.bat_pos[self.player_turn] = max(self.bat_pos[self.player_turn] - 1, 0)
        elif action == Action.ACTION_RIGHT.value:
            self.bat_pos[self.player_turn] = min(self.bat_pos[self.player_turn] + 1, 10)

        self.player_turn = (self.player_turn + 1) % 2
        if self.player_turn == 0:
            self._update()

        state = self._get_state()
        reward = 0
        game_over = self.game_over
        info = {}

        return state, reward, game_over, info

    def reset(self):
        self.ball_pos = np.asarray([5, 5])
        self.bat_pos = [0, 0]
        self.ball_velocity = np.asarray([1, 1])
        self.game_over = False
        self.player_turn = 0
        self.winner = None
        self.cycle = 0

    def _update(self):
        self.cycle += 1
        self.ball_pos += self.ball_velocity

        # bounce off bat
        if (self.ball_pos[1] == 1 and abs(self.ball_pos[0] - self.bat_pos[0]) <= 1) or (
                self.ball_pos[1] == 10 and abs(self.ball_pos[0] - self.bat_pos[1]) <= 1):  # player 1 bat y axis
            self.ball_velocity[1] *= -1

        # reflect off side walls
        if self.ball_pos[0] >= 10 or self.ball_pos[0] <= 0:
            self.ball_velocity[0] *= -1

        # lose and win conditions
        if self.ball_pos[1] >= 11:
            self.game_over = True
            self.winner = 0
        if self.ball_pos[1] <= 0:
            self.game_over = True
            self.winner = 1

    def render(self, mode='human'):
        print('Cycle:{} Player:{} Bat:{} Ball_P:({},{}) Ball_V:({},{})'.format(self.cycle, self.player_turn,
                                                                               *self._get_state()))

    def _get_state(self):
        return np.asarray([self.bat_pos[self.player_turn]] + list(self.ball_pos) + list(self.ball_velocity))
