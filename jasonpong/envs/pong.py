from enum import Enum

from gym import Env
from gym import spaces
import numpy as np


class Action(Enum):
    ACTION_NOOP = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2


BOARD_WIDTH = 8
BOARD_HEIGHT = 16
PADDLE_WIDTH = 2
PADDLE_HEIGHT = 1


class Pong(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(5)
        self.game_over = False
        self.winner = None
        self.time = 0
        self.player_turn = 0
        self.paddle_position = None
        self.ball_position = None
        self.ball_velocity = None

    def step(self, action):
        if self.game_over:
            return

        if action == Action.ACTION_LEFT.value:
            self.paddle_position[self.player_turn] = max(self.paddle_position[self.player_turn] - 1, 0)
        elif action == Action.ACTION_RIGHT.value:
            self.paddle_position[self.player_turn] = min(self.paddle_position[self.player_turn] + 1, BOARD_WIDTH)

        if self.player_turn == 1:
            self._update()

        state = self._get_state()[:]
        if self.game_over:
            reward = 1 if self.winner == self.player_turn else -1
        else:
            reward = 0
        info = {}

        self.player_turn = (self.player_turn + 1) % 2
        if self.player_turn == 1:
            self.time += 1

        return state, reward, self.game_over, info

    def reset(self):
        self.game_over = False
        self.winner = None
        self.time = 0
        self.player_turn = 0
        self.paddle_position = [BOARD_WIDTH // 2, BOARD_WIDTH // 2]
        self.ball_position = np.array([BOARD_WIDTH // 2, BOARD_HEIGHT // 2])
        self.ball_velocity = np.array([1, 1])

        return self._get_state()

    def _update(self):
        self.ball_position += self.ball_velocity

        # bounce off paddle
        if self.ball_position[1] == PADDLE_HEIGHT:
            delta = abs(self.ball_position[0] - self.paddle_position[0])
            if delta <= (PADDLE_WIDTH // 2):
                self.ball_velocity[1] *= -1
        elif self.ball_position[1] == (BOARD_HEIGHT - PADDLE_HEIGHT):
            delta = abs(self.ball_position[0] - self.paddle_position[1])
            if delta <= (PADDLE_WIDTH // 2):
                self.ball_velocity[1] *= -1

        # reflect off side walls
        if self.ball_position[0] >= BOARD_WIDTH or self.ball_position[0] <= 0:
            self.ball_velocity[0] *= -1

        # lose and win conditions
        if self.ball_position[1] >= BOARD_HEIGHT:
            self.game_over = True
            self.winner = 0
        if self.ball_position[1] <= 0:
            self.game_over = True
            self.winner = 1

    def render(self, mode='human'):
        print('Time:{} Player:{} Bat:{} Ball_P:({},{}) Ball_V:({},{})'.format(self.time, self.player_turn,
                                                                              *self._get_state()))

    def _get_state(self):
        return np.array([self.paddle_position[self.player_turn]] + list(self.ball_position) + list(self.ball_velocity))
