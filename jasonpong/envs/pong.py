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
        self.time = 0
        self.game_over = False
        self.winner = None
        self.paddle_positions = None
        self.ball_position = None
        self.ball_velocity = None

    def reset(self):
        self.time = 0
        self.game_over = False
        self.winner = None
        self.paddle_positions = [BOARD_WIDTH // 2, BOARD_WIDTH // 2]
        self.ball_position = np.array([BOARD_WIDTH // 2, BOARD_HEIGHT // 2])
        self.ball_velocity = np.array([1, 1])

        return self._get_state()

    def step(self, actions):
        if self.game_over:
            return

        for player, action in enumerate(actions):
            if action == Action.ACTION_LEFT.value:
                self.paddle_positions[player] = max(self.paddle_positions[player] - 1, 0)
            elif action == Action.ACTION_RIGHT.value:
                self.paddle_positions[player] = min(self.paddle_positions[player] + 1, BOARD_WIDTH)

        self._update()

        state = self._get_state()[:]
        if self.game_over:
            reward = (1, -1) if self.winner == 0 else (-1, 1)
        else:
            reward = (0, 0)
        info = {}

        self.time += 1

        return state, reward, self.game_over, info

    def _update(self):
        self.ball_position += self.ball_velocity

        # bounce off paddle
        if self.ball_position[1] == PADDLE_HEIGHT:
            delta = abs(self.ball_position[0] - self.paddle_positions[0])
            if delta <= (PADDLE_WIDTH // 2):
                self.ball_velocity[1] *= -1
        elif self.ball_position[1] == (BOARD_HEIGHT - PADDLE_HEIGHT):
            delta = abs(self.ball_position[0] - self.paddle_positions[1])
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
        print('Time:{} Paddles:({}, {}) Ball_P:({},{}) Ball_V:({},{})'.format(self.time, *self._get_state()))

    def _get_state(self):
        return np.concatenate((np.array(self.paddle_positions), self.ball_position, self.ball_velocity), axis=0)
