from enum import Enum
from typing import Tuple

import gym
from gym import spaces
import numpy as np


class Actions(Enum):
    ACTION_LEFT = 0
    ACTION_NOOP = 1
    ACTION_RIGHT = 2


BOARD_WIDTH = 9
BOARD_HEIGHT = 17
PADDLE_WIDTH = 1
PADDLE_HEIGHT = 1
RANDOM_START_X = True


class JasonPongEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(len(Actions))
        low = np.array([0, 0, 0, 0, 0, 0])
        high = np.array([BOARD_WIDTH, BOARD_WIDTH, BOARD_WIDTH, BOARD_HEIGHT, 1, 1])
        self.observation_space = spaces.Box(low, high)
        self.time = 0
        self.game_over = False
        self.player = 0
        self.winner = None
        self.paddle_positions = None
        self.ball_position = None
        self.ball_velocity = None
        self.bonus_reward = [0.0, 0.0]

    def reset(self) -> np.ndarray:
        self.time = 0
        self.game_over = False
        self.player = 0
        self.winner = None
        self.paddle_positions = np.array([BOARD_WIDTH // 2, BOARD_WIDTH // 2], dtype=np.int8)
        ball_start_x = np.random.choice(BOARD_WIDTH - 2) + 1 if RANDOM_START_X else BOARD_WIDTH // 2
        self.ball_position = np.array([ball_start_x, BOARD_HEIGHT // 2], dtype=np.int8)
        self.ball_velocity = np.array(np.random.choice([-1, 1], size=(2,)), dtype=np.int8)

        return self._get_state()

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        if not self.game_over:
            if action == Actions.ACTION_LEFT.value:
                self.paddle_positions[self.player] = max(self.paddle_positions[self.player] - 1, 0)
            elif action == Actions.ACTION_RIGHT.value:
                self.paddle_positions[self.player] = min(self.paddle_positions[self.player] + 1, (BOARD_WIDTH - 1))

            if self.player == 1:
                self._update()

        state = self._get_state()[:]
        # state = self._get_state()
        reward = self._calculate_reward()
        info = {}

        if self.player == 1:
            self.time += 1
        self.player = (self.player + 1) % 2

        return state, reward, self.game_over, info

    def _update(self):
        self.ball_position += self.ball_velocity

        # bounce off paddle
        if self.ball_position[1] == PADDLE_HEIGHT:
            delta = abs(self.ball_position[0] - self.paddle_positions[0])
            if delta <= (PADDLE_WIDTH // 2):
                self.ball_velocity[1] *= -1
                self.bonus_reward[0] = 0.1
        elif self.ball_position[1] == (BOARD_HEIGHT - PADDLE_HEIGHT):
            delta = abs(self.ball_position[0] - self.paddle_positions[1])
            if delta <= (PADDLE_WIDTH // 2):
                self.ball_velocity[1] *= -1
                self.bonus_reward[1] = 0.1

        # reflect off side walls
        if self.ball_position[0] >= (BOARD_WIDTH - 1) or self.ball_position[0] <= 0:
            self.ball_velocity[0] *= -1

        # lose and win conditions
        if self.ball_position[1] >= (BOARD_HEIGHT - 1):
            self.game_over = True
            self.winner = 0
        elif self.ball_position[1] <= 0:
            self.game_over = True
            self.winner = 1

    def _calculate_reward(self) -> float:
        if not self.game_over:
            # reward = 0.0
            reward = self.bonus_reward[self.player]
            self.bonus_reward[self.player] = 0.0
        elif self.player == self.winner:
            reward = 1.0
        else:
            reward = -1.0
        return reward

    def render(self, mode='human'):
        print('Time:{} Paddles:({}, {}) Ball_Pos:({},{}) Ball_Vel:({},{})'.format(self.time, *self._get_state()))

    def _get_state(self) -> np.ndarray:
        return np.concatenate((self.paddle_positions, self.ball_position, self.ball_velocity), axis=0)
