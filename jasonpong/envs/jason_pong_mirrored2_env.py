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
MAX_STEPS_PER_GAME = int(1e4)


class JasonPongMirrored2Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(len(Actions))
        low = np.array([0, 0, 0, 0, -1, -1], dtype=np.float16)
        high = np.array([BOARD_WIDTH, BOARD_WIDTH, BOARD_WIDTH, BOARD_HEIGHT, 1, 1], dtype=np.float16)
        self.observation_space = spaces.Box(low, high, dtype=np.float16)
        self.obs_type = 'tuple'
        self.time = 0
        self.game_over = False
        self.winner = None
        self.paddle_positions = None
        self.ball_position = None
        self.ball_velocity = None
        self.win_reward_value = 1.0
        self.hit_reward_value = 0.1
        self.bonus_reward = [0.0, 0.0]

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.time = 0
        self.game_over = False
        self.winner = None
        self.paddle_positions = np.array([BOARD_WIDTH // 2, BOARD_WIDTH // 2], dtype=np.float16)
        ball_start_x = np.random.choice(BOARD_WIDTH - 2) + 1 if RANDOM_START_X else BOARD_WIDTH // 2
        self.ball_position = np.array([ball_start_x, BOARD_HEIGHT // 2], dtype=np.float16)
        self.ball_velocity = np.array(np.random.choice([-1, 1], size=(2,)), dtype=np.float16)

        return self._get_state()

    def step(self, actions) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float], bool, dict]:
        if not self.game_over:
            for player, action in enumerate(actions):
                if action == Actions.ACTION_LEFT.value:
                    self.paddle_positions[player] = max(self.paddle_positions[player] - 1, 0)
                elif action == Actions.ACTION_RIGHT.value:
                    self.paddle_positions[player] = min(self.paddle_positions[player] + 1, (BOARD_WIDTH - 1))
            self._update()

        states = self._get_state()[:]
        rewards = self._calculate_reward()
        info = {}
        self.time += 1

        return states, rewards, self.game_over, info

    def _update(self):
        self.ball_position += self.ball_velocity

        # bounce off paddle
        if self.ball_position[1] == PADDLE_HEIGHT:
            delta = abs(self.ball_position[0] - self.paddle_positions[0])
            if delta <= (PADDLE_WIDTH // 2):
                self.ball_velocity[1] *= -1
                self.bonus_reward[0] += self.hit_reward_value
        elif self.ball_position[1] == (BOARD_HEIGHT - PADDLE_HEIGHT - 1):
            delta = abs(self.ball_position[0] - self.paddle_positions[1])
            if delta <= (PADDLE_WIDTH // 2):
                self.ball_velocity[1] *= -1
                self.bonus_reward[1] += self.hit_reward_value

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
        elif self.time >= (MAX_STEPS_PER_GAME - 1):  # solved
            self.game_over = True
            self.winner = 2  # both players won

    def _calculate_reward(self) -> Tuple[float, float]:
        if not self.game_over:
            p1_reward = self.bonus_reward[0]
            p2_reward = self.bonus_reward[1]
            self.bonus_reward = [0.0, 0.0]
        else:
            p1_reward = 1.0 if self.winner == 0 else -1.0
            p2_reward = 1.0 if self.winner == 1 else -1.0
        return p1_reward, p2_reward

    def render(self, mode='human'):
        paddle_x, paddle_y = self.paddle_positions
        ball_x, ball_y = self.ball_position
        vel_x, vel_y = self.ball_velocity
        print(f'Time:{self.time} '
              f'Paddles:({paddle_x:.0f},{paddle_y:.0f}) '
              f'Ball_Pos:({ball_x:.1f},{ball_y:.1f}) '
              f'Ball_Vel:({vel_x:.1f},{vel_y:.1f})')

    def _get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        p1_paddle_positions = self.paddle_positions
        p1_ball_x = self.ball_position[0]
        p1_ball_y = self.ball_position[1]
        p1_ball_position = np.array([p1_ball_x, p1_ball_y], dtype=np.float16)
        p1_ball_velocity = self.ball_velocity
        p1_state = np.concatenate((p1_paddle_positions, p1_ball_position, p1_ball_velocity), axis=0)

        p2_paddle0 = (BOARD_WIDTH - self.paddle_positions[0] - 1)
        p2_paddle1 = (BOARD_WIDTH - self.paddle_positions[1] - 1)
        p2_ball_x = (BOARD_WIDTH - self.ball_position[0] - 1)
        p2_ball_y = (BOARD_HEIGHT - self.ball_position[1] - 1)
        p2_paddle_positions = np.array([p2_paddle1, p2_paddle0], dtype=np.float16)
        p2_ball_position = np.array([p2_ball_x, p2_ball_y], dtype=np.float16)
        p2_ball_velocity = self.ball_velocity * -1
        p2_state = np.concatenate((p2_paddle_positions, p2_ball_position, p2_ball_velocity), axis=0)

        return p1_state, p2_state
