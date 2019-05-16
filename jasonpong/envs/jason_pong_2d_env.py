from typing import Tuple

import numpy as np
from gym import spaces

from jasonpong.envs.jason_pong_env import JasonPongEnv, BOARD_WIDTH, BOARD_HEIGHT, PADDLE_HEIGHT, PADDLE_WIDTH


class JasonPong2dEnv(JasonPongEnv):

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=3, shape=(BOARD_HEIGHT, BOARD_WIDTH, 1), dtype=np.uint8)
        # self.obs_type = 'image'

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        player0_obs = super().reset()
        self.player = 1
        player1_obs = super().reset()
        self.player = 0
        return player0_obs, player1_obs

    def render(self, mode='human'):
        print(self._get_state())

    def _get_state(self) -> np.ndarray:
        state = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
        ball_x, ball_y = self.ball_position.astype(np.int8)
        state[ball_y, ball_x] = 2
        my_paddle_x = int(self.paddle_positions[self.player])
        my_paddle_y = 0 if self.player == 0 else BOARD_HEIGHT - 1
        opp_paddle_x = int(self.paddle_positions[1 - self.player])
        opp_paddle_y = BOARD_HEIGHT - 1 if self.player == 0 else 0
        state[my_paddle_y, my_paddle_x] = 1
        state[opp_paddle_y, opp_paddle_x] = 3

        return state
