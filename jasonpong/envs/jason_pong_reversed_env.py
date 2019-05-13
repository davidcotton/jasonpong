from typing import Tuple

import numpy as np

from jasonpong.envs.jason_pong_env import JasonPongEnv, BOARD_WIDTH, BOARD_HEIGHT, PADDLE_HEIGHT, PADDLE_WIDTH


class JasonPongReversedEnv(JasonPongEnv):

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        player0_state = super().reset()
        self.player = 1
        player1_state = self._get_state()
        self.player = 0

        return player0_state, player1_state

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        # reverse the action for player 1
        if self.player == 1:
            action = 2 - action

        return super().step(action)

    def _get_state(self) -> np.ndarray:
        # reverse the observation for player 1
        if self.player == 0:
            paddle_positions = self.paddle_positions
            ball_position = self.ball_position
            ball_velocity = self.ball_velocity
        else:
            paddle_positions = self.paddle_positions[::-1]
            ball_x = BOARD_WIDTH - self.ball_position[0]
            ball_y = BOARD_HEIGHT - self.ball_position[1]
            ball_position = np.array([ball_x, ball_y])
            ball_velocity = self.ball_velocity * -1

        return np.concatenate((paddle_positions, ball_position, ball_velocity), axis=0)
