from typing import Tuple

import numpy as np

from jasonpong.envs.jason_pong_env import JasonPongEnv, BOARD_WIDTH, BOARD_HEIGHT, PADDLE_HEIGHT, PADDLE_WIDTH


class JasonPongMirroredEnv(JasonPongEnv):

    def __init__(self):
        super().__init__()
        self.win_reward_value = 0.0
        self.bonus_reward_value = 1.0

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        # reverse the action for player 1
        if self.player == 1:
            action = 2 - action

        return super().step(action)

    def _get_state(self) -> np.ndarray:
        # reverse the observation for player 1
        if self.player == 0:
            # paddle_positions = self.paddle_positions / BOARD_WIDTH
            # ball_x = self.ball_position[0] / BOARD_WIDTH
            # ball_y = self.ball_position[1] / BOARD_HEIGHT

            paddle_positions = self.paddle_positions
            ball_x = self.ball_position[0]
            ball_y = self.ball_position[1]

            ball_position = np.array([ball_x, ball_y], dtype=np.float16)
            ball_velocity = self.ball_velocity
        else:
            # paddle0 = (BOARD_WIDTH - self.paddle_positions[0] - 1) / BOARD_WIDTH
            # paddle1 = (BOARD_WIDTH - self.paddle_positions[1] - 1) / BOARD_WIDTH
            # ball_x = (BOARD_WIDTH - self.ball_position[0] - 1) / BOARD_WIDTH
            # ball_y = (BOARD_HEIGHT - self.ball_position[1] - 1) / BOARD_HEIGHT

            paddle0 = (BOARD_WIDTH - self.paddle_positions[0] - 1)
            paddle1 = (BOARD_WIDTH - self.paddle_positions[1] - 1)
            ball_x = (BOARD_WIDTH - self.ball_position[0] - 1)
            ball_y = (BOARD_HEIGHT - self.ball_position[1] - 1)

            paddle_positions = np.array([paddle1, paddle0], dtype=np.float16)
            ball_position = np.array([ball_x, ball_y], dtype=np.float16)
            ball_velocity = self.ball_velocity * -1

        return np.concatenate((paddle_positions, ball_position, ball_velocity), axis=0)
