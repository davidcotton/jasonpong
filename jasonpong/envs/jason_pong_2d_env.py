import numpy as np

from jasonpong.envs.jason_pong_env import JasonPongEnv, BOARD_WIDTH, BOARD_HEIGHT, PADDLE_HEIGHT, PADDLE_WIDTH


class JasonPong2dEnv(JasonPongEnv):

    def render(self, mode='human'):
        print(self._get_state())

    def _get_state(self) -> np.ndarray:
        state = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
        state[self.ball_position[1], self.ball_position[0]] = 2
        my_paddle_x = self.paddle_positions[self.player]
        my_paddle_y = 0 if self.player == 0 else BOARD_HEIGHT - 1
        opp_paddle_x = self.paddle_positions[1 - self.player]
        opp_paddle_y = BOARD_HEIGHT - 1 if self.player == 0 else 0
        state[my_paddle_y, my_paddle_x] = 1
        state[opp_paddle_y, opp_paddle_x] = 3

        return state
