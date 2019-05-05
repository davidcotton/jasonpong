from abc import ABC, abstractmethod

import numpy as np

from jasonpong.envs.pong import Pong, BOARD_WIDTH, BOARD_HEIGHT, Action


class Agent(ABC):

    def __init__(self, player) -> None:
        super().__init__()
        self.player = player

    @abstractmethod
    def forward(self, state) -> int:
        pass

    def backwards(self, reward, game_over, next_state):
        pass


class SimpleAgent(Agent):
    def forward(self, state) -> int:
        if state[0] < 5:
            action = 2  # right
        elif state[0] > 5:
            action = 1  # left
        else:
            action = 0
        return action


class QTableAgent(Agent):
    def __init__(self, player) -> None:
        super().__init__(player)
        self.gamma = 0.95
        self.learning_rate = 0.8
        self.num_actions = len(Action)
        self.q = np.zeros([BOARD_WIDTH, BOARD_WIDTH, BOARD_HEIGHT, self.num_actions])
        self.epsilon = 0.1
        self.last_state = None
        self.last_action = None

    def forward(self, state) -> int:
        paddle_position = state[self.player] - 1
        ball_x = state[2] - 1
        ball_y = state[3] - 1
        q = self.q[paddle_position, ball_x, ball_y]
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.num_actions - 1)
        else:
            # action = np.argmax(q)
            best_actions = np.argwhere(q == np.amax(q)).flatten()
            action = np.random.choice(best_actions)
        self.last_state = paddle_position, ball_x, ball_y
        self.last_action = action
        return action

    def backwards(self, reward, game_over, next_state):
        paddle_pos = self.last_state[0]
        ball_x = self.last_state[1]
        ball_y = self.last_state[2]
        action = self.last_action

        next_q = self.q[next_state[self.player]-1, next_state[2]-1, next_state[3]-1]
        target_q = reward + self.gamma * np.max(next_q)
        prev_q = self.q[paddle_pos, ball_x, ball_y, action]
        self.q[paddle_pos, ball_x, ball_y, action] += self.learning_rate * (target_q - prev_q)
        foo = 1


def play_game():
    env = Pong()
    agent_cls = QTableAgent
    agents = [agent_cls(i) for i in range(2)]

    for episode in range(10000):
        # print('episode:', episode)
        is_game_over = False
        obs = env.reset()
        step = 0

        while not is_game_over:
            actions = []
            for agent in agents:
                action = agent.forward(obs)
                actions.append(action)

            obs, rewards, is_game_over, _ = env.step(actions)

            for player, agent in enumerate(agents):
                reward = rewards[player]
                agent.backwards(reward, is_game_over, obs)
                # env.render()
                if reward == 1:
                    print('episode:', episode, 'winner:', player, 'steps', step)
            step += 1
        # print()
    foo = 1


if __name__ == '__main__':
    play_game()
