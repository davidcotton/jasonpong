from abc import ABC, abstractmethod
from collections import deque, namedtuple

import gym
import numpy as np

import jasonpong

from Q_Learner import Q_Learner

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'game_over'])


class Agent(ABC):
    def __init__(self, player, env) -> None:
        super().__init__()
        self.player = player
        self.env = env

    @abstractmethod
    def forward(self, state) -> int:
        pass

    def backwards(self, transition: Transition):
        pass


class RandomAgent(Agent):
    def forward(self, state) -> int:
        action = np.random.choice([0, 1, 2])
        return action

# Modified to use Jason's q backprop q learning AI
class QTableAgent(Agent):
    def __init__(self, player, env) -> None:
        super().__init__(player, env)
        self.learning_mode= True
        exploration_rate = 0.05
        # Use Jason's refined backprop q learning code
        self.ql = Q_Learner(exploration_rate)
        # To make the agent more modular I'm keeping track of
        # all the revelant states,actions for training
        self.reset()

    # Change between learning and testing modes
    def set_learn_mode(self, learning_mode):
        self.learning_mode = learning_mode

    def reset(self):
        # Start each game with no information
        self.state_seq = []
        self.action_seq = []
        self.all_actions = []

    def forward(self, state) -> int:
        paddle_pos = int(state[0])
        ball_x = int(state[2])
        new_state = (paddle_pos, ball_x)

        action = self.ql.select(new_state, 3, self.learning_mode, [0,1,2])

        # Push state and action data for training
        if self.learning_mode:
            self.state_seq.append(new_state)
            self.action_seq.append(action)
            self.all_actions.append([0,1,2])

        return action

    def backwards(self, transition: Transition):
        # If its game over push the state action data into the q learner bot
        if self.state_seq and transition.reward:
            self.ql.update(self.state_seq, self.action_seq, self.all_actions, transition.reward)
            self.reset()

    def encode_position(self, x, y) -> int:
        return self.board_width * y + x


RENDER_STEP = False
RENDER_EPISODE = False


def play_game():
    env = gym.make('JasonPong-v0')
    # env = gym.make('JasonPong2d-v0')
    # agents = [QTableAgent(i, env) for i in range(2)]
    # agents = [RandomAgent(i, env) for i in range(2)]
    agents = [QTableAgent(0, env), RandomAgent(1, env)]
    # agents = [RandomAgent(0, env), QTableAgent(1, env)]

    for learning_mode in (True,False):
        obs_buffer = deque(maxlen=2)
        results = {
            'winners': [],
            'steps': []
        }

        agents[0].set_learn_mode(learning_mode)
        if learning_mode:
            print ("Running in Training mode")
        else:
            print ("Running in Testing mode")

        for episode in range(int(1e4)):
            step = 0
            obs = env.reset()
            obs_buffer.extend([obs, obs])

            if RENDER_STEP:
                print(f'Episode {episode:,}')
                env.render()
            is_game_over = False
            while not is_game_over:
                state = np.array(obs_buffer)
                transitions = [None, None]
                for player, agent in enumerate(agents):
                    action = agent.forward(state[1])
                    next_obs, reward, is_game_over, _ = env.step(action)
                    transitions[player] = Transition(state[1], action, reward, is_game_over)

                obs_buffer.append(next_obs)

                if is_game_over:
                    transitions[0] = Transition(transitions[0].state, transitions[0].action, transitions[1].reward * -1,
                                                is_game_over)

                for player, agent in enumerate(agents):
                    agent.backwards(transitions[player])

                step += 1
                if RENDER_STEP:
                    env.render()
                if is_game_over:
                    # calculate metrics
                    results['winners'].append(env.winner)
                    results['steps'].append(step)
                    if RENDER_EPISODE:
                        print(f'Episode: {episode:,}    winner:', env.winner, '   steps', step)

            if RENDER_STEP:
                print()

        steps = np.array(results['steps'])
        winners = np.array(results['winners'])
        print()
        print('steps mean:', np.mean(steps))
        print('steps max:', np.max(steps))
        print('winners:', np.bincount(winners))


if __name__ == '__main__':
    play_game()
