from abc import ABC, abstractmethod
from collections import deque, namedtuple
import timeit

import gym
import numpy as np

import jasonpong
from q_learner import QLearner

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


class QTableAgent(Agent):
    def __init__(self, player, env) -> None:
        super().__init__(player, env)
        self.gamma = 0.999
        self.learning_rate = 0.001
        self.epsilon = 0.001
        self.num_actions = env.action_space.n
        obs_space = env.observation_space
        self.board_width = int(obs_space.high[2])
        self.board_height = int(obs_space.high[3])
        board_size = self.board_width * self.board_height
        self.q = np.zeros([self.board_width, board_size, self.num_actions], dtype=np.float16)

    def forward(self, state) -> int:
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.num_actions - 1)
        else:
            paddle_position = int(state[-1, self.player])
            ball_x = int(state[-1, 2])
            ball_y = int(state[-1, 3])
            ball_pos = self.encode_position(ball_x, ball_y)
            q = self.q[paddle_position, ball_pos]

            # randomly break ties
            best_actions = np.argwhere(q == np.amax(q)).flatten()
            action = np.random.choice(best_actions)
        return action

    def backwards(self, transition: Transition):
        paddle_pos = int(transition.state[0, self.player])
        ball_x = int(transition.state[0, 2])
        ball_y = int(transition.state[0, 3])
        ball_pos = self.encode_position(ball_x, ball_y)

        next_paddle_pos = int(transition.state[-1, self.player])
        next_ball_x = int(transition.state[-1, 2])
        next_ball_y = int(transition.state[-1, 3])
        next_ball_pos = self.encode_position(next_ball_x, next_ball_y)

        action = transition.action

        next_q = self.q[next_paddle_pos, next_ball_pos]
        target_q = transition.reward + self.gamma * np.max(next_q)
        prev_q = self.q[paddle_pos, ball_pos, action]
        self.q[paddle_pos, ball_pos, action] += self.learning_rate * (target_q - prev_q)
        foo = 1

    def encode_position(self, x, y) -> int:
        return self.board_width * y + x


class EpisodicQTableAgent(QTableAgent):
    def __init__(self, player, env) -> None:
        super().__init__(player, env)
        # self.epsilon = 0.001
        self.epsilon = 0.0
        self.alpha = 0.5
        self.gamma = 0.99
        self.q = np.zeros([self.board_width, self.board_width, self.board_width, self.board_height, 2, 2, self.num_actions], dtype=np.float16)
        self.visits = np.zeros([self.board_width, self.board_width, self.board_width, self.board_height, 2, 2, 1], dtype=np.uint8)
        self.transitions = []

    def forward(self, state) -> int:
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.num_actions - 1)
        else:
            paddle_x, paddle_y, ball_x, ball_y, vel_x, vel_y = state[-1].astype(int)
            if vel_x < 0: vel_x = 0
            if vel_y < 0: vel_y = 0
            q = self.q[paddle_x, paddle_y, ball_x, ball_y, vel_x, vel_y]

            # randomly break ties
            best_actions = np.argwhere(q == np.amax(q)).flatten()
            action = np.random.choice(best_actions)
        return action

    def backwards(self, transition: Transition):
        self.transitions.append(transition)
        if transition.game_over:
            prev_reward = transition.reward
            for i in range(len(self.transitions) - 1, -1, -1):
                state = self.transitions[i].state[-1].astype(int)
                paddle0, paddle1, ball_x, ball_y, vel_x, vel_y = state
                if vel_x < 0: vel_x = 0
                if vel_y < 0: vel_y = 0
                action = self.transitions[i].action
                prev_q = self.q[paddle0, paddle1, ball_x, ball_y, vel_x, vel_y, action]
                visits = self.visits[paddle0, paddle1, ball_x, ball_y, vel_x, vel_y, 0]
                if i == len(self.transitions) - 1:
                    new_q = prev_reward + visits * prev_q
                    visits += 1
                    new_q /= visits
                else:
                    # visits = 1
                    new_q = (1 - self.alpha) * prev_q + self.alpha * (self.gamma * prev_reward)
                self.q[paddle0, paddle1, ball_x, ball_y, vel_x, vel_y, action] = new_q
                self.visits[paddle0, paddle1, ball_x, ball_y, vel_x, vel_y, 0] = visits
                prev_reward = self.q[paddle0, paddle1, ball_x, ball_y, vel_x, vel_y, action]
            self.transitions.clear()


class QLearnerAgent(Agent):
    def __init__(self, player, env) -> None:
        super().__init__(player, env)
        self.learning_mode = True
        # exploration_rate = 0.001
        exploration_rate = 0.0
        # Use Jason's refined backprop q learning code
        self.ql = QLearner(exploration_rate, 3)
        self.state_seq = []
        self.action_seq = []

    def forward(self, state) -> int:
        # Internally the q learner agent uses a dictionary which requires a format
        # that can be converted to a tuple as a key
        # Thus we convert the numpy array to a list -> tuple
        new_state = state.tolist()

        # For faster convergence you can have the new state just refined to the bat and ball position
        action = self.ql.select(new_state, self.learning_mode)

        # Push state and action data for training
        if self.learning_mode:
            self.state_seq.append(new_state)
            self.action_seq.append(action)

        return action

    def backwards(self, transition: Transition):
        # If its game over push the state action data into the q learner bot
        if self.state_seq and transition.reward:
            self.ql.update(self.state_seq, self.action_seq, transition.reward)
            self.reset()

    def reset(self):
        # Start each game with no information
        self.state_seq.clear()
        self.action_seq.clear()

    # Change between learning and testing modes
    def set_learn_mode(self, learning_mode):
        self.learning_mode = learning_mode


RENDER_STEP = False
RENDER_EPISODE = True


def play_game():
    env = gym.make('JasonPong-v0')
    # env = gym.make('JasonPong2d-v0')
    # agents = [QTableAgent(i, env) for i in range(2)]
    # agents = [RandomAgent(i, env) for i in range(2)]
    # agents = [QTableAgent(0, env), RandomAgent(1, env)]
    # agents = [RandomAgent(0, env), QTableAgent(1, env)]
    # agents = [QLearnerAgent(i, env) for i in range(2)]
    agents = [EpisodicQTableAgent(i, env) for i in range(2)]
    obs_buffer = deque(maxlen=2)
    results = {
        'winners': [],
        'steps': []
    }
    start_time = timeit.default_timer()
    update_time = timeit.default_timer()
    update_times = deque(maxlen=100)

    for episode in range(int(1e3)):
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
                action = agent.forward(state)
                next_obs, reward, is_game_over, _ = env.step(action)
                transitions[player] = Transition(state, action, reward, is_game_over)

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
                update_times.append(step / (timeit.default_timer() - update_time))
                update_time = timeit.default_timer()
                if RENDER_EPISODE:
                    print(f'Episode: {episode:>4,}   winner: {env.winner}   steps: {step:<5,} ',
                          f'steps/second: {np.mean(update_times):,.0f}')
                # if env.winner == 2:
                #     raise ValueError('Solved')

        if RENDER_STEP:
            print()

    steps = np.array(results['steps'])
    winners = np.array(results['winners'])
    print()
    print('run time:', timeit.default_timer() - start_time)
    print('steps mean:', np.mean(steps))
    print('steps max:', np.max(steps))
    print('winners:', np.bincount(winners))


if __name__ == '__main__':
    play_game()
