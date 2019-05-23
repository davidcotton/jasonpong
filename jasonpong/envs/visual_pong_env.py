# .../gym/envs/two_player/pong.py
# Pong environment for Gym
# Thanks fo https://gist.github.com/vinothpandian/4337527 for supplying the non-Gym base


import numpy.random as random
import pygame, sys
from pygame.locals import *
import pygame.surfarray as sarray
import numpy as np
from gym import Env, spaces

pygame.init()

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

WIDTH = 600
HEIGHT = 400
BALL_RADIUS = 5
PAD_WIDTH = 8
PAD_HEIGHT = 200
HALF_PAD_WIDTH = PAD_WIDTH / 2
HALF_PAD_HEIGHT = PAD_HEIGHT / 2


class PongObject:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel

    def accelerate(self, acceleration):
        v = self.vel
        sign_a = np.sign(acceleration)
        if np.sign(v) == sign_a:
            self.vel = 0.8 * (v + acceleration)
        else:
            self.vel = v + acceleration


class VisualPongEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    # canvas declaration
    # window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
    # pygame.display.set_caption('Hello World')

    # define event handlers
    def __init__(self):
        self.action_space = spaces.Box(low=-2, high=2, shape=(2,))
        self.observation_space = spaces.Box(low=0, high=255, shape=(WIDTH, HEIGHT, 3))
        self.viewer = None
        self.canvas = pygame.Surface((WIDTH, HEIGHT))
        self.screen = sarray.array3d(self.canvas)
        self.rng = np.random.RandomState()
        self.paddle1 = PongObject([HALF_PAD_WIDTH - 1, HEIGHT / 2], 0)
        self.paddle2 = PongObject([WIDTH + 1 - HALF_PAD_WIDTH, HEIGHT / 2], 0)
        self.reward = [0.0, 0.0]
        self.is_finished = False
        self.ball = None

    def seed(self, seed=None):
        self.rng.seed(seed)

    def reset(self):
        self.is_finished = False
        self.paddle1 = PongObject([HALF_PAD_WIDTH - 1, HEIGHT / 2], 0)
        self.paddle2 = PongObject([WIDTH + 1 - HALF_PAD_WIDTH, HEIGHT / 2], 0)
        self.reward = [0.0, 0.0]
        horz = self.rng.uniform(2, 4)
        vert = self.rng.uniform(-3, 3)
        if self.rng.randint(2):
            self.ball = PongObject([WIDTH / 2, HEIGHT / 2], [-horz, vert])
        else:
            self.ball = PongObject([WIDTH / 2, HEIGHT / 2], [horz, vert])
        return self.step([0, 0])[0]

    # draw function of canvas
    def step(self, actions):
        self.paddle1.accelerate(actions[0])
        self.paddle2.accelerate(actions[1])

        self.canvas.fill(BLACK)
        pygame.draw.line(self.canvas, WHITE, [WIDTH / 2, 0], [WIDTH / 2, HEIGHT], 1)
        pygame.draw.line(self.canvas, WHITE, [PAD_WIDTH, 0], [PAD_WIDTH, HEIGHT], 1)
        pygame.draw.line(self.canvas, WHITE, [WIDTH - PAD_WIDTH, 0], [WIDTH - PAD_WIDTH, HEIGHT], 1)
        pygame.draw.circle(self.canvas, WHITE, [WIDTH // 2, HEIGHT // 2], 70, 1)

        # update paddle's vertical position, keep paddle on the screen
        self.paddle1.pos[1] = min(max(HALF_PAD_HEIGHT, self.paddle1.vel + self.paddle1.pos[1]), HEIGHT - HALF_PAD_HEIGHT)

        self.paddle2.pos[1] = min(max(HALF_PAD_HEIGHT, self.paddle2.vel + self.paddle2.pos[1]), HEIGHT - HALF_PAD_HEIGHT)

        # draw paddles and ball
        pygame.draw.circle(self.canvas, RED, [*map(int, self.ball.pos)], BALL_RADIUS, 0)
        pygame.draw.polygon(self.canvas, GREEN,
                            [[self.paddle1.pos[0] - HALF_PAD_WIDTH, self.paddle1.pos[1] - HALF_PAD_HEIGHT],
                             [self.paddle1.pos[0] - HALF_PAD_WIDTH, self.paddle1.pos[1] + HALF_PAD_HEIGHT],
                             [self.paddle1.pos[0] + HALF_PAD_WIDTH, self.paddle1.pos[1] + HALF_PAD_HEIGHT],
                             [self.paddle1.pos[0] + HALF_PAD_WIDTH, self.paddle1.pos[1] - HALF_PAD_HEIGHT]], 0)
        pygame.draw.polygon(self.canvas, GREEN,
                            [[self.paddle2.pos[0] - HALF_PAD_WIDTH, self.paddle2.pos[1] - HALF_PAD_HEIGHT],
                             [self.paddle2.pos[0] - HALF_PAD_WIDTH, self.paddle2.pos[1] + HALF_PAD_HEIGHT],
                             [self.paddle2.pos[0] + HALF_PAD_WIDTH, self.paddle2.pos[1] + HALF_PAD_HEIGHT],
                             [self.paddle2.pos[0] + HALF_PAD_WIDTH, self.paddle2.pos[1] - HALF_PAD_HEIGHT]], 0)

        # update ball
        self.ball.pos[0] += int(self.ball.vel[0])
        # ball collision check on top and bottom walls
        self.ball.pos[1] = min(HEIGHT + 1 - BALL_RADIUS, max(self.ball.pos[1] + self.ball.vel[1], BALL_RADIUS))
        if self.ball.pos[1] in (HEIGHT + 1 - BALL_RADIUS, BALL_RADIUS):
            self.ball.vel[1] *= -1

        # self.reward = np.absolute(self.ball.vel[1])
        # ball collision check on gutters or paddles
        if int(self.ball.pos[0]) <= BALL_RADIUS + HALF_PAD_WIDTH:
            if int(self.ball.pos[1]) in range(int(self.paddle1.pos[1] - HALF_PAD_HEIGHT),
                                              int(self.paddle1.pos[1] + HALF_PAD_HEIGHT)):
                self.ball.vel[0] = -self.ball.vel[0]
                self.ball.vel[0] *= 1.1
                self.ball.vel[1] *= 1.1
            elif int(self.ball.pos[0]) <= 0:
                self.is_finished = True
                self.reward = [0.0, 1.0]

        if int(self.ball.pos[0]) >= WIDTH - BALL_RADIUS - HALF_PAD_WIDTH:
            if int(self.ball.pos[1]) in range(int(self.paddle2.pos[1] - HALF_PAD_HEIGHT),
                                              int(self.paddle2.pos[1] + HALF_PAD_HEIGHT)):
                self.ball.vel[0] = -self.ball.vel[0]
                self.ball.vel[0] *= 1.1
                self.ball.vel[1] *= 1.1
            elif int(self.ball.pos[0]) >= WIDTH:
                self.is_finished = True
                self.reward = [1.0, 0.0]
        self.screen = sarray.array3d(self.canvas)

        return self.screen, self.reward, self.is_finished, None

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if mode == 'rgb_array':
            return self.screen
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self.screen)
