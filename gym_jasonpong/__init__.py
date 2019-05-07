__version__ = '0.1.0'

import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='JasonPong-v0',
    entry_point='gym_jasonpong.envs:JasonPongEnv',
)
