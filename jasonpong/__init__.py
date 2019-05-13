import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='JasonPong-v0',
    entry_point='jasonpong.envs:JasonPongEnv',
)
register(
    id='JasonPong2d-v0',
    entry_point='jasonpong.envs:JasonPong2dEnv',
)
register(
    id='JasonPongReversed-v0',
    entry_point='jasonpong.envs:JasonPongReversedEnv',
)
