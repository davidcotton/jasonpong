import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='JasonPong-v1',
    entry_point='jasonpong.envs:JasonPongEnv',
)
register(
    id='JasonPong2d-v1',
    entry_point='jasonpong.envs:JasonPong2dEnv',
)
register(
    id='JasonPongMirrored-v1',
    entry_point='jasonpong.envs:JasonPongMirroredEnv',
)
register(
    id='JasonPongMirrored-v2',
    entry_point='jasonpong.envs:JasonPongMirrored2Env',
)
