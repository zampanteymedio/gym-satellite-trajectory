from gym import spaces
from gym_satellite_trajectory.envs.perigee_raising.common import PerigeeRaisingEnvNormBase


class PerigeeRaisingDiscreteOneAxisEnv(PerigeeRaisingEnvNormBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Discrete(3)
        self._ACTIONS = {
            0: [0., 0., 0.],
            1: [0., 1., 0.],
            2: [0., -1., 0.]
        }
