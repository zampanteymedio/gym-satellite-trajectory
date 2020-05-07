from gym import spaces
from gym_satellite_trajectory.envs.perigee_raising.common import PerigeeRaisingEnvNormBase


class PerigeeRaisingDiscreteThreeAxesEnv(PerigeeRaisingEnvNormBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Discrete(7)
        self._ACTIONS = {
            0: [0., 0., 0.],
            1: [1., 0., 0.],
            2: [-1., 0., 0.],
            3: [0., 1., 0.],
            4: [0., -1., 0.],
            5: [0., 0., 1.],
            6: [0., 0., -1.]
        }

    def step(self, action):
        # noinspection PyTypeChecker
        return super().step(self._ACTIONS[action])
