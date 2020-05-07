from gym import spaces
from gym_satellite_trajectory.envs.perigee_raising.common import PerigeeRaisingNormEnvBase


class PerigeeRaisingContinuousThreeAxesEnv(PerigeeRaisingNormEnvBase):
    def __init__(self, **kwargs):
        self.action_space = spaces.Box(low=-1.01, high=1.01, shape=(1,), dtype=np.float64)
        super().__init__(**kwargs)

    def step(self, action):
        return super().step([0.0, action[0], 0.0])
