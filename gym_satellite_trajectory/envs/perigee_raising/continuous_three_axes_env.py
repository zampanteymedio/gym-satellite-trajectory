from gym_satellite_trajectory.envs.perigee_raising.common import PerigeeRaisingNormEnvBase


class PerigeeRaisingContinuousThreeAxesEnv(PerigeeRaisingNormEnvBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
