import test.ignore_deprecation_warnings
import unittest
from gym_satellite_trajectory.envs.perigee_raising_env import PerigeeRaisingEnv
from stable_baselines.common.env_checker import check_env


class TestPerigeeRaisingEnv(unittest.TestCase):
    def test_init(self):
        env = PerigeeRaisingEnv()
        check_env(env)

# TODO: Add tests for the environment physics

    def test_render(self):
        env = PerigeeRaisingEnv()
        env.render()
