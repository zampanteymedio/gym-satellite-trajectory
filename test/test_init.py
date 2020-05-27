import gym
import gym_satellite_trajectory
import unittest


class TestInit(unittest.TestCase):
    def test_create_env_perigeeraising_continuous3d(self):
        env = gym.make("PerigeeRaising-Continuous3D-v0")
        self.assertIsInstance(env, gym.Env, "Loaded environment is not a gym.Env class")

    def test_create_env_perigeeraising_continuous1d(self):
        env = gym.make("PerigeeRaising-Continuous1D-v0")
        self.assertIsInstance(env, gym.Env, "Loaded environment is not a gym.Env class")

    def test_create_env_perigeeraising_discrete3d(self):
        env = gym.make("PerigeeRaising-Discrete3D-v0")
        self.assertIsInstance(env, gym.Env, "Loaded environment is not a gym.Env class")

    def test_create_env_perigeeraising_discrete1d(self):
        env = gym.make("PerigeeRaising-Discrete1D-v0")
        self.assertIsInstance(env, gym.Env, "Loaded environment is not a gym.Env class")
