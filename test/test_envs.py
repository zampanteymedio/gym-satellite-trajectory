import unittest

import gym
from stable_baselines3.common.env_checker import check_env


class TestEnvs(unittest.TestCase):
    def test_create_env_perigeeraising_continuous3d(self):
        env = gym.make("PerigeeRaising-Continuous3D-v0")
        check_env(env)

    def test_create_env_perigeeraising_continuous3d_perturbed(self):
        env = gym.make("PerigeeRaising-Continuous3D-v0", use_perturbations=True, perturb_action=True)
        check_env(env)

    def test_create_env_perigeeraising_continuous1d(self):
        env = gym.make("PerigeeRaising-Continuous1D-v0")
        check_env(env)

    def test_create_env_perigeeraising_discrete3d(self):
        env = gym.make("PerigeeRaising-Discrete3D-v0")
        check_env(env)

    def test_create_env_perigeeraising_discrete1d(self):
        env = gym.make("PerigeeRaising-Discrete1D-v0")
        check_env(env)
