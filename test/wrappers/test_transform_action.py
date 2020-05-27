import unittest

import gym
import numpy as np

from gym_satellite_trajectory.wrappers.transform_action import TransformAction
from stable_baselines.common.env_checker import check_env


class TestTransformAction(unittest.TestCase):
    def test_init_reversed(self):
        env = self._create_env_cartpole_changed()
        self.assertIsInstance(env, gym.ActionWrapper, "The environment has to inherit from gym.ActionWrapper")
        check_env(env)

    def test_init_continuous(self):
        env = self._create_env_cartpole_continuous()
        self.assertIsInstance(env, gym.ActionWrapper, "The environment has to inherit from gym.ActionWrapper")
        check_env(env)

    def test_action_changed(self):
        env = self._create_env_cartpole_changed()
        self.assertEquals(env.action(0), 1, "The action function for 0 is not being applied correctly")
        self.assertEquals(env.action(1), 0, "The action function for 1 is not being applied correctly")

    def test_action_continuous(self):
        env = self._create_env_cartpole_continuous()
        self.assertEquals(env.action([0.7]), 1, "The action function for 0.7 is not being applied correctly")
        self.assertEquals(env.action([-0.5]), 0, "The action function for -0.5 is not being applied correctly")

    @staticmethod
    def _create_env_cartpole_changed():
        env = gym.make("CartPole-v0")
        return TransformAction(env, env.action_space, lambda a: 1 - a)

    @staticmethod
    def _create_env_cartpole_continuous():
        env = gym.make("CartPole-v0")
        box = gym.spaces.Box(low=-1.01, high=1.01, shape=(1,), dtype=np.float64)
        return TransformAction(env, box, lambda a: 1 if a[0] >= 0. else 0)
