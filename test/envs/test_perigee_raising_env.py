import unittest

import matplotlib.pyplot as plt
from stable_baselines3.a2c import A2C
from stable_baselines3.a2c.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from gym_satellite_trajectory.envs.perigee_raising_env import PerigeeRaisingEnv


class TestPerigeeRaisingEnv(unittest.TestCase):
    def test_init(self):
        env = PerigeeRaisingEnv()
        check_env(env)

    def test_run_full_episode(self):
        env = PerigeeRaisingEnv()
        agent = A2C(policy=MlpPolicy, env=env)
        evaluate_policy(agent, env, n_eval_episodes=1)

    # TODO: Add tests for the environment physics

    def test_render(self):
        env = PerigeeRaisingEnv()
        env.reset()
        assert env.render() is None

    def test_render_plot(self):
        env = PerigeeRaisingEnv()
        env.reset()
        assert type(env.render('plot')) == plt.Figure

    def test_render_prev_plot(self):
        env = PerigeeRaisingEnv()
        env.reset()
        assert type(env.render('prev_plot')) == plt.Figure
