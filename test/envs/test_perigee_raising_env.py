import unittest

import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env

from gym_satellite_trajectory.envs.perigee_raising_env import PerigeeRaisingEnv


class TestPerigeeRaisingEnv(unittest.TestCase):
    def test_init(self):
        env = PerigeeRaisingEnv()
        check_env(env)

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
