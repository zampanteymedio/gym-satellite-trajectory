import logging
logging.basicConfig()
logging.root.setLevel(logging.INFO)

import gym
# noinspection PyUnresolvedReferences
import gym_satellite_trajectory


env = gym.make('PerigeeRaisingDiscreteOneAxis-v0')
env.seed(42)
env.reset()

for i in range(10):
    env.step(1)

env.render()
env.close()
