import logging
logging.basicConfig()
logging.root.setLevel(logging.INFO)

import gym
import gym_satellite_trajectory

env = gym.make('PerigeeRaisingDiscreteOneAxis-v0')
env.reset()
env.seed(42)
for i in range(10):
    env.step(1)

env.render()
env.close()
