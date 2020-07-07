# Gym envs for Satellite Trajectory Optimization

[![Build Status](https://travis-ci.com/zampanteymedio/gym-satellite-trajectory.svg?branch=master)](https://travis-ci.com/zampanteymedio/gym-satellite-trajectory)
[![codecov](https://codecov.io/gh/zampanteymedio/gym-satellite-trajectory/branch/master/graph/badge.svg)](https://codecov.io/gh/zampanteymedio/gym-satellite-trajectory)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.6&nbsp;|&nbsp;3.7](https://img.shields.io/badge/python-3.6&nbsp;|&nbsp;3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)

![Code scanning - action](https://github.com/zampanteymedio/gym-satellite-trajectory/workflows/Code%20scanning%20-%20action/badge.svg?branch=master)

The [Satellite Trajectory](https://github.com/zampanteymedio/gym-satellite-trajectory) library is a mono agent
set of environments related to common trajectory design problems.

# Environments

The following scenarios are supported:

## Perigee raising

In this scenario, the agent needs to increase the perigee of a satellite without changing the apogee.

* __Continuous - 1 axis:__ the agent needs to select continuously the intensity for 2 thrusters: one in
each Y inertial direction.

* __Continuous - 3 axes:__ the agent needs to select continuously the intensity for the 6 thrusters: one
in each main inertial direction.

* __Discrete - 1 axis:__ the agent needs to select a thruster (or not) out of the 2 available thrusters:
one in each Y inertial direction.

* __Discrete - 3 axes:__ the agent needs to select a thruster (or not) out of the 6 possible thrusters:
one in each main inertial direction.

# General Comments
When using these environments, please take into consideration the folowing points:
- The observations, the actions and the rewards are __NOT normalised__. You might want to do your own
normalisation as your agent needs. Something like:
```python
def get_env():
    env = gym.make('PerigeeRaising-Discrete1D-v0')
    wrapper_observation = gym.wrappers.TransformObservation(env, lambda o: o / env.observation_space.high)
    wrapper_reward = gym.wrappers.TransformReward(wrapper_observation, lambda r: 1.e-5 * r)
    return wrapper_reward
```

# Installation

You can follow the directions in the [.travis.yml](.travis.yml) file, which describes the necessary steps
to install all the dependencies and this package in a clean environment.
