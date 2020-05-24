# Gym envs for Satellite Trajectory Optimization

The [Satellite Trajectory](https://github.com/zampanteymedio/gym-satellite-trajectory) is a mono agent
set of environments related to common trajectory design problems.

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

Assuming that all dependencies are met, you need to install the downloaded package using pip:

```bash
cd gym-satellite-trajectory
pip install -e .
```

If you are working on a clean environment, you need to install all dependencies. You can follow the
directions in the [create_environment.sh](aws/create_environment.sh) file, which describes the necessary steps to install all
dependencies in a clean Amazon Linux EC2 instance.

# Dependencies

This package depends on Orekit wrapper for Python:
https://gitlab.orekit.org/orekit-labs/python-wrapper/-/wikis/installation

# Installation in AWS EC2 instance

Check the aws folder for a script that fully configures an Amazon Linux EC2 instance to install this package
and all its dependencies.
