# Gym envs for Satellite Trajectory Optimization

The [Satellite Trajectory](https://github.com/zampanteymedio/gym-satellite-trajectory) is a mono agent
set of environments related to common trajectory design problems.

The following scenarios are supported:

## Perigee raising

In this scenario, the agent needs to increase the perigee of a satellite without changing the apogee.

### Discrete - 1 axis

The agent needs to select a thruster out of the 2 available thrusters: one in each Y inertial direction.

### Discrete - 3 axis

The agent needs to select a thruster out of the 6 possible thrusters: one in each main inertial direction.

### Continuous

The agent needs to select continuously the intensity for the 6 thrusters, one in each main inertial direction.

# Installation

```bash
cd gym-satellite-trajectory
pip install -e .
```

# Dependencies

This package depends on Orekit wrapper for Python:
https://gitlab.orekit.org/orekit-labs/python-wrapper/-/wikis/installation

# Installation in AWS EC2 instance

Check the aws folder for a script that fully configures an Amazon Linux EC2 instance to use this package.
