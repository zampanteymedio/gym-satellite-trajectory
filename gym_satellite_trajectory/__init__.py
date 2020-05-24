import logging
import orekit
import os
import gym
import numpy as np
from gym.envs.registration import register
from java.io import File
from org.orekit.data import DataProvidersManager, ZipJarCrawler

# Logger
logger = logging.getLogger(__name__)

# Import Orekit
# noinspection PyUnresolvedReferences
vm = orekit.initVM()
logger.info('Java version: %s', vm.java_version)

# Load Orekit data
data_provider_manager = DataProvidersManager.getInstance()
datafile = File(os.path.join(os.path.dirname(__file__), '..', 'data', 'orekit-data.zip'))
logger.info('Orekit data file: %s', datafile)
crawler = ZipJarCrawler(datafile)
data_provider_manager.clearProviders()
data_provider_manager.addProvider(crawler)

# Register gym environments
register(
    id='PerigeeRaising-Continuous3D-v0',
    entry_point='gym_satellite_trajectory.envs.perigee_raising_env:PerigeeRaisingEnv',
    kwargs={},
)

register(
    id='PerigeeRaising-Continuous1D-v0',
    entry_point='gym_satellite_trajectory.wrappers.transform_action:TransformAction',
    kwargs={
        'env': gym.make('PerigeeRaising-Continuous3D-v0'),
        'action_space': gym.spaces.Box(low=-1.01, high=1.01, shape=(1,), dtype=np.float64),
        'f': lambda action: [0., action, 0.],
    },
)

pr_d2c_3d = [
    [0., 0., 0.],
    [1., 0., 0.],
    [-1., 0., 0.],
    [0., 1., 0.],
    [0., -1., 0.],
    [0., 0., 1.],
    [0., 0., -1.],
]
register(
    id='PerigeeRaising-Discrete3D-v0',
    entry_point='gym_satellite_trajectory.wrappers.transform_action:TransformAction',
    kwargs={
        'env': gym.make('PerigeeRaising-Continuous3D-v0'),
        'action_space': gym.spaces.Discrete(len(pr_d2c_3d)),
        'f': lambda action: pr_d2c_3d[action],
    },
)

pr_d2c_1d = [
    [0., 0., 0.],
    [0., 1., 0.],
    [0., -1., 0.],
]
register(
    id='PerigeeRaising-Discrete1D-v0',
    entry_point='gym_satellite_trajectory.wrappers.transform_action:TransformAction',
    kwargs={
        'env': gym.make('PerigeeRaising-Continuous3D-v0'),
        'action_space': gym.spaces.Discrete(len(pr_d2c_1d)),
        'f': lambda action: pr_d2c_1d[action],
    },
)
