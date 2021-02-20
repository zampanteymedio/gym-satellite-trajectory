import logging
import os

import numpy as np
import orekit
from gym.envs.registration import register
from gym.spaces import Box, Discrete
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
datafile = File(os.path.join(os.path.dirname(__file__), 'data', 'orekit-data.zip'))
logger.info('Orekit data file: %s', datafile)
crawler = ZipJarCrawler(datafile)
data_provider_manager.clearProviders()
data_provider_manager.addProvider(crawler)


# Register gym environments
def register_environments():
    register(
        id='PerigeeRaising-Continuous3D-v0',
        entry_point='gym_satellite_trajectory.envs.perigee_raising_env:PerigeeRaisingEnv',
        kwargs={},
    )

    register(
        id='PerigeeRaising-Continuous1D-v0',
        entry_point='gym_satellite_trajectory.wrappers.transform_action:TransformAction',
        kwargs={
            'env': 'PerigeeRaising-Continuous3D-v0',
            'action_space': Box(low=-1., high=1., shape=(1,), dtype=np.float64),
            'f': lambda action: np.array([0., action[0], 0.]),
        },
    )

    pr_d2c_3d = [
        np.array([0., 0., 0.]),
        np.array([1., 0., 0.]),
        np.array([-1., 0., 0.]),
        np.array([0., 1., 0.]),
        np.array([0., -1., 0.]),
        np.array([0., 0., 1.]),
        np.array([0., 0., -1.]),
    ]
    register(
        id='PerigeeRaising-Discrete3D-v0',
        entry_point='gym_satellite_trajectory.wrappers.transform_action:TransformAction',
        kwargs={
            'env': 'PerigeeRaising-Continuous3D-v0',
            'action_space': Discrete(len(pr_d2c_3d)),
            'f': lambda action: pr_d2c_3d[action],
        },
    )

    pr_d2c_1d = [
        np.array([0., 0., 0.]),
        np.array([0., 1., 0.]),
        np.array([0., -1., 0.]),
    ]
    register(
        id='PerigeeRaising-Discrete1D-v0',
        entry_point='gym_satellite_trajectory.wrappers.transform_action:TransformAction',
        kwargs={
            'env': 'PerigeeRaising-Continuous3D-v0',
            'action_space': Discrete(len(pr_d2c_1d)),
            'f': lambda action: pr_d2c_1d[action],
        },
    )
