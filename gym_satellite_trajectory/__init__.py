import logging
import orekit
import os
from gym.envs.registration import register
from java.io import File
from org.orekit.data import DataProvidersManager, ZipJarCrawler

# Logger
logger = logging.getLogger(__name__)

# Import Orekit
vm = orekit.initVM()
logger.info('Java version: %s', vm.java_version)

# Load sample data
data_provider_manager = DataProvidersManager.getInstance()
datafile = File(os.path.join(os.path.dirname(__file__), '..', 'data', 'orekit-data.zip'))
logger.info('Orekit data file: %s', datafile)
crawler = ZipJarCrawler(datafile)
data_provider_manager.clearProviders()
data_provider_manager.addProvider(crawler)

# Register gym environments
register(
    id='PerigeeRaisingDiscreteOneAxis-v0',
    entry_point='gym_satellite_trajectory.envs.perigee_raising:PerigeeRaisingDiscreteOneAxisEnv',
    timestep_limit=150,
)

# Register gym environments
register(
    id='PerigeeRaisingDiscreteThreeAxes-v0',
    entry_point='gym_satellite_trajectory.envs.perigee_raising:PerigeeRaisingDiscreteThreeAxesEnv',
    timestep_limit=150,
)
