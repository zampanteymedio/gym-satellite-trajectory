import gym
from gym import spaces
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState

from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.attitudes import LofOffset
from org.orekit.attitudes import InertialProvider
from org.orekit.propagation.events import DateDetector
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity import NewtonianAttraction
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.maneuvers import ConstantThrustManeuver
from org.orekit.frames import FramesFactory
from org.orekit.frames import LOFType
from org.orekit.orbits import KeplerianOrbit
from org.orekit.orbits import OrbitType
from org.orekit.orbits import PositionAngle
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.conversion import DormandPrince853IntegratorBuilder
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.time import AbsoluteDate
from org.orekit.time import TimeScalesFactory
from org.orekit.utils import Constants


class PerigeeRaisingEnvBase(gym.Env):
    def __init__(self, **kwargs):
        self._ref_time = AbsoluteDate(2004, 2, 1, 0, 0, 0.0, TimeScalesFactory.getUTC())
        self._ref_frame = FramesFactory.getGCRF()
        self._ref_sv = np.array([10000.e3, 0.1, 0.0, 0.0, 0.0, 0.0])
        self._ref_sv_pert = np.array([0.0, 0.0, 0.0, 0.0, 0.0, math.pi])
        self._ref_mass = 1000.0
        self._ref_sc_frame = FramesFactory.getGCRF()

        self._truster_force = 1.0  # N
        self._truster_isp = 4000.0  # s

        self._time_step = 60.0 * 5.0  # 5 minutes
        self._max_steps = 150
        self._end_time = self._ref_time.shiftedBy(self._time_step * self._max_steps)

        box = np.array([self._time_step * self._max_steps * 1.1,
                        15000.e3, 15000.e3, 15000.e3,
                        15.e3, 15.e3, 15.e3,
                        self._ref_mass * 1.1])
        self.internal_observation_space = spaces.Box(low=-1. * box, high=box, dtype=np.float64)
        self.observation_space = self.internal_observation_space
        self.action_space = spaces.Box(low=-1.01, high=1.01, shape=(3,), dtype=np.float64)

        self._propagator = None
        self.current_sc_state = None
        self.current_step = 0

        self._random_state = RandomState()

        self.seed()
        self.reset()

    def reset(self):
        self.seed()
        kep = (self._ref_sv + (self._random_state.rand(6) * 2. - 1.) * self._ref_sv_pert).tolist()
        orbit = KeplerianOrbit(kep[0], kep[1], kep[2], kep[3], kep[4], kep[5],
                               PositionAngle.MEAN, self._ref_frame, self._ref_time, Constants.WGS84_EARTH_MU)

        integrator = DormandPrince853IntegratorBuilder(1.0, 1000., 1.0).buildIntegrator(orbit, OrbitType.CARTESIAN)
        self._propagator = NumericalPropagator(integrator)
        self._propagator.setSlaveMode()
        self._propagator.setOrbitType(OrbitType.CARTESIAN)

        point_gravity = NewtonianAttraction(Constants.WGS84_EARTH_MU)
        self._propagator.addForceModel(point_gravity)

        #         Commented to make it faster...to be added once a more precise model is needed        
        #         provider = GravityFieldFactory.getNormalizedProvider(8, 8)
        #         holmesFeatherstone = HolmesFeatherstoneAttractionModel(self._fixed_frame, provider)
        #         self._propagator.addForceModel(holmesFeatherstone)

        self._propagator.setInitialState(SpacecraftState(orbit, self._ref_mass))

        attitude = InertialProvider(
            FramesFactory.getEME2000().getTransformTo(self._ref_sc_frame, self._ref_time).getRotation())
        self._propagator.setAttitudeProvider(attitude)

        self.current_step = 0

        state = self._propagate(self._propagator.getInitialState().getDate())
        return state

    def step(self, action):
        self.action = action
        action_norm = np.linalg.norm(action)
        if action_norm > 0.:
            direction = Vector3D((action / action_norm).tolist())
            force = (self._truster_force * action_norm).item()
            manoeuvre = ConstantThrustManeuver(self._propagator.getInitialState().getDate(), self._time_step,
                                               force, self._truster_isp, direction)
            self._propagator.addForceModel(manoeuvre)

        self.current_step = self.current_step + 1
		
        state = self._propagate(self._propagator.getInitialState().getDate().shiftedBy(self._time_step))
        reward = self._get_reward()
        done = not self.internal_observation_space.contains(self.state) or self.current_step >= self._max_steps
        return state, reward, done, {}

    def seed(self, seed=None):
        self._random_state = RandomState(seed)
        return [seed]

    # noinspection PyUnusedLocal
    def render(self, mode=None):
	    pass

    def close(self):
        pass

    def _propagate(self, time):
        self.current_sc_state = self._propagator.propagate(time)
        pv = self.current_sc_state.getPVCoordinates()
        return np.array([self._end_time.durationFrom(self.current_sc_state.getDate())] +
                        list(pv.getPosition().toArray()) +
                        list(pv.getVelocity().toArray()) +
                        [self.current_sc_state.getMass()])

    def _get_reward(self):
        a = self.current_sc_state.getA()
        e = self.current_sc_state.getE()
        ra = a * (1.0 + e)
        rp = a * (1.0 - e)
        m = self.current_sc_state.getMass()

        initial_sc_state = self._propagator.getInitialState()
        a0 = initial_sc_state.getA()
        e0 = initial_sc_state.getE()
        ra0 = a0 * (1.0 + e0)
        rp0 = a0 * (1.0 - e0)
        m0 = initial_sc_state.getMass()

        return -1.0e-5 * abs(ra - ra0) + \
            1.0e-5 * (rp - rp0) + \
            1.0e-1 * (m - m0)


class PerigeeRaisingEnvNormBase(PerigeeRaisingEnvBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        box = self.internal_observation_space.high / self.internal_observation_space.high
        self.observation_space = spaces.Box(low=-1. * box, high=box, dtype=np.float64)

    def step(self, action):
        state, reward, done, param = super().step(action)
        return state / self.internal_observation_space.high, reward, done, param
