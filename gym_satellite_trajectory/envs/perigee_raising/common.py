import gym
from gym import spaces
import math
import numpy as np
from numpy.random import RandomState

from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.attitudes import InertialProvider
from org.orekit.forces.gravity import NewtonianAttraction
from org.orekit.forces.maneuvers import ConstantThrustManeuver
from org.orekit.frames import FramesFactory
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

        min_pos = self._ref_sv[0] * (1.0 - self._ref_sv[1])
        max_pos = self._ref_sv[0] * (1.0 + self._ref_sv[1])
        max_vel = np.sqrt(Constants.WGS84_EARTH_MU * (2.0 / min_pos - 1.0 / self._ref_sv[0]))
        box = np.array([self._time_step * self._max_steps * 1.1,
                        max_pos * 1.1, max_pos * 1.1, max_pos * 1.1,
                        max_vel * 1.1, max_vel * 1.1, max_vel * 1.1,
                        self._ref_mass * 1.1])
        self.internal_observation_space = spaces.Box(low=-1. * box, high=box, dtype=np.float64)
        self.observation_space = self.internal_observation_space
        self.action_space = spaces.Box(low=-1.01, high=1.01, shape=(3,), dtype=np.float64)

        self._propagator = None
        self.hist_sc_state = None
        self.hist_action = None
        self._current_step = None
        self._random_generator = None

        self.close()
        self.seed()
        self.reset()

    def reset(self):
        # noinspection PyArgumentList
        kep = (self._ref_sv + (self._random_generator.rand(6) * 2. - 1.) * self._ref_sv_pert).tolist()
        orbit = KeplerianOrbit(kep[0], kep[1], kep[2], kep[3], kep[4], kep[5],
                               PositionAngle.MEAN, self._ref_frame, self._ref_time, Constants.WGS84_EARTH_MU)

        integrator = DormandPrince853IntegratorBuilder(1.0, 1000., 1.0).buildIntegrator(orbit, OrbitType.CARTESIAN)
        self._propagator = NumericalPropagator(integrator)
        self._propagator.setSlaveMode()
        self._propagator.setOrbitType(OrbitType.CARTESIAN)

        point_gravity = NewtonianAttraction(Constants.WGS84_EARTH_MU)
        self._propagator.addForceModel(point_gravity)

        self.hist_sc_state = [SpacecraftState(orbit, self._ref_mass)]
        self._propagator.setInitialState(self.hist_sc_state[0])

        rotation = FramesFactory.getEME2000().getTransformTo(self._ref_sc_frame, self._ref_time).getRotation()
        attitude = InertialProvider(rotation)
        self._propagator.setAttitudeProvider(attitude)

        self._current_step = 0
        self.hist_action = []

        state = self._propagate(self.hist_sc_state[-1].getDate())
        return state

    def step(self, action):
        self.hist_action.append(action)

        current_time = self.hist_sc_state[-1].getDate()
        self._current_step += 1
        new_time = self.hist_sc_state[0].getDate().shiftedBy(self._time_step * self._current_step)

        # noinspection PyTypeChecker
        action_norm = np.linalg.norm(action)
        if action_norm > 0.0:
            direction = Vector3D((action / action_norm).tolist())
            force = (self._truster_force * action_norm).item()
            manoeuvre = ConstantThrustManeuver(current_time, self._time_step,
                                               force, self._truster_isp, direction)
            self._propagator.addForceModel(manoeuvre)

        state = self._propagate(new_time)
        reward = self._get_reward()
        done = not self.internal_observation_space.contains(state) or self._current_step >= self._max_steps
        return state, reward, done, {}

    def seed(self, seed=None):
        self._random_generator = RandomState(seed)
        return [seed]

    # noinspection PyUnusedLocal
    def render(self, mode=None):
        print(self.hist_sc_state[-1])

    def close(self):
        self._propagator = None
        self.hist_sc_state = None
        self.hist_action = None
        self._current_step = None
        self._random_generator = None

    def _propagate(self, time):
        self.hist_sc_state.append(self._propagator.propagate(time))
        pv = self.hist_sc_state[-1].getPVCoordinates()
        return np.array([self.hist_sc_state[-1].getDate().durationFrom(self.hist_sc_state[0].getDate())] +
                        list(pv.getPosition().toArray()) +
                        list(pv.getVelocity().toArray()) +
                        [self.hist_sc_state[-1].getMass()])

    def _get_reward(self):
        ra0, rp0, m0 = self._get_ra_rp_m(self.hist_sc_state[0])
        ra, rp, m = self._get_ra_rp_m(self.hist_sc_state[-1])

        return -1.0e-5 * abs(ra - ra0) + \
            1.0e-5 * (rp - rp0) + \
            1.0e-1 * (m - m0)

    @staticmethod
    def _get_ra_rp_m(sc_state):
        a = sc_state.getA()
        e = sc_state.getE()
        ra = a * (1.0 + e)
        rp = a * (1.0 - e)
        m = sc_state.getMass()
        return ra, rp, m


class PerigeeRaisingNormEnvBase(PerigeeRaisingEnvBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        box = self.internal_observation_space.high / self.internal_observation_space.high
        self.observation_space = spaces.Box(low=-1. * box, high=box, dtype=np.float64)

    def reset(self):
        state = super().reset()
        return state / self.internal_observation_space.high

    def step(self, action):
        state, reward, done, param = super().step(action)
        return state / self.internal_observation_space.high, reward, done, param
