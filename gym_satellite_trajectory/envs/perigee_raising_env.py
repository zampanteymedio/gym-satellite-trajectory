import gym
from gym import spaces
import math
import matplotlib.pyplot as plt
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


class PerigeeRaisingEnv(gym.Env):
    def __init__(self, **kwargs):
        super(gym.Env, self).__init__(**kwargs)
        self._ref_time = AbsoluteDate(2004, 2, 1, 0, 0, 0.0, TimeScalesFactory.getUTC())
        self._ref_frame = FramesFactory.getGCRF()
        self._ref_sv = np.array([10000.e3, 0.1, 0.0, 0.0, 0.0, 0.0])
        self._ref_sv_pert = np.array([0.0, 0.0, 0.0, 0.0, 0.0, math.pi])
        self._ref_mass = 1000.0
        self._ref_sc_frame = FramesFactory.getGCRF()

        self._thruster_force = 1.0  # N
        self._thruster_isp = 4000.0  # s

        self._time_step = 60.0 * 5.0  # 5 minutes
        self._max_steps = 150

        min_pos = self._ref_sv[0] * (1.0 - self._ref_sv[1])
        max_pos = self._ref_sv[0] * (1.0 + self._ref_sv[1])
        max_vel = np.sqrt(Constants.WGS84_EARTH_MU * (2.0 / min_pos - 1.0 / self._ref_sv[0]))
        box = np.array([self._time_step * self._max_steps * 1.1,
                        max_pos * 1.1, max_pos * 1.1, max_pos * 1.1,
                        max_vel * 1.1, max_vel * 1.1, max_vel * 1.1,
                        self._ref_mass * 1.1])
        self.observation_space = spaces.Box(low=-1. * box, high=box, dtype=np.float64)
        self.action_space = spaces.Box(low=-1., high=1., shape=(3,), dtype=np.float64)

        self._propagator = None
        self.hist_sc_state = None
        self.hist_action = None
        self.prev_hist_sc_state = None
        self.prev_hist_action = None
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

        self.prev_hist_sc_state = self.hist_sc_state
        self.hist_sc_state = [SpacecraftState(orbit, self._ref_mass)]
        self._propagator.setInitialState(self.hist_sc_state[0])

        rotation = FramesFactory.getEME2000().getTransformTo(self._ref_sc_frame, self._ref_time).getRotation()
        attitude = InertialProvider(rotation)
        self._propagator.setAttitudeProvider(attitude)

        self._current_step = 0
        self.prev_hist_action = self.hist_action
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
            force = (self._thruster_force * action_norm).item()
            manoeuvre = ConstantThrustManeuver(current_time, self._time_step,
                                               force, self._thruster_isp, direction)
            self._propagator.addForceModel(manoeuvre)

        state = self._propagate(new_time)
        reward = self._get_reward()
        done = not self.observation_space.contains(state) or self._current_step >= self._max_steps
        info = {'is_success': True} if done else {}
        return state, reward, done, info

    def seed(self, seed=None):
        self._random_generator = RandomState(seed)
        return [seed]

    # noinspection PyUnusedLocal
    def render(self, mode=None):
        if mode == 'plot':
            return self._plot(self.hist_sc_state)
        if mode == 'prev_plot':
            return self._plot(self.prev_hist_sc_state)
        else:
            print(self.hist_sc_state[-1])

    def close(self):
        self._propagator = None
        self.hist_sc_state = None
        self.hist_action = None
        self._current_step = None
        self._random_generator = None

    @staticmethod
    def _plot(hist_sc_state):
        fig, axs = plt.subplots(2, 2, figsize=(15.0, 10.0))
        time = np.array(list(map(lambda sc_state: sc_state.getDate().durationFrom(hist_sc_state[0].getDate()),
                                 hist_sc_state))) / 3600.0  # Convert to hours
        a = np.array(list(map(lambda sc_state: sc_state.getA(), hist_sc_state))) / 1000.0  # Convert to km
        e = np.array(list(map(lambda sc_state: sc_state.getE(), hist_sc_state)))
        mass = np.array(list(map(lambda sc_state: sc_state.getMass(), hist_sc_state)))
        ra = a * (1.0 + e)
        rp = a * (1.0 - e)

        axs[0, 0].ticklabel_format(axis='y', style='plain', useOffset=ra[0])
        axs[0, 0].set_xlim(time[0], time[-1])
        axs[0, 0].set_ylim(ra[0]-100.0, ra[0]+100.0)
        axs[0, 0].grid(True)
        axs[0, 0].set_xlabel("time (h)")
        axs[0, 0].set_ylabel("ra (km)")
        axs[0, 0].plot(time, ra)

        axs[0, 1].ticklabel_format(axis='y', style='plain', useOffset=rp[0])
        axs[0, 1].set_xlim(time[0], time[-1])
        axs[0, 1].set_ylim(rp[0]-100.0, rp[0]+100.0)
        axs[0, 1].grid(True)
        axs[0, 1].set_xlabel("time (h)")
        axs[0, 1].set_ylabel("rp (km)")
        axs[0, 1].plot(time, rp)

        axs[1, 0].ticklabel_format(axis='y', style='plain', useOffset=mass[0])
        axs[1, 0].set_xlim(time[0], time[-1])
        axs[1, 0].set_ylim(mass[0]-2.0, mass[0])
        axs[1, 0].grid(True)
        axs[1, 0].set_xlabel("time (h)")
        axs[1, 0].set_ylabel("mass (kg)")
        axs[1, 0].plot(time, mass)

        return fig

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

        return -1.0 * abs(ra - ra0) + \
            1.0 * (rp - rp0) + \
            0.0e+4 * (m - m0) # Increase me if you want mass to be optimised

    @staticmethod
    def _get_ra_rp_m(sc_state):
        a = sc_state.getA()
        e = sc_state.getE()
        ra = a * (1.0 + e)
        rp = a * (1.0 - e)
        m = sc_state.getMass()
        return ra, rp, m
