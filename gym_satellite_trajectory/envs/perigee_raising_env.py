import math

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from numpy.random import RandomState
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.attitudes import InertialProvider
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity import NewtonianAttraction
from org.orekit.forces.gravity import ThirdBodyAttraction
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.maneuvers import ConstantThrustManeuver
from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient
from org.orekit.forces.radiation import SolarRadiationPressure
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
from org.orekit.utils import IERSConventions


class PerigeeRaisingEnv(gym.Env):
    def __init__(self, **kwargs):
        super(gym.Env, self).__init__(**kwargs)
        self._ref_time = AbsoluteDate(2022, 6, 16, 0, 0, 0.0, TimeScalesFactory.getUTC())
        self._ref_frame = FramesFactory.getGCRF()
        self._ref_sv = np.array([10000.e3, 0.1, math.pi / 3.0, 4.0 * math.pi / 3.0, 2.0 * math.pi / 3.0, 0.0])
        self._ref_sv_pert = np.array([0.0, 0.0, 0.0, 0.0, 0.0, math.pi])
        self._ref_mass = 100.0  # Kg
        self._ref_sc_frame = FramesFactory.getGCRF()
        self._earth_degree = 4
        self._earth_order = 4
        self._use_perturbations = False

        self._spacecraft_area = 1.0  # m^2
        self._spacecraft_reflection = 2.0  # Perfect reflection
        self._thruster_max_force = 0.1  # N
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
        self.observation_space = Box(low=-1. * box, high=box, dtype=np.float64)
        self.action_space = Box(low=-1., high=1., shape=(3,), dtype=np.float64)

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

        # Earth gravity field
        if self._earth_degree == 0 or not self._use_perturbations:
            point_gravity = NewtonianAttraction(Constants.WGS84_EARTH_MU)
            self._propagator.addForceModel(point_gravity)
        else:
            earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                                     Constants.WGS84_EARTH_FLATTENING,
                                     FramesFactory.getITRF(IERSConventions.IERS_2010, True))
            harmonics_gravity_provider = GravityFieldFactory.getNormalizedProvider(self._earth_degree, self._earth_order)
            self._propagator.addForceModel(
                HolmesFeatherstoneAttractionModel(earth.getBodyFrame(), harmonics_gravity_provider))

        if self._use_perturbations:
            # Sun and Moon attraction
            self._propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getSun()))
            self._propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getMoon()))

            # solar radiation pressure
            self._propagator.addForceModel(
                SolarRadiationPressure(CelestialBodyFactory.getSun(),
                                       earth.getEquatorialRadius(),
                                       IsotropicRadiationSingleCoefficient(self._spacecraft_area,
                                                                           self._spacecraft_reflection)))

        self.prev_hist_sc_state = self.hist_sc_state
        self.hist_sc_state = []

        self._propagator.setInitialState(SpacecraftState(orbit, self._ref_mass))

        rotation = FramesFactory.getEME2000().getTransformTo(self._ref_sc_frame, self._ref_time).getRotation()
        attitude = InertialProvider(rotation)
        self._propagator.setAttitudeProvider(attitude)

        self._current_step = 0
        self.prev_hist_action = self.hist_action
        self.hist_action = []

        state = self._propagate(self._propagator.getInitialState().getDate())
        return state

    def step(self, action):
        assert all(abs(a) <= 1.0 for a in action), "Force in each direction can't be greater than 1"

        self.hist_action.append(action)

        current_time = self.hist_sc_state[-1].getDate()
        self._current_step += 1
        new_time = self.hist_sc_state[0].getDate().shiftedBy(self._time_step * self._current_step)

        # We assume we have 3 pairs of thrusters, each of them can be used independently
        for i in range(3):
            if abs(action[i]) > 0.0:
                direction = Vector3D(list((1.0 if action[i] > 0 else -1.0) if i == j else 0.0 for j in range(3)))
                force = (self._thruster_max_force * abs(action[i])).item()
                manoeuvre = ConstantThrustManeuver(current_time, self._time_step,
                                                   force, self._thruster_isp, direction)
                self._propagator.addForceModel(manoeuvre)

        state = self._propagate(new_time)
        done = self._is_done()
        reward = self._get_reward()
        info = {'is_success': True} if done else {}
        return state, reward, done, info

    def seed(self, seed=None):
        self._random_generator = RandomState(seed)
        return [seed]

    # noinspection PyUnusedLocal
    def render(self, mode=None):
        if mode == 'plot':
            return self._plot(self.hist_sc_state, self.hist_action)
        if mode == 'prev_plot':
            return self._plot(self.prev_hist_sc_state, self.prev_hist_action)
        else:
            print(self.hist_sc_state[-1])

    def close(self):
        self._propagator = None
        self.hist_sc_state = None
        self.hist_action = None
        self._current_step = None
        self._random_generator = None

    def _plot(self, hist_sc_state, hist_action):
        fig, axs = plt.subplots(3, 2, figsize=(15.0, 15.0))
        time = np.array(list(map(lambda sc_state: sc_state.getDate().durationFrom(hist_sc_state[0].getDate()),
                                 hist_sc_state))) / 3600.0  # Convert to hours
        a = np.array(list(map(lambda sc_state: sc_state.getA(), hist_sc_state))) / 1000.0  # Convert to km
        e = np.array(list(map(lambda sc_state: sc_state.getE(), hist_sc_state)))
        mass = np.array(list(map(lambda sc_state: sc_state.getMass(), hist_sc_state)))
        ra = a * (1.0 + e)
        rp = a * (1.0 - e)
        v = np.array(list(map(lambda sc_state: sc_state.getPVCoordinates().getVelocity().toArray(), hist_sc_state)))
        h = np.array(list(map(lambda sc_state: sc_state.getPVCoordinates().getMomentum().toArray(), hist_sc_state)))
        f_mod = np.array(list(map(lambda action: np.linalg.norm(action), hist_action))) * self._thruster_max_force
        angle_f_v = list(map(lambda q:
                             np.degrees(np.arccos(
                                 np.dot(q[0], q[1]) / np.linalg.norm(q[0]) / (np.linalg.norm(q[1]) + 1e-10)
                             )),
                             zip(v, hist_action)))
        hist_action_plane = list(map(lambda q: q[1] - np.dot(q[1], q[0]) * q[0] / (np.linalg.norm(q[0]) ** 2),
                                     zip(h, hist_action)))
        angle_fp_v = list(map(lambda q:
                              np.degrees(np.arccos(
                                  np.dot(q[0], q[1] * [1, 1, 0]) / np.linalg.norm(q[0]) / (
                                          np.linalg.norm(q[1] * [1, 1, 0]) + 1e-10)
                              )),
                              zip(v, hist_action_plane)))
        axs[0, 0].ticklabel_format(axis='y', style='plain', useOffset=ra[0])
        axs[0, 0].set_xlim(time[0], time[-1])
        axs[0, 0].set_ylim(ra[0] - 50.0, ra[0] + 25.0)
        axs[0, 0].grid(True)
        axs[0, 0].set_xlabel("time (h)")
        axs[0, 0].set_ylabel("ra (km)")
        axs[0, 0].plot(time, ra)

        axs[0, 1].ticklabel_format(axis='y', style='plain', useOffset=rp[0])
        axs[0, 1].set_xlim(time[0], time[-1])
        axs[0, 1].set_ylim(rp[0] - 50.0, rp[0] + 25.0)
        axs[0, 1].grid(True)
        axs[0, 1].set_xlabel("time (h)")
        axs[0, 1].set_ylabel("rp (km)")
        axs[0, 1].plot(time, rp)

        axs[1, 0].ticklabel_format(axis='y', style='plain', useOffset=mass[0])
        axs[1, 0].set_xlim(time[0], time[-1])
        axs[1, 0].set_ylim(mass[0] - 0.4, mass[0])
        axs[1, 0].grid(True)
        axs[1, 0].set_xlabel("time (h)")
        axs[1, 0].set_ylabel("mass (kg)")
        axs[1, 0].plot(time, mass)

        axs[1, 1].ticklabel_format(axis='y', style='plain')
        axs[1, 1].set_xlim(time[0], time[-1])
        axs[1, 1].set_ylim(-0.1 * self._thruster_max_force, 2.0 * self._thruster_max_force)
        axs[1, 1].grid(True)
        axs[1, 1].set_xlabel("time (h)")
        axs[1, 1].set_ylabel("|F| (-)")
        axs[1, 1].plot(time[0:-1], f_mod)

        axs[2, 0].ticklabel_format(axis='y', style='plain')
        axs[2, 0].set_xlim(time[0], time[-1])
        axs[2, 0].set_ylim(-1.1 * self._thruster_max_force, 1.1 * self._thruster_max_force)
        axs[2, 0].grid(True)
        axs[2, 0].set_xlabel("time (h)")
        axs[2, 0].set_ylabel("F (-)")
        axs[2, 0].plot(time[0:-1], hist_action)

        axs[2, 1].ticklabel_format(axis='y', style='plain')
        axs[2, 1].set_xlim(time[0], time[-1])
        axs[2, 1].set_ylim(-10.0, 190.0)
        axs[2, 1].grid(True)
        axs[2, 1].set_xlabel("time (h)")
        axs[2, 1].set_ylabel("angle F-v (deg)")
        axs[2, 1].plot(time[0:-1], angle_f_v)
        axs[2, 1].plot(time[0:-1], angle_fp_v)

        return fig

    def _propagate(self, time):
        self.hist_sc_state.append(self._propagator.propagate(time))
        pv = self.hist_sc_state[-1].getPVCoordinates()
        return np.array([self.hist_sc_state[-1].getDate().durationFrom(self.hist_sc_state[0].getDate())] +
                        list(pv.getPosition().toArray()) +
                        list(pv.getVelocity().toArray()) +
                        [self.hist_sc_state[-1].getMass()])

    def _is_done(self):
        return self._current_step >= self._max_steps

    def _get_reward(self):
        # Only give a reward at the end of the episode
        if not self._is_done():
            return 0.0
        ra0, rp0, m0 = self._get_ra_rp_m(self.hist_sc_state[0])
        ra, rp, m = self._get_ra_rp_m(self.hist_sc_state[-1])

        return \
            -1.0 * abs(ra - ra0) + \
            1.0 * (rp - rp0) + \
            4.0e4 * (m - m0)

    @staticmethod
    def _get_ra_rp_m(sc_state):
        a = sc_state.getA()
        e = sc_state.getE()
        ra = a * (1.0 + e)
        rp = a * (1.0 - e)
        m = sc_state.getMass()
        return ra, rp, m

    def add_event_detector(self, detector):
        self._propagator.addEventDetector(detector)
