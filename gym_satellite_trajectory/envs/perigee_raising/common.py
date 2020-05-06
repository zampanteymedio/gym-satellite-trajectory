import gym
from gym import spaces
from gym.utils import seeding
import math
import matplotlib.pyplot as plt
import numpy as np

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
    def __init__(self):
        self._frame = FramesFactory.getGCRF()

        self._initial_time = AbsoluteDate(2004, 2, 1, 0, 0, 0.0, TimeScalesFactory.getUTC())
        self._initial_sv = np.array([10000.e3, 0.1, 0.0, 0.0, 0.0, 0.0])
        self._initial_pert = np.array([0.0, 0.0, 0.0, 0.0, 0.0, math.pi])
        self._initial_mass = 1000.0
        self._time_step = 60.0 * 5.0  # 5 minutes
        self._max_steps = 150
        self._end_time = self._initial_time.shiftedBy(self._time_step * self._max_steps)
        self._force = 1.0  # N
        self._isp = 4000.0  # s

        self._propagator = None

        box = np.array([self._time_step * self._max_steps * 1.1,
                        20000.e3, 20000.e3, 20000.e3,
                        20.e3, 20.e3, 20.e3,
                        self._initial_mass * 1.1])
        self.internal_observation_space = spaces.Box(low=-1. * box, high=box, dtype=np.float64)
        self.action_space = spaces.Box(low=-1.01, high=1.01, shape=(3,), dtype=np.float64)

        self._plot_fig = None
        self._initial_sc_state = None
        self.state = None
        self.action = None
        self.reward = None
        self.done = None
        self.steps = 0
        self._plot_ax1 = None
        self._plot_ax2 = None
        self._plot_ax3 = None
        self._plot_ax4 = None
        self._plot_ax5 = None
        self._plot_ax6 = None
        self._plot_line1 = None
        self._plot_line2 = None
        self._plot_line3 = None
        self._plot_line4 = None
        self._plot_line5 = None
        self._plot_line6 = None

        self.seed()
        self.reset()

    def reset(self):
        kep = (self._initial_sv + (np.random.rand(6) * 2. - 1.) * self._initial_pert).tolist()
        orbit = KeplerianOrbit(kep[0], kep[1], kep[2], kep[3], kep[4], kep[5],
                               PositionAngle.MEAN, self._frame, self._initial_time, Constants.WGS84_EARTH_MU)

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

        self._initial_sc_state = SpacecraftState(orbit, self._initial_mass)
        self._propagator.setInitialState(self._initial_sc_state)

        attitude = InertialProvider(
            FramesFactory.getEME2000().getTransformTo(self._frame, self._initial_time).getRotation())
        self._propagator.setAttitudeProvider(attitude)

        self.state = self._propagate(self._propagator.getInitialState().getDate())
        self.action = None
        self.reward = None
        self.done = None
        self.steps = 0

        self._plot_line1 = None
        self._plot_line2 = None

        return self.state

    def step(self, action):
        self.action = action
        action_norm = np.linalg.norm(self.action)
        if action_norm > 0.:
            direction = Vector3D((self.action / action_norm).tolist())
            force = (self._force * action_norm).item()
            manoeuvre = ConstantThrustManeuver(self._propagator.getInitialState().getDate(), self._time_step,
                                               force, self._isp, direction)
            self._propagator.addForceModel(manoeuvre)

        self.steps = self.steps + 1
        self.state = self._propagate(self._propagator.getInitialState().getDate().shiftedBy(self._time_step))
        self.done = not self.internal_observation_space.contains(self.state) or self.steps >= self._max_steps
        self.reward = self._get_reward()

        return self.state, self.reward, self.done, {}

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]

    # noinspection PyUnusedLocal
    def render(self, mode=None):
        if self._plot_fig is None:
            plt.ion()
            self._plot_fig = plt.figure(figsize=(12, 24), dpi=80, facecolor='w', edgecolor='k')

            self._plot_ax1 = self._plot_fig.add_subplot(6, 1, 1)
            plt.title('pericentre radius')
            plt.ylabel('rp (km)')
            plt.xlabel('step')
            self._plot_ax1.set_xlim(0.0, self._max_steps)
            self._plot_ax1.set_ylim(8950.0, 9200.0)

            self._plot_ax2 = self._plot_fig.add_subplot(6, 1, 2)
            plt.title('apocentre radius')
            plt.ylabel('ra (km)')
            plt.xlabel('step')
            self._plot_ax2.set_xlim(0.0, self._max_steps)
            self._plot_ax2.set_ylim(10900.0, 11100.0)

            self._plot_ax3 = self._plot_fig.add_subplot(6, 1, 3)
            plt.title('thrust')
            plt.ylabel('norm')
            plt.xlabel('step')
            self._plot_ax3.set_xlim(0.0, self._max_steps)
            self._plot_ax3.set_ylim(-0.5, 1.5)

            self._plot_ax4 = self._plot_fig.add_subplot(6, 1, 4)
            plt.title('thrust')
            plt.ylabel('direction X-Y')
            plt.xlabel('step')
            self._plot_ax4.set_xlim(0.0, self._max_steps)
            self._plot_ax4.set_ylim(-4, 4)

            self._plot_ax5 = self._plot_fig.add_subplot(6, 1, 5)
            plt.title('thrust')
            plt.ylabel('direction XY-Z')
            plt.xlabel('step')
            self._plot_ax5.set_xlim(0.0, self._max_steps)
            self._plot_ax5.set_ylim(-2, 2)

            self._plot_ax6 = self._plot_fig.add_subplot(6, 1, 6)
            plt.title('mass')
            plt.ylabel('mass (kg)')
            plt.xlabel('step')
            self._plot_ax6.set_xlim(0.0, self._max_steps)
            self._plot_ax6.set_ylim(998, 1000.1)

        if self._plot_line1 is None:
            self._plot_line1, = self._plot_ax1.plot([], [])
            self._plot_line2, = self._plot_ax2.plot([], [])
            self._plot_line3, = self._plot_ax3.plot([], [])
            self._plot_line4, = self._plot_ax4.plot([], [])
            self._plot_line5, = self._plot_ax5.plot([], [])
            self._plot_line6, = self._plot_ax6.plot([], [])

        x = self.sc_state.getDate().durationFrom(self._initial_time) / self._time_step
        y1 = self.sc_state.getA() * (1. - self.sc_state.getE()) / 1000.
        y2 = self.sc_state.getA() * (1. + self.sc_state.getE()) / 1000.
        y3 = np.linalg.norm(self.action) if self.action is not None else None
        y4 = np.arctan2(self.action[1], self.action[0]) if self.action is not None else None
        y5 = np.arctan2(self.action[2], np.linalg.norm(self.action[0:1])) if self.action is not None else None
        y6 = self.sc_state.getMass()

        self._plot_line1.set_xdata(np.append(self._plot_line1.get_xdata(), [x]))
        self._plot_line1.set_ydata(np.append(self._plot_line1.get_ydata(), [y1]))

        self._plot_line2.set_xdata(np.append(self._plot_line2.get_xdata(), [x]))
        self._plot_line2.set_ydata(np.append(self._plot_line2.get_ydata(), [y2]))

        self._plot_line3.set_xdata(np.append(self._plot_line3.get_xdata(), [x]))
        self._plot_line3.set_ydata(np.append(self._plot_line3.get_ydata(), [y3]))

        self._plot_line4.set_xdata(np.append(self._plot_line4.get_xdata(), [x]))
        self._plot_line4.set_ydata(np.append(self._plot_line4.get_ydata(), [y4]))

        self._plot_line5.set_xdata(np.append(self._plot_line5.get_xdata(), [x]))
        self._plot_line5.set_ydata(np.append(self._plot_line5.get_ydata(), [y5]))

        self._plot_line6.set_xdata(np.append(self._plot_line6.get_xdata(), [x]))
        self._plot_line6.set_ydata(np.append(self._plot_line6.get_ydata(), [y6]))

        self._plot_fig.canvas.draw()

    def close(self):
        self._plot_fig = None

    def _propagate(self, time):
        self.sc_state = self._propagator.propagate(time)
        pv = self.sc_state.getPVCoordinates()
        return np.array([self._end_time.durationFrom(self.sc_state.getDate())] +
                        list(pv.getPosition().toArray()) +
                        list(pv.getVelocity().toArray()) +
                        [self.sc_state.getMass()])

    def _get_reward(self):
        # Commented out because rewards only at the end are very difficult to converge
        #        if not self.done:
        #                return 0.
        a = self.sc_state.getA()
        e = self.sc_state.getE()
        ra = a * (1.0 + e)
        rp = a * (1.0 - e)
        m = self.sc_state.getMass()

        a0 = self._initial_sc_state.getA()
        e0 = self._initial_sc_state.getE()
        ra0 = a0 * (1.0 + e0)
        rp0 = a0 * (1.0 - e0)
        m0 = self._initial_sc_state.getMass()

        return -1.0e-5 * abs(ra - ra0) + \
            1.0e-5 * (rp - rp0) + \
            1.0e-1 * (m - m0)


class PerigeeRaisingEnvNormBase(PerigeeRaisingEnvBase):
    def __init__(self):
        super().__init__()
        box = self.internal_observation_space.high / self.internal_observation_space.high
        self.observation_space = spaces.Box(low=-1. * box, high=box, dtype=np.float64)

    def step(self, action):
        state, reward, done, param = super().step(action)
        return state / self.internal_observation_space.high, reward, done, param
