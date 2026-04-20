"""
transit_agent.py — A train that traverses a multi-segment transit route.

Port of: +OceanMath/+Transit/TransitAgent.m (395 lines)

Uses the State Pattern to manage behavior across Running, Dwelling, and
Arrived states. Bridges CurveSimulator (single-spline physics) with
TransitNetwork (multi-segment graph). Chains simulators across consecutive
splines, with configurable dwell time at each intermediate station.

Includes trapezoidal acceleration profiles: accel → cruise → brake per segment.

Usage:
    agent = TransitAgent(net, 'NSL', speed=3.0, dwell_time=1.0)
    while not agent.is_arrived():
        agent.step(dt=0.01, real_dt=0.01)
        pos = agent.get_position()
"""

from __future__ import annotations
from math import sqrt
import numpy as np

from ocean.topology.transit_network import TransitNetwork
from ocean.geometry.reversed_curve import ReversedCurve
from ocean.simulation.curve_simulator import CurveSimulator
from ocean.simulation.agent_states import IAgentState, RunningState
from ocean.simulation.kinematic_state import KinematicState


def _trapezoidal_accel(t: float, t_accel: float, t_brake_start: float,
                       t_total: float, a_rate: float, b_rate: float) -> float:
    """Piecewise trapezoidal acceleration profile.

    Returns acceleration at time t:
      t < t_accel       → +a_rate    (accelerating)
      t < t_brake_start → 0          (cruising)
      t < t_total       → -b_rate    (braking)
      t >= t_total      → +0.1*a_rate (gentle push to finish)
    """
    if t < t_accel:
        return a_rate
    elif t < t_brake_start:
        return 0.0
    elif t < t_total:
        return -b_rate
    else:
        return a_rate * 0.1  # gentle push to reach segment end


class TransitAgent:
    """A train that traverses a multi-segment transit route."""

    def __init__(
        self,
        network: TransitNetwork,
        line_code: str,
        speed: float = 1.0,
        dt: float = 0.01,
        dwell_time: float = 3.0,
        accel_rate: float = 0.0,
        brake_rate: float = 0.0,
        start_segment: int = 0,
    ):
        """Create a train agent on a transit line.

        Parameters
        ----------
        network : TransitNetwork
        line_code : str
            Which transit line to run on.
        speed : float
            Cruising speed (units/s).
        dt : float
            Default time step (seconds).
        dwell_time : float
            Pause duration at intermediate stations (seconds, real-time).
        accel_rate : float
            Acceleration magnitude (units/s^2). 0 = constant speed.
        brake_rate : float
            Braking magnitude (units/s^2). 0 = constant speed.
        start_segment : int
            0-indexed segment to start on.
        """
        self._network = network
        self._line_code = line_code
        self._speed = speed
        self._dt = dt
        self._dwell_time = dwell_time
        self._accel_rate = accel_rate
        self._brake_rate = brake_rate

        # Resolve route from TransitLine
        line = network.get_line(line_code)
        self._route = line.get_route()

        if len(self._route) < 2:
            raise ValueError(
                f'Line "{line_code}" must have at least 2 stations.')

        # Resolve splines for each segment
        n_seg = len(self._route) - 1

        if start_segment >= n_seg:
            raise ValueError(
                f'start_segment {start_segment} >= total segments {n_seg}.')

        self._splines = []
        self._edge_indices = []
        G = network.get_graph()

        for i in range(n_seg):
            src = self._route[i]
            tgt = self._route[i + 1]
            edge_idx = network.find_edge_index(src, tgt)
            if edge_idx == 0:
                raise ValueError(
                    f'No route between stations {src} and {tgt} '
                    f'on line "{line_code}".')

            spl = network.get_spline(edge_idx)

            # Check if draw direction is reversed relative to travel direction
            # G is a MultiGraph; G[src][tgt] returns a dict of {key: edge_data}
            edge_data = {}
            if G.has_edge(src, tgt):
                for d in G[src][tgt].values():
                    if d.get('edge_idx') == edge_idx:
                        edge_data = d
                        break

            draw_src = edge_data.get('draw_src', src)
            if draw_src != src:
                spl = ReversedCurve(spl)

            self._edge_indices.append(edge_idx)
            self._splines.append(spl)

        # State machine
        self._current_seg_idx = start_segment
        self._simulator = self._build_simulator(self._current_seg_idx)
        self._state: IAgentState = RunningState()
        self._dwell_remaining = 0.0
        self._clock = 0.0
        self._history = []
        self.record_history(self._simulator.get_state())

    # ------------------------------------------------------------------
    # Public properties (accessed by state objects)
    # ------------------------------------------------------------------
    #@property to avoid get_. ...() syntax.
    @property
    def network(self) -> TransitNetwork:
        return self._network

    @property
    def route(self) -> list[int]:
        return self._route

    @property
    def simulator(self) -> CurveSimulator:
        return self._simulator

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def current_seg_idx(self) -> int:
        return self._current_seg_idx

    @property
    def dwell_time(self) -> float:
        return self._dwell_time

    @property
    def dwell_remaining(self) -> float:
        return self._dwell_remaining

    #Same trick as property
    @dwell_remaining.setter
    def dwell_remaining(self, value: float) -> None:
        self._dwell_remaining = value

    # ------------------------------------------------------------------
    # Simulation — delegated to State
    # ------------------------------------------------------------------

    def step(self, dt: float = None, real_dt: float = None) -> None:
        """Advance the agent by one time step."""
        if dt is not None:
            self._dt = dt
        effective_real_dt = real_dt if real_dt is not None else self._dt
        self._clock += self._dt
        self._state.step(self, self._dt, effective_real_dt)

    def run_to_end(self) -> None:
        """Step until the agent has arrived."""
        max_iter = int(1e6)
        for _ in range(max_iter):
            if self.is_arrived():
                break
            self.step()

    def step_n(self, n: int) -> None:
        """Step N times."""
        for _ in range(n):
            self.step()
            if self.is_arrived():
                break

    # ------------------------------------------------------------------
    # Queries — delegated to State
    # ------------------------------------------------------------------

    def get_position(self) -> np.ndarray:
        """Current [x, y] position on the network."""
        return self._state.get_position(self)

    def get_tangent(self) -> np.ndarray:
        """Current heading direction."""
        return self._state.get_tangent(self)

    def get_progress(self) -> float:
        """Fraction of total route completed [0, 1]."""
        return self._state.get_progress(self)

    def is_arrived(self) -> bool:
        """True when the train has reached the terminal station."""
        return self._state.is_arrived()

    def get_line_code(self) -> str:
        return self._line_code

    def get_state(self) -> KinematicState:
        """Return the active KinematicState."""
        if self._simulator is not None:
            return self._simulator.get_state()
        return None

    def get_history(self) -> list:
        return list(self._history)

    def get_clock(self) -> float:
        return self._clock

    def get_agent_state(self) -> str:
        """Return 'running', 'dwelling', or 'arrived'."""
        return self._state.get_name()

    def get_color(self) -> tuple:
        """Return the line's RGB color."""
        line = self._network.get_line(self._line_code)
        return line.color

    def get_segment_count(self) -> int:
        """Total number of segments on route."""
        return len(self._route) - 1

    def get_active_spline(self):
        """Return the ParametricCurve for the current segment."""
        if self._current_seg_idx < len(self._splines):
            return self._splines[self._current_seg_idx]
        return None

    # ------------------------------------------------------------------
    # State management (called by state objects)
    # ------------------------------------------------------------------

    def set_state(self, new_state: IAgentState) -> None:
        """Transition to a new IAgentState."""
        self._state = new_state

    def record_history(self, s: KinematicState) -> None:
        """Append a KinematicState snapshot to history."""
        self._history.append({
            'time': self._clock,
            'seg_idx': self._current_seg_idx,
            'state': s,
        })

    def advance_to_next_segment(self) -> None:
        """Move to the next segment's spline."""
        self._current_seg_idx += 1
        self._simulator = self._build_simulator(self._current_seg_idx)
        self._state = RunningState()

    # ------------------------------------------------------------------
    # Private: Simulator Builder
    # ------------------------------------------------------------------

    def _build_simulator(self, seg_idx: int) -> CurveSimulator:
        """Build a CurveSimulator for the given segment.

        If accel_rate and brake_rate are both > 0, computes a trapezoidal
        acceleration profile (accel → cruise → brake).

        Kinematics: K = 1/2 mv^2, W = Fd = mad
        To stop: 1/2 v^2 = a*d  =>  d = v^2 / (2a)
        """
        spline = self._splines[seg_idx]

        if self._accel_rate > 0 and self._brake_rate > 0:
            v_cruise = self._speed
            a_rate = self._accel_rate
            b_rate = self._brake_rate
            L = spline.total_length()

            # Distances needed to accel to cruise and brake from cruise
            d_accel = v_cruise ** 2 / (2 * a_rate)
            d_brake = v_cruise ** 2 / (2 * b_rate)

            if d_accel + d_brake <= L:
                # Long track — Full trapezoidal profile
                t_accel = v_cruise / a_rate
                d_cruise = L - d_accel - d_brake
                t_cruise = d_cruise / v_cruise
                t_brake_start = t_accel + t_cruise
                t_total = t_brake_start + v_cruise / b_rate
            else:
                # Short track — Triangle profile (no cruise phase)
                v_peak = sqrt(2 * L / (1 / a_rate + 1 / b_rate))
                t_accel = v_peak / a_rate
                t_brake_start = t_accel  # cruise duration = 0
                t_total = t_accel + v_peak / b_rate

            def accel_fn(t):
                return _trapezoidal_accel(
                    t, t_accel, t_brake_start, t_total, a_rate, b_rate)

            return CurveSimulator(
                spline,
                initial_velocity=0.0,
                accel_profile=accel_fn,
                dt=self._dt,
                loop=False,
                max_velocity=self._speed * 1.1,
            )
        else:
            # Constant velocity (original behavior)
            return CurveSimulator(
                spline,
                initial_velocity=self._speed,
                acceleration=0.0,
                dt=self._dt,
                loop=False,
                max_velocity=self._speed * 2,
            )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def describe(self) -> str:
        n_seg = len(self._route) - 1
        names = self._network.get_station_names()
        nodes_sorted = sorted(self._network.get_graph().nodes())
        id_to_idx = {nid: i for i, nid in enumerate(nodes_sorted)}
        route_names = [names[id_to_idx[r]] for r in self._route if r in id_to_idx]

        desc = f"TransitAgent [{self._line_code}]:\n"
        desc += f"  Route:    {' -> '.join(route_names)}\n"
        desc += f"  State:    {self._state.get_name()}\n"
        desc += f"  Segment:  {self._current_seg_idx + 1} / {n_seg}\n"
        desc += f"  Progress: {self.get_progress() * 100:.1f}%\n"
        desc += f"  Clock:    {self._clock:.2f} s\n"
        if self._state.get_name() == "dwelling":
            desc += f"  Dwell:    {self._dwell_remaining:.2f} s remaining\n"
        print(desc)
        return desc
