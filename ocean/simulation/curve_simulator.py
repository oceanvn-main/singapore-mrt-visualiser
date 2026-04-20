"""
curve_simulator.py — SUVAT physics engine for moving a body along a ParametricCurve.

Port of: +OceanMath/+Simulation/CurveSimulator.m (376 lines)

Given a parametric curve and initial conditions, this simulator advances
the body's position using Newtonian kinematics:

    ds = v * dt + 0.5 * a * dt^2   (SUVAT distance increment)
    v' = v + a * dt                 (velocity update)

The position on the curve is derived via arc-length inversion:
    target_s = old_arc_distance + ds
    tau      = curve.get_parameter_at_arc_length(target_s)
    position = curve.evaluate(tau)

Acceleration modes:
    CONSTANT  - a = 0, body moves at fixed speed
    UNIFORM   - a = constant scalar
    PROFILE   - a = user-supplied function a(time)
"""

from __future__ import annotations
import numpy as np

from ocean.geometry.parametric_curve import ParametricCurve
from ocean.simulation.kinematic_state import KinematicState


class CurveSimulator:
    """Physics engine for moving a body along a parametric curve."""

    def __init__(
        self,
        curve: ParametricCurve,
        initial_velocity: float = 1.0,
        acceleration: float = 0.0,
        accel_profile=None,
        dt: float = 0.01,
        loop: bool = False,
        max_velocity: float = float('inf'),
        start_parameter: float = 0.0,
    ):
        """Construct and configure the SUVAT physics engine.

        Parameters
        ----------
        curve : ParametricCurve
            The spline to traverse.
        initial_velocity : float
            Starting speed (units/s).
        acceleration : float
            Constant acceleration scalar (ignored if accel_profile given).
        accel_profile : callable or None
            Function a(time) -> float. Overrides `acceleration`.
        dt : float
            Time step in seconds.
        loop : bool
            Wrap around at end of curve?
        max_velocity : float
            Speed cap.
        start_parameter : float
            Starting parameter t on the curve, in [0, 1].
        """
        self._curve = curve
        self._total_arc = curve.total_length()
        self._dt = dt
        self._loop = loop
        self._max_velocity = max_velocity

        # Acceleration mode
        if accel_profile is not None:
            self._accel_mode = 'profile'
            self._accel_fn = accel_profile
            self._accel_value = 0.0
        elif acceleration == 0.0:
            self._accel_mode = 'constant'
            self._accel_fn = None
            self._accel_value = 0.0
        else:
            self._accel_mode = 'uniform'
            self._accel_fn = None
            self._accel_value = acceleration

        # Build initial state
        tau0 = start_parameter
        v0 = initial_velocity
        a0 = self._get_acceleration(0.0)
        pos = curve.evaluate(np.array([tau0]))[0]
        tang_raw = curve.tangent(np.array([tau0]))[0]
        norm_t = np.linalg.norm(tang_raw)
        tang = tang_raw / norm_t if norm_t > 1e-12 else np.array([1.0, 0.0])

        self._state = KinematicState(
            time=0.0, parameter=tau0, position=pos,
            velocity=v0, acceleration=a0, arc_distance=0.0,
            tangent=tang,
        )

        # History buffer
        self._history: list[KinematicState] = [self._state]
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def step(self, dt_override: float = None) -> None:
        """Advance the simulation by one time step.

        SUVAT kinematics:
            ds   = v * dt + 0.5 * a * dt^2    (distance increment)
            v'   = v + a * dt                  (velocity update)
            tau' = curve.get_parameter_at_arc_length(old_dist + ds)
        """
        dt = dt_override if dt_override is not None else self._dt

        v = self._state.velocity
        a = self._get_acceleration(self._state.time + dt)

        # SUVAT distance increment
        ds = v * dt + 0.5 * a * dt ** 2
        v_new = v + a * dt

        # Clamp velocity
        v_new = max(0.0, min(v_new, self._max_velocity))

        # Prevent negative distance (body can't go backward)
        if ds < 0:
            ds = 0.0
            v_new = 0.0

        # Arc-length tracking
        target_s = self._state.arc_distance + ds

        # Handle loop
        if self._loop:
            target_s = target_s % self._total_arc

        # True arc-length inversion via LUT
        tau_new = self._curve.get_parameter_at_arc_length(target_s)

        # Handle end-of-curve
        if self._loop:
            tau_new = tau_new % 1.0
        else:
            tau_new = min(tau_new, 1.0)
            if tau_new >= 1.0:
                v_new = 0.0  # stop at end

        # Recompute spatial quantities from the curve
        pos_new = self._curve.evaluate(np.array([tau_new]))[0]
        tang_raw = self._curve.tangent(np.array([tau_new]))[0]
        norm_t = np.linalg.norm(tang_raw)
        tang_new = tang_raw / norm_t if norm_t > 1e-12 else self._state.tangent

        # Build new state
        new_state = KinematicState(
            time=self._state.time + dt,
            parameter=tau_new,
            position=pos_new,
            velocity=v_new,
            acceleration=a,
            arc_distance=self._state.arc_distance + ds,
            tangent=tang_new,
        )

        self._state = new_state
        self._step_count += 1
        self._history.append(new_state)

    def run_for(self, n_steps: int) -> None:
        """Step N times."""
        for _ in range(n_steps):
            self.step()
            if not self._loop and self._state.parameter >= 1.0:
                break

    def run_until(self, target_time: float) -> None:
        """Step until simulation clock reaches target_time."""
        while self._state.time < target_time:
            self.step()
            if not self._loop and self._state.parameter >= 1.0:
                break

    def reset(self, start_parameter: float = 0.0,
              initial_velocity: float = 1.0) -> None:
        """Return to initial conditions."""
        tau0 = start_parameter
        pos = self._curve.evaluate(np.array([tau0]))[0]
        tang_raw = self._curve.tangent(np.array([tau0]))[0]
        norm_t = np.linalg.norm(tang_raw)
        tang = tang_raw / norm_t if norm_t > 1e-12 else np.array([1.0, 0.0])
        a0 = self._get_acceleration(0.0)

        self._state = KinematicState(
            time=0.0, parameter=tau0, position=pos,
            velocity=initial_velocity, acceleration=a0, arc_distance=0.0,
            tangent=tang,
        )
        self._history = [self._state]
        self._step_count = 0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_state(self) -> KinematicState:
        """Return the current KinematicState."""
        return self._state

    def get_history(self) -> list[KinematicState]:
        """Return all recorded KinematicState snapshots."""
        return list(self._history)

    def get_step_count(self) -> int:
        """Number of steps taken since last reset."""
        return self._step_count

    def estimate_travel_time(self, tau0: float = 0.0, tau1: float = 1.0) -> float:
        """Estimate travel time between two curve parameters."""
        seg_len = self._curve.arc_length(tau0, tau1)
        v = self._state.velocity

        if self._accel_mode == 'constant' or self._accel_value == 0.0:
            return seg_len / v if v > 1e-12 else float('inf')
        else:
            a = self._accel_value
            discriminant = v ** 2 + 2 * a * seg_len
            if discriminant < 0:
                return float('inf')
            t = (-v + np.sqrt(discriminant)) / a
            return t if t > 0 else float('inf')

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_acceleration(self, time: float) -> float:
        """Return acceleration at a given simulation time."""
        if self._accel_mode == 'constant':
            return 0.0
        elif self._accel_mode == 'uniform':
            return self._accel_value
        elif self._accel_mode == 'profile':
            return self._accel_fn(time)
        return 0.0
