"""
parametric_curve.py — Abstract base class for parametric curves in N-dimensional space.

Port of: +OceanMath/+Geometry/IParametricCurve.m

A parametric curve maps a scalar parameter t in [0, 1] to a point in R^d
(typically d = 2 or 3):

    P(t) = [x(t), y(t)]    t in [0, 1]

Contract (subclasses must implement):
    evaluate(t)          -> (N, d) points on the curve
    tangent(t)           -> (N, d) first derivative dP/dt
    second_derivative(t) -> (N, d) second derivative d2P/dt2

Concrete methods provided:
    curvature(t)                  -> (N,) scalar curvature
    arc_length(t0, t1)            -> scalar, true curve distance
    total_length()                -> scalar, full arc length
    get_parameter_at_arc_length(L)-> scalar, inverse arc-length lookup via LUT
    parameterize(n)               -> (n,) uniform parameter values
    parameterize_by_arc_length(n) -> (n,) equally-spaced-by-distance parameters
    get_control_points()          -> (K, d) defining knots
    get_dimension()               -> int
    get_num_segments()            -> int
"""

from abc import ABC, abstractmethod
import numpy as np
from ocean.geometry.simpson_estimate import simpson_estimate


class ParametricCurve(ABC):
    """Abstract base class for all parametric curves."""

    def __init__(self):
        self._control_points: np.ndarray = np.empty((0, 2))
        self._dimension: int = 2

        # Lazy arc-length Look-Up Table (built on first access)
        self._lut_s: np.ndarray | None = None
        self._lut_t: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Abstract methods (subclass MUST implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Map parameter values t in [0,1] to (N, d) spatial coordinates."""
        ...

    @abstractmethod
    def tangent(self, t: np.ndarray) -> np.ndarray:
        """First derivative dP/dt at parameter values t. Returns (N, d)."""
        ...

    @abstractmethod
    def second_derivative(self, t: np.ndarray) -> np.ndarray:
        """Second derivative d2P/dt2 at parameter values t. Returns (N, d)."""
        ...

    # ------------------------------------------------------------------
    # Concrete methods
    # ------------------------------------------------------------------

    def curvature(self, t: np.ndarray) -> np.ndarray:
        """Compute scalar curvature k at parameter t.

        k = |x'*y'' - y'*x''| / (x'^2 + y'^2)^(3/2)

        Returns (N,) array of curvature values.
        """
        t = np.atleast_1d(t).ravel()
        T = self.tangent(t)
        D2 = self.second_derivative(t)

        # Cross-product magnitude (2D: x'*y'' - y'*x'')
        norm_t3 = (T[:, 0] ** 2 + T[:, 1] ** 2) ** 1.5
        norm_t3 = np.maximum(norm_t3, 1e-12)  # guard against zero-speed

        return np.abs(T[:, 0] * D2[:, 1] - T[:, 1] * D2[:, 0]) / norm_t3

    def arc_length(self, t0: float = 0.0, t1: float = 1.0) -> float:
        """Compute true curved distance between two parameter values.

        Uses Simpson's 1/3 rule to integrate ||dP/dt|| from t0 to t1.
        """
        if t0 == t1:
            return 0.0

        # Speed function: ||dP/dt|| at parameter values
        speed_fn = lambda t_vals: self._compute_speed(t_vals)

        return simpson_estimate(speed_fn, t0, t1, 100)

    def total_length(self) -> float:
        """Full arc length of the curve from t=0 to t=1."""
        return self.arc_length(0.0, 1.0)

    def get_parameter_at_arc_length(self, target_l: float) -> float:
        """Find curve parameter t that matches arc length L.

        Uses a high-precision pre-computed Look-Up Table (LUT) to guarantee
        perfect monotonic evaluation, eliminating integration noise jitter.
        """
        # 1. Lazy-initialize the LUT
        if self._lut_s is None:
            self._build_lut()

        # 2. Safe extrapolation guards
        max_l = self._lut_s[-1]
        target_l = max(0.0, min(target_l, max_l))
        if target_l == 0.0:
            return 0.0
        if target_l >= max_l:
            return 1.0

        # 3. Ultra-fast monotonic interpolation (replaces MATLAB interp1)
        t = float(np.interp(target_l, self._lut_s, self._lut_t))
        return max(0.0, min(t, 1.0))

    def parameterize(self, n: int = 100) -> np.ndarray:
        """Generate N uniformly-spaced parameter values in [0, 1]."""
        return np.linspace(0.0, 1.0, n)

    def parameterize_by_arc_length(self, n: int = 100) -> np.ndarray:
        """Generate N parameter values equally spaced along curve length.

        Prevents bunching on sharp bends.
        """
        # Step 1: Dense uniform parameterization
        n_dense = max(n * 10, 1000)
        t_dense = np.linspace(0.0, 1.0, n_dense)

        # Step 2: Cumulative arc length at each dense sample
        speeds = self._compute_speed(t_dense)
        dt = t_dense[1] - t_dense[0]
        cum_len = np.concatenate([[0.0], np.cumsum(speeds[:-1] * dt)])
        total_len = cum_len[-1]

        # Step 3: Target equally-spaced arc length positions
        target_lengths = np.linspace(0.0, total_len, n)

        # Step 4: Invert cumulative length mapping via linear interp
        t_vals = np.zeros(n)
        t_vals[0] = 0.0
        t_vals[-1] = 1.0
        idx = 0
        for k in range(1, n - 1):
            s_target = target_lengths[k]
            while idx < n_dense - 1 and cum_len[idx + 1] < s_target:
                idx += 1
            # Linear interpolation between t_dense[idx] and t_dense[idx+1]
            denom = max(cum_len[idx + 1] - cum_len[idx], 1e-15)
            frac = (s_target - cum_len[idx]) / denom
            t_vals[k] = t_dense[idx] + frac * dt

        return t_vals

    def get_control_points(self) -> np.ndarray:
        """Return the defining knots (K, d)."""
        return self._control_points.copy()

    def get_dimension(self) -> int:
        """Return spatial dimension (2 or 3)."""
        return self._dimension

    def get_num_segments(self) -> int:
        """Number of piecewise polynomial segments (K - 1 by default)."""
        return len(self._control_points) - 1

    # ------------------------------------------------------------------
    # Protected methods
    # ------------------------------------------------------------------

    def _build_lut(self) -> None:
        """Pre-compute deterministic arc-length ↔ parameter mapping.

        Uses 5000 densely-sampled points to create a monotonic LUT
        that eliminates numerical jitter from repeated integration.
        """
        n_dense = 5000
        t_dense = np.linspace(0.0, 1.0, n_dense)

        speeds = self._compute_speed(t_dense)
        dt = t_dense[1] - t_dense[0]

        # Cumulative geometric distance
        s_dense = np.concatenate([[0.0], np.cumsum(speeds[:-1] * dt)])

        # Force strict monotonicity (remove duplicate arc-length values)
        s_unique, unique_idx = np.unique(s_dense, return_index=True)
        t_unique = t_dense[unique_idx]

        self._lut_s = s_unique
        self._lut_t = t_unique

    def _compute_speed(self, t_vals: np.ndarray) -> np.ndarray:
        """Compute ||dP/dt|| at each parameter value.

        Parameters
        ----------
        t_vals : (N,) array of parameter values

        Returns
        -------
        (N,) array of scalar speeds
        """
        t_vals = np.atleast_1d(t_vals).ravel()
        T = self.tangent(t_vals)  # (N, d)
        return np.sqrt(np.sum(T ** 2, axis=1))  # (N,)
