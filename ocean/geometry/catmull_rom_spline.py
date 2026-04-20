"""
catmull_rom_spline.py — Centripetal Catmull-Rom spline through N control points.

Port of: +OceanMath/+Geometry/CatmullRomSpline.m

A Catmull-Rom spline is a piecewise cubic curve that passes exactly through
every control point (interpolating spline). The "centripetal" variant
(alpha = 0.5) prevents cusps and self-intersections.

Mathematics (Barry-Goldman recursive linear interpolation):
    Given 4 consecutive control points P0, P1, P2, P3 and their knot
    parameters t0, t1, t2, t3 derived from:

        t_{i+1} = t_i + ||P_{i+1} - P_i||^alpha

    The segment between P1 and P2 is:

        A1 = ((t1-u)/(t1-t0))*P0 + ((u-t0)/(t1-t0))*P1
        A2 = ((t2-u)/(t2-t1))*P1 + ((u-t1)/(t2-t1))*P2
        A3 = ((t3-u)/(t3-t2))*P2 + ((u-t2)/(t3-t2))*P3
        B1 = ((t2-u)/(t2-t0))*A1 + ((u-t0)/(t2-t0))*A2
        B2 = ((t3-u)/(t3-t1))*A2 + ((u-t1)/(t3-t1))*A3
        C  = ((t2-u)/(t2-t1))*B1 + ((u-t1)/(t2-t1))*B2

Usage:
    pts = np.array([[0,0], [1,2], [3,1], [5,3], [7,0]])
    spline = CatmullRomSpline(pts)
    P = spline.evaluate(np.linspace(0, 1, 200))
    plt.plot(P[:, 0], P[:, 1])
"""

import numpy as np
from ocean.geometry.parametric_curve import ParametricCurve


class CatmullRomSpline(ParametricCurve):
    """Centripetal Catmull-Rom spline (interpolating, C1-continuous)."""

    def __init__(self, control_points: np.ndarray, alpha: float = 0.5):
        """Construct a centripetal Catmull-Rom spline.

        Parameters
        ----------
        control_points : (K, d) array, K >= 2, d in {2, 3}
            The defining knots that the curve passes through.
        alpha : float in [0, 1]
            Parameterization exponent.
            0   = uniform  (can produce cusps)
            0.5 = centripetal (recommended, no cusps)
            1   = chordal  (can overshoot)
        """
        super().__init__()

        control_points = np.asarray(control_points, dtype=float)
        K, d = control_points.shape

        if K < 2:
            raise ValueError(f"At least 2 control points required. Got {K}.")
        if d < 2 or d > 3:
            raise ValueError(f"Control points must be 2D or 3D. Got d={d}.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be in [0, 1]. Got {alpha}.")

        self._control_points = control_points.copy()
        self._dimension = d
        self._alpha = alpha

        # Internal state (populated by _rebuild)
        self._knot_params: np.ndarray = np.array([])
        self._segment_map: np.ndarray = np.array([])
        self._padded: np.ndarray = np.array([])

        self._rebuild()

    # ------------------------------------------------------------------
    # Public API (implements ParametricCurve contract)
    # ------------------------------------------------------------------

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Map parameter t in [0,1] to (N, d) spatial coordinates."""
        t = np.atleast_1d(t).ravel()
        return self._evaluate_derivative(t, 0)

    def tangent(self, t: np.ndarray) -> np.ndarray:
        """First derivative dP/dt at parameter values t. Returns (N, d)."""
        t = np.atleast_1d(t).ravel()
        return self._evaluate_derivative(t, 1)

    def second_derivative(self, t: np.ndarray) -> np.ndarray:
        """Second derivative d2P/dt2 at parameter values t. Returns (N, d)."""
        t = np.atleast_1d(t).ravel()
        return self._evaluate_derivative(t, 2)

    def get_num_segments(self) -> int:
        """Number of cubic segments (K - 1)."""
        return len(self._segment_map)

    # ------------------------------------------------------------------
    # Mutation API (used by Track Designer for interactive editing)
    # ------------------------------------------------------------------

    def set_control_point(self, idx: int, new_pos: np.ndarray) -> None:
        """Move a control point and recompute the spline.

        Parameters
        ----------
        idx : int
            0-based index of the control point to move.
        new_pos : (d,) array
            New position for the control point.
        """
        K = len(self._control_points)
        if idx < 0 or idx >= K:
            raise IndexError(f"Index {idx} out of range [0, {K - 1}].")
        new_pos = np.asarray(new_pos, dtype=float)
        if len(new_pos) != self._dimension:
            raise ValueError(f"Position must have {self._dimension} elements.")

        self._control_points[idx] = new_pos
        self._rebuild()

    def add_control_point(self, new_pos: np.ndarray) -> None:
        """Append a new control point at the end."""
        new_pos = np.asarray(new_pos, dtype=float).reshape(1, -1)
        if new_pos.shape[1] != self._dimension:
            raise ValueError(f"Position must have {self._dimension} elements.")
        self._control_points = np.vstack([self._control_points, new_pos])
        self._rebuild()

    def remove_control_point(self, idx: int) -> None:
        """Remove a control point by 0-based index."""
        K = len(self._control_points)
        if K <= 2:
            raise ValueError("Cannot remove: minimum 2 control points required.")
        if idx < 0 or idx >= K:
            raise IndexError(f"Index {idx} out of range [0, {K - 1}].")
        self._control_points = np.delete(self._control_points, idx, axis=0)
        self._rebuild()

    def describe(self) -> str:
        """Return a human-readable summary string."""
        K = len(self._control_points)
        L = self.total_length()
        desc = (
            f"CatmullRomSpline:\n"
            f"  Dimension:      {self._dimension}D\n"
            f"  Control Points: {K}\n"
            f"  Segments:       {self.get_num_segments()}\n"
            f"  Alpha:          {self._alpha:.2f} (centripetal)\n"
            f"  Total Length:   {L:.4f}\n"
        )
        print(desc)
        return desc

    # ------------------------------------------------------------------
    # Private: rebuild internal state
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        """Recompute phantom endpoints, knot parameters, and segment map.

        Called whenever _control_points change (constructor, set, add, remove).
        This is the single source of truth for the spline's internal state.
        """
        cp = self._control_points
        K = len(cp)

        # 1. Phantom endpoints (momentum-preserving reflection)
        #    pStart = 2*P[0] - P[1],  pEnd = 2*P[-1] - P[-2]
        p_start = 2.0 * cp[0] - cp[1]
        p_end = 2.0 * cp[-1] - cp[-2]
        self._padded = np.vstack([p_start, cp, p_end])

        # 2. Knot parameters (centripetal parameterization)
        self._knot_params = CatmullRomSpline._compute_knot_params(
            self._padded, self._alpha
        )

        # 3. Segment map: each segment s uses padded points [s, s+1, s+2, s+3]
        #    and interpolates between points s+1 and s+2
        n_seg = K - 1
        kp = self._knot_params
        self._segment_map = np.zeros((n_seg, 2))
        for s in range(n_seg):
            self._segment_map[s, :] = [kp[s + 1], kp[s + 2]]

        # 4. Invalidate LUT (force lazy recomputation)
        self._lut_s = None
        self._lut_t = None

    # ------------------------------------------------------------------
    # Private: core evaluation engine
    # ------------------------------------------------------------------

    def _evaluate_derivative(self, t: np.ndarray, deriv_order: int) -> np.ndarray:
        """Unified evaluation of position / 1st / 2nd derivative.

        Uses the Barry-Goldman algorithm for position, and analytical
        differentiation of the same recursive formula for derivatives.

        The chain rule is applied because global t in [0,1] must be mapped
        to local knot-parameter space per segment.

        Parameters
        ----------
        t : (N,) array of global parameter values in [0, 1]
        deriv_order : 0 = position, 1 = tangent, 2 = second derivative

        Returns
        -------
        (N, d) array of evaluated points or derivative vectors
        """
        N = len(t)
        d = self._dimension
        result = np.zeros((N, d))

        kp = self._knot_params
        t_min = kp[1]          # First actual control point (index 1 in padded)
        t_max = kp[-2]         # Last actual control point (index K in padded)
        t_range = t_max - t_min

        n_seg = self.get_num_segments()
        pts = self._padded

        for i in range(N):
            # Map global t in [0,1] to knot-parameter space
            t_global = t_min + t[i] * t_range
            t_global = max(t_min, min(t_max, t_global))  # clamp

            # Find which segment this t belongs to
            seg_idx = 0
            for s in range(n_seg):
                if (t_global >= self._segment_map[s, 0] and
                        (t_global < self._segment_map[s, 1] or s == n_seg - 1)):
                    seg_idx = s
                    break

            # Extract the 4 control points and their knot params
            # Padded array indexing: segment s uses indices [s, s+1, s+2, s+3]
            P0 = pts[seg_idx]
            P1 = pts[seg_idx + 1]
            P2 = pts[seg_idx + 2]
            P3 = pts[seg_idx + 3]

            kt0 = kp[seg_idx]
            kt1 = kp[seg_idx + 1]
            kt2 = kp[seg_idx + 2]
            kt3 = kp[seg_idx + 3]

            u = t_global  # local knot-parameter value

            raw = CatmullRomSpline._barry_goldman_eval(
                u, P0, P1, P2, P3, kt0, kt1, kt2, kt3, deriv_order
            )
            # Chain rule: scale by t_range^derivOrder
            result[i, :] = raw * (t_range ** deriv_order)

        return result

    # ------------------------------------------------------------------
    # Private static: Barry-Goldman algorithm
    # ------------------------------------------------------------------

    @staticmethod
    def _barry_goldman_eval(
        u: float,
        P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, P3: np.ndarray,
        t0: float, t1: float, t2: float, t3: float,
        deriv_order: int,
    ) -> np.ndarray:
        """Unified Barry-Goldman evaluation for position and derivatives.

        Parameters
        ----------
        u : float
            Local knot-parameter value
        P0, P1, P2, P3 : (d,) arrays
            The 4 control points surrounding this segment
        t0, t1, t2, t3 : float
            Knot parameter values for these 4 points
        deriv_order : int
            0 = position, 1 = first derivative, 2 = second derivative

        Returns
        -------
        (d,) array — the evaluated value in knot-parameter space
        """
        # --- Level 1: A values (linear interpolation of adjacent points) ---
        A1 = ((t1 - u) / (t1 - t0)) * P0 + ((u - t0) / (t1 - t0)) * P1
        A2 = ((t2 - u) / (t2 - t1)) * P1 + ((u - t1) / (t2 - t1)) * P2
        A3 = ((t3 - u) / (t3 - t2)) * P2 + ((u - t2) / (t3 - t2)) * P3

        # --- Level 2: B values ---
        B1 = ((t2 - u) / (t2 - t0)) * A1 + ((u - t0) / (t2 - t0)) * A2
        B2 = ((t3 - u) / (t3 - t1)) * A2 + ((u - t1) / (t3 - t1)) * A3

        # --- Level 3: Position ---
        if deriv_order == 0:
            C = ((t2 - u) / (t2 - t1)) * B1 + ((u - t1) / (t2 - t1)) * B2
            return C

        # --- Level 1 derivatives (needed for 1st and 2nd order) ---
        dA1 = (P1 - P0) / (t1 - t0)
        dA2 = (P2 - P1) / (t2 - t1)
        dA3 = (P3 - P2) / (t3 - t2)

        # --- Level 2 derivatives (product rule) ---
        dB1 = ((-1.0 / (t2 - t0)) * A1 + ((t2 - u) / (t2 - t0)) * dA1
               + (1.0 / (t2 - t0)) * A2 + ((u - t0) / (t2 - t0)) * dA2)
        dB2 = ((-1.0 / (t3 - t1)) * A2 + ((t3 - u) / (t3 - t1)) * dA2
               + (1.0 / (t3 - t1)) * A3 + ((u - t1) / (t3 - t1)) * dA3)

        # --- Level 3: First derivative ---
        if deriv_order == 1:
            dC = ((-1.0 / (t2 - t1)) * B1 + ((t2 - u) / (t2 - t1)) * dB1
                  + (1.0 / (t2 - t1)) * B2 + ((u - t1) / (t2 - t1)) * dB2)
            return dC

        # --- Level 2 second derivatives ---
        # d2A = 0 (A's are linear in u), so only cross-terms survive.
        d2B1 = 2.0 * (-1.0 / (t2 - t0)) * dA1 + 2.0 * (1.0 / (t2 - t0)) * dA2
        d2B2 = 2.0 * (-1.0 / (t3 - t1)) * dA2 + 2.0 * (1.0 / (t3 - t1)) * dA3

        # --- Level 3: Second derivative ---
        d2C = (-2.0 / (t2 - t1) * dB1 + ((t2 - u) / (t2 - t1)) * d2B1
               + 2.0 / (t2 - t1) * dB2 + ((u - t1) / (t2 - t1)) * d2B2)
        return d2C

    # ------------------------------------------------------------------
    # Private static: knot parameterization
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_knot_params(points: np.ndarray, alpha: float) -> np.ndarray:
        """Centripetal knot parameterization.

        t_{i+1} = t_i + ||P_{i+1} - P_i||^alpha

        Parameters
        ----------
        points : (M, d) array of (padded) control points
        alpha : float parameterization exponent

        Returns
        -------
        (M,) array of cumulative knot parameter values
        """
        n_pts = len(points)
        kp = np.zeros(n_pts)
        for i in range(1, n_pts):
            dist = np.linalg.norm(points[i] - points[i - 1])
            kp[i] = kp[i - 1] + dist ** alpha
        return kp
