"""
bezier_curve.py — Bezier curve of arbitrary degree via Bernstein basis.

Port of: +OceanMath/+Geometry/BezierCurve.m (184 lines)

A Bezier curve of degree n is defined by (n+1) control points.
Uses vectorized Bernstein polynomial evaluation (de Casteljau's numerical equivalent).
"""

from __future__ import annotations
import math
import numpy as np

from ocean.geometry.parametric_curve import ParametricCurve


class BezierCurve(ParametricCurve):
    """Bezier curve of arbitrary degree via Bernstein basis."""

    def __init__(self, control_points: np.ndarray):
        """Construct a Bezier curve from control points.

        Parameters
        ----------
        control_points : np.ndarray
            Shape (K, d) where K >= 2 and d in {2, 3}.
        """
        super().__init__()
        pts = np.asarray(control_points, dtype=float)
        
        K, d = pts.shape
        if K < 2:
            raise ValueError(f"At least 2 control points required. Got {K}.")
        if d < 2 or d > 3:
            raise ValueError(f"Control points must be 2D or 3D. Got d = {d}.")

        self._control_points = pts.copy()
        self._dim = d
        self._degree = K - 1

    @property
    def degree(self) -> int:
        """Polynomial degree (= K - 1)."""
        return self._degree

    def get_control_points(self) -> np.ndarray:
        """Return the (K, d) array of control points."""
        return self._control_points.copy()

    def set_control_point(self, idx: int, new_pos: np.ndarray) -> None:
        """Move a control point."""
        if idx < 0 or idx >= len(self._control_points):
            raise IndexError(f"Index {idx} out of range [0, {len(self._control_points)-1}].")
        pos = np.asarray(new_pos, dtype=float)
        if pos.shape[0] != self._dim:
            raise ValueError(f"New position must have {self._dim} elements.")
        self._control_points[idx] = pos
        self._invalidate_arc_length()

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Map parameter t in [0,1] to [N x d] points."""
        return self._de_casteljau(t, 0)

    def tangent(self, t: np.ndarray) -> np.ndarray:
        """First derivative dC/dt at parameter values t."""
        return self._de_casteljau(t, 1)

    def second_derivative(self, t: np.ndarray) -> np.ndarray:
        """Second derivative d2C/dt2."""
        return self._de_casteljau(t, 2)

    def get_num_segments(self) -> int:
        """Bezier is a single segment."""
        return 1

    def _de_casteljau(self, t: np.ndarray, deriv_order: int) -> np.ndarray:
        """Unified evaluation via Bernstein polynomials."""
        t_arr = np.atleast_1d(t)
        pts = self._control_points.copy()
        n = self._degree

        # Build derivative control points
        if deriv_order >= 1:
            # First difference: Q_i = n * (P_{i+1} - P_i)
            pts = n * np.diff(pts, n=1, axis=0)
            if deriv_order >= 2 and len(pts) >= 2:
                # Second difference: R_i = (n-1) * (Q_{i+1} - Q_i)
                pts = (n - 1) * np.diff(pts, n=1, axis=0)

        K = pts.shape[0]
        if K == 0:
            return np.zeros((len(t_arr), self._dim))

        result = np.zeros((len(t_arr), self._dim))
        n_poly = K - 1
        
        for i in range(K):
            # Broadcast friendly: C(n, i) * (t^i) * (1-t)^(n-i)
            # Math.comb requires integers, so we compute coefficients directly
            coeff = math.comb(n_poly, i)
            B = coeff * (t_arr ** i) * ((1.0 - t_arr) ** (n_poly - i))
            # B is shape (N,)
            # _pts[i] is shape (d,)
            result += B[:, np.newaxis] * pts[i]

        return result if np.ndim(t) > 0 else result[0]

    def describe(self) -> str:
        """Print a human-readable summary."""
        K = len(self._control_points)
        L = self.total_length()
        desc = (
            f"BezierCurve:\n"
            f"  Dimension:      {self._dim}D\n"
            f"  Control Points: {K}\n"
            f"  Degree:         {self._degree}\n"
            f"  Total Length:   {L:.4f}\n"
        )
        print(desc)
        return desc
