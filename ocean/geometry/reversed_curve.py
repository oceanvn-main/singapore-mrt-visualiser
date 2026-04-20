"""
reversed_curve.py — Adapter that flips the parameterization of a curve.

Port of: +OceanMath/+Geometry/ReversedCurve.m

Wraps an existing ParametricCurve so that t=0 evaluates at the original t=1,
and t=1 evaluates at the original t=0. This solves routing when a transit
agent must traverse a generically-drawn undirected spline in reverse.

Chain rule:
    evaluate(t)          = base.evaluate(1 - t)
    tangent(t)           = -base.tangent(1 - t)        # d/dt of (1-t) = -1
    second_derivative(t) = base.second_derivative(1-t)  # (-1)^2 = +1
"""

import numpy as np
from ocean.geometry.parametric_curve import ParametricCurve


class ReversedCurve(ParametricCurve):
    """Decorator that reverses the direction of any ParametricCurve."""

    def __init__(self, base_curve: ParametricCurve):
        """Wrap an existing curve with reversed parameterization.

        Parameters
        ----------
        base_curve : ParametricCurve
            The original curve to reverse.
        """
        super().__init__()
        self._base = base_curve
        self._dimension = base_curve.get_dimension()
        self._control_points = np.flipud(base_curve.get_control_points())

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Position at reversed parameter."""
        t = np.atleast_1d(t).ravel()
        return self._base.evaluate(1.0 - t)

    def tangent(self, t: np.ndarray) -> np.ndarray:
        """First derivative with chain rule: multiply by -1."""
        t = np.atleast_1d(t).ravel()
        return -self._base.tangent(1.0 - t)

    def second_derivative(self, t: np.ndarray) -> np.ndarray:
        """Second derivative: (-1)^2 = 1, no negation needed."""
        t = np.atleast_1d(t).ravel()
        return self._base.second_derivative(1.0 - t)

    def get_parameter_at_arc_length(self, s: float) -> float:
        """Distance s from the reversed start = distance (L - s) from base start."""
        L = self._base.total_length()
        t_base = self._base.get_parameter_at_arc_length(L - s)
        return 1.0 - t_base

    def arc_length(self, t0: float = 0.0, t1: float = 1.0) -> float:
        """Range [t0, t1] in reversed = [1-t1, 1-t0] in the base curve."""
        return self._base.arc_length(1.0 - t1, 1.0 - t0)

    def total_length(self) -> float:
        """Total length is identical in both directions."""
        return self._base.total_length()
