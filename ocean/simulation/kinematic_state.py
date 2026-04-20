"""
kinematic_state.py — Immutable snapshot of a body's physical state on a curve.

Port of: +OceanMath/+Simulation/KinematicState.m (78 lines)

Frozen dataclass holding the complete kinematic state at a single instant:
time, curve parameter, position, velocity, acceleration, arc distance, tangent.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class KinematicState:
    """Immutable value object describing a body's physical state on a curve.

    Attributes
    ----------
    time : float
        Simulation clock (seconds).
    parameter : float
        Curve parameter t in [0, 1] (where on the curve).
    position : np.ndarray
        (d,) spatial coordinate [x, y].
    velocity : float
        Scalar speed along the curve (units/s).
    acceleration : float
        Scalar acceleration along the curve (units/s^2).
    arc_distance : float
        Cumulative distance traveled (units).
    tangent : np.ndarray
        (d,) unit tangent vector (direction of motion).
    """
    time: float = 0.0
    parameter: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    velocity: float = 0.0
    acceleration: float = 0.0
    arc_distance: float = 0.0
    tangent: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))

    def __repr__(self) -> str:
        pos_str = ', '.join(f'{x:.3f}' for x in self.position)
        return (
            f"KinematicState(t={self.time:.4f}s, tau={self.parameter:.4f}, "
            f"pos=[{pos_str}], v={self.velocity:.4f}, a={self.acceleration:.4f}, "
            f"dist={self.arc_distance:.4f})"
        )
