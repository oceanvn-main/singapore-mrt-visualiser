"""
simpson_estimate.py — Hand-implemented Simpson's 1/3 Rule for numerical integration.

Port of: +OceanMath/SimpsonEstimate.m

This replaces any need for scipy.integrate.quad. We compute definite integrals
using the composite Simpson's 1/3 rule with vectorized NumPy operations.
"""

import numpy as np


def simpson_estimate(g, a: float, b: float, n: int = 100) -> float:
    """Approximate the definite integral of g from a to b using Simpson's 1/3 Rule.

    Parameters
    ----------
    g : callable
        The integrand function. Must accept a 1-D numpy array and return
        a 1-D numpy array of the same length.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    n : int
        Number of subintervals (MUST be a positive even integer).

    Returns
    -------
    float
        Approximate value of the definite integral.

    Examples
    --------
    >>> simpson_estimate(np.sin, 0, np.pi, 20)  # ≈ 2.0
    """
    if n <= 0 or n % 2 != 0:
        raise ValueError(f"n must be a positive even integer, got {n}")

    # Step width
    dx = (b - a) / n

    # Grid points: x0, x1, ..., xn  (n+1 values)
    x = np.linspace(a, b, n + 1)

    # Evaluate the integrand at all grid points (vectorized)
    y = g(x)

    # Build Simpson coefficient vector: [1, 4, 2, 4, 2, ..., 4, 1]
    coeffs = np.ones(n + 1)
    coeffs[1::2] = 4    # odd-index positions: 4
    coeffs[2:-1:2] = 2  # even-index interior positions: 2

    # Simpson's 1/3 formula
    return (dx / 3.0) * np.dot(coeffs, y)
