"""
test_phase1.py — Verification tests for the Geometry Engine (Phase 1).

Runs 6 tests to validate:
  1. Simpson integration accuracy
  2. Spline interpolation (passes through control points)
  3. Interior evaluation sanity check
  4. Arc length consistency (LUT inversion)
  5. ReversedCurve symmetry
  6. Derivative chain rule (reversed tangent = -original tangent)

Usage:
    cd python
    python -m ocean.geometry.test_phase1
"""

import numpy as np
import sys

from ocean.geometry.simpson_estimate import simpson_estimate
from ocean.geometry.catmull_rom_spline import CatmullRomSpline
from ocean.geometry.reversed_curve import ReversedCurve


def test_simpson_integration():
    """Test 1: Simpson's rule on sin(x) from 0 to pi should equal 2.0."""
    result = simpson_estimate(np.sin, 0, np.pi, 100)
    error = abs(result - 2.0)
    print(f"  Test 1 — Simpson sin(x) integral: {result:.12f}  (error: {error:.2e})")
    assert error < 1e-6, f"FAIL: error {error} exceeds 1e-6"
    print("  ✓ PASSED")


def test_spline_endpoints():
    """Test 2: Spline must pass exactly through first and last control points."""
    pts = np.array([[0, 0], [1, 2], [3, 1], [5, 3], [7, 0]], dtype=float)
    spline = CatmullRomSpline(pts)

    p_start = spline.evaluate(np.array([0.0]))[0]
    p_end = spline.evaluate(np.array([1.0]))[0]

    err_start = np.linalg.norm(p_start - pts[0])
    err_end = np.linalg.norm(p_end - pts[-1])

    print(f"  Test 2 — Start error: {err_start:.2e}, End error: {err_end:.2e}")
    assert err_start < 1e-10, f"FAIL: start error {err_start}"
    assert err_end < 1e-10, f"FAIL: end error {err_end}"
    print("  ✓ PASSED")


def test_spline_interior():
    """Test 3: Spline at t=0.5 should be between control points, not NaN."""
    pts = np.array([[0, 0], [1, 2], [3, 1], [5, 3], [7, 0]], dtype=float)
    spline = CatmullRomSpline(pts)

    p_mid = spline.evaluate(np.array([0.5]))[0]
    print(f"  Test 3 — Mid-point at t=0.5: ({p_mid[0]:.4f}, {p_mid[1]:.4f})")
    assert not np.any(np.isnan(p_mid)), "FAIL: NaN detected"
    # Should be roughly in the middle of the bounding box
    assert 0.0 <= p_mid[0] <= 7.0, f"FAIL: x={p_mid[0]} out of range"
    assert -1.0 <= p_mid[1] <= 4.0, f"FAIL: y={p_mid[1]} out of range"
    print("  ✓ PASSED")


def test_arc_length_consistency():
    """Test 4: Arc length LUT inversion must be self-consistent.

    If we find t_mid = get_parameter_at_arc_length(L/2),
    then arc_length(0, t_mid) should ≈ L/2.
    """
    pts = np.array([[0, 0], [1, 2], [3, 1], [5, 3], [7, 0]], dtype=float)
    spline = CatmullRomSpline(pts)

    L = spline.total_length()
    t_mid = spline.get_parameter_at_arc_length(L / 2)
    L_first_half = spline.arc_length(0.0, t_mid)
    error = abs(L_first_half - L / 2)

    print(f"  Test 4 — Total L: {L:.6f}, Half-L via LUT: {L_first_half:.6f}, error: {error:.2e}")
    assert error < 1e-3, f"FAIL: arc length error {error}"
    print("  ✓ PASSED")


def test_reversed_symmetry():
    """Test 5: ReversedCurve at t=0 must equal original at t=1."""
    pts = np.array([[0, 0], [1, 2], [3, 1], [5, 3], [7, 0]], dtype=float)
    spline = CatmullRomSpline(pts)
    rev = ReversedCurve(spline)

    p_rev_start = rev.evaluate(np.array([0.0]))[0]
    p_orig_end = spline.evaluate(np.array([1.0]))[0]
    err = np.linalg.norm(p_rev_start - p_orig_end)

    L_rev = rev.total_length()
    L_orig = spline.total_length()
    L_err = abs(L_rev - L_orig)

    print(f"  Test 5 — Position error: {err:.2e}, Length error: {L_err:.2e}")
    assert err < 1e-10, f"FAIL: position error {err}"
    assert L_err < 1e-8, f"FAIL: length error {L_err}"
    print("  ✓ PASSED")


def test_reversed_tangent():
    """Test 6: Tangent at t=0 of reversed = -tangent at t=1 of original."""
    pts = np.array([[0, 0], [1, 2], [3, 1], [5, 3], [7, 0]], dtype=float)
    spline = CatmullRomSpline(pts)
    rev = ReversedCurve(spline)

    tan_rev = rev.tangent(np.array([0.0]))[0]
    tan_orig = spline.tangent(np.array([1.0]))[0]
    err = np.linalg.norm(tan_rev - (-tan_orig))

    print(f"  Test 6 — Tangent rev(0): ({tan_rev[0]:.4f}, {tan_rev[1]:.4f})")
    print(f"           Tangent -orig(1): ({-tan_orig[0]:.4f}, {-tan_orig[1]:.4f})")
    print(f"           Error: {err:.2e}")
    assert err < 1e-8, f"FAIL: tangent error {err}"
    print("  ✓ PASSED")


def main():
    print("=" * 60)
    print("  Phase 1 Verification: Geometry Engine")
    print("=" * 60)

    tests = [
        test_simpson_integration,
        test_spline_endpoints,
        test_spline_interior,
        test_arc_length_consistency,
        test_reversed_symmetry,
        test_reversed_tangent,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        print(f"\n  Running: {test_fn.__doc__.strip().splitlines()[0]}")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
