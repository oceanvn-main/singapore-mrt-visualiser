"""
test_physics_parity.py — Verify SUVAT kinematics match first-principles math.

Creates a simple straight-line track, spawns a train with known physics
parameters, and asserts position/velocity at fixed timestamps against
hand-calculated SUVAT values.
"""

import sys
import numpy as np

from ocean.topology.transit_network import TransitNetwork
from ocean.geometry.catmull_rom_spline import CatmullRomSpline
from ocean.transit.transit_simulation import TransitSimulation


def test_constant_velocity():
    """Test 1: A train at constant velocity covers exact distance."""
    net = TransitNetwork()
    net.add_station("A", np.array([0.0, 0.0]))
    net.add_station("B", np.array([100.0, 0.0]))
    net.add_line("T1", "Test Line", (1.0, 0.0, 0.0))

    spl = CatmullRomSpline(np.array([[0.0, 0.0], [100.0, 0.0]]))
    net.add_route_to_line("T1", 1, 2, spl)
    net.reconcile_line_routes()

    sim = TransitSimulation(net, dt=0.01)
    sim.add_agent("T1", speed=10.0, accel_rate=0.0, brake_rate=0.0,
                  dwell_time=0.0, start_segment=0)

    # At constant 10 u/s, should reach 100 units in 10 seconds
    sim.run_for(500)  # 500 * 0.01 = 5 seconds
    pos = sim.get_positions()[0]

    # After 5 seconds at 10 u/s = 50 units along x-axis
    expected_x = 50.0
    error = abs(pos[0] - expected_x)
    assert error < 2.0, f"FAIL: Expected x~{expected_x}, got {pos[0]:.2f} (err={error:.2f})"
    print(f"  Position at t=5s: ({pos[0]:.2f}, {pos[1]:.2f}), expected ~({expected_x}, 0)")
    print(f"  ✓ PASSED: Constant velocity covers correct distance")


def test_trapezoidal_profile_reaches_end():
    """Test 2: A train with accel/brake profile reaches the end station."""
    net = TransitNetwork()
    net.add_station("A", np.array([0.0, 0.0]))
    net.add_station("B", np.array([100.0, 0.0]))
    net.add_line("T2", "Test Line 2", (0.0, 1.0, 0.0))

    spl = CatmullRomSpline(np.array([[0.0, 0.0], [100.0, 0.0]]))
    net.add_route_to_line("T2", 1, 2, spl)
    net.reconcile_line_routes()

    sim = TransitSimulation(net, dt=0.01)
    agent = sim.add_agent("T2", speed=5.0, accel_rate=1.0, brake_rate=1.5,
                          dwell_time=0.0, start_segment=0)

    # Run until arrived
    sim.run_to_end()

    assert agent.is_arrived(), "FAIL: Agent did not arrive"
    final_pos = agent.get_position()
    error = np.linalg.norm(final_pos - np.array([100.0, 0.0]))
    assert error < 2.0, f"FAIL: Final pos {final_pos} too far from (100,0)"

    print(f"  Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"  Sim clock: {sim.get_clock():.2f}s")
    print(f"  ✓ PASSED: Trapezoidal profile train arrives at destination")


def test_multi_segment_dwell():
    """Test 3: A 3-station route with dwell time pauses correctly."""
    net = TransitNetwork()
    net.add_station("A", np.array([0.0, 0.0]))
    net.add_station("B", np.array([50.0, 0.0]))
    net.add_station("C", np.array([100.0, 0.0]))
    net.add_line("T3", "Test Line 3", (0.0, 0.0, 1.0))

    spl1 = CatmullRomSpline(np.array([[0.0, 0.0], [50.0, 0.0]]))
    spl2 = CatmullRomSpline(np.array([[50.0, 0.0], [100.0, 0.0]]))
    net.add_route_to_line("T3", 1, 2, spl1)
    net.add_route_to_line("T3", 2, 3, spl2)
    net.reconcile_line_routes()

    sim = TransitSimulation(net, dt=0.01)
    agent = sim.add_agent("T3", speed=10.0, accel_rate=0.0, brake_rate=0.0,
                          dwell_time=2.0, start_segment=0)

    # Run until arrived
    sim.run_to_end()

    assert agent.is_arrived(), "FAIL: Agent did not arrive"

    # Total time should be: (50/10) + 2.0 dwell + (50/10) = 12.0 seconds
    clock = sim.get_clock()
    expected_time = 12.0
    error = abs(clock - expected_time)
    assert error < 1.0, f"FAIL: Expected ~{expected_time}s, got {clock:.2f}s"

    print(f"  Total time: {clock:.2f}s (expected ~{expected_time}s, dwell=2s at B)")
    print(f"  ✓ PASSED: Multi-segment route with dwell time is correct")


def test_suvat_position_at_timestamps():
    """Test 4: SUVAT position verification at fixed timestamps.

    Setup: Straight track L=200, Speed=5.0, Accel=1.0, Brake=1.5
    Trapezoidal profile:
        d_accel = v^2/(2a) = 25/2 = 12.5 units,  t_accel = v/a = 5.0s
        d_brake = v^2/(2b) = 25/3 = 8.33 units,   t_brake = v/b = 3.33s
        d_cruise = 200 - 12.5 - 8.33 = 179.17 units, t_cruise = 179.17/5.0 = 35.83s
        t_total = 5.0 + 35.83 + 3.33 = 44.17s

    Expected positions (cumulative distance along x-axis):
        t=0:  x = 0
        t=1:  x = 0.5*1*1^2 = 0.5  (still accelerating)
        t=2:  x = 0.5*1*4 = 2.0
        t=5:  x = 0.5*1*25 = 12.5  (just finished accel)
        t=10: x = 12.5 + 5*5 = 37.5 (cruising)
    """
    net = TransitNetwork()
    net.add_station("A", np.array([0.0, 0.0]))
    net.add_station("B", np.array([200.0, 0.0]))
    net.add_line("T4", "Test SUVAT", (1.0, 1.0, 0.0))

    spl = CatmullRomSpline(np.array([[0.0, 0.0], [200.0, 0.0]]))
    net.add_route_to_line("T4", 1, 2, spl)
    net.reconcile_line_routes()

    sim = TransitSimulation(net, dt=0.01)
    sim.add_agent("T4", speed=5.0, accel_rate=1.0, brake_rate=1.5,
                  dwell_time=0.0, start_segment=0)

    checkpoints = {
        1.0: 0.5,     # s = 0.5*a*t^2 = 0.5
        2.0: 2.0,     # s = 0.5*1*4 = 2.0
        5.0: 12.5,    # s = 0.5*1*25 = 12.5
        10.0: 37.5,   # s = 12.5 + 5*(10-5) = 37.5
    }

    all_pass = True
    for target_time, expected_x in checkpoints.items():
        sim_steps = int(target_time / 0.01) - int(sim.get_clock() / 0.01)
        sim.run_for(max(1, sim_steps))

        pos = sim.get_positions()[0]
        error = abs(pos[0] - expected_x)
        status = "✓" if error < 2.0 else "✗"
        if error >= 2.0:
            all_pass = False
        print(f"    t={target_time:4.1f}s: x={pos[0]:7.2f} (expected {expected_x:7.2f}, err={error:.2f}) {status}")

    assert all_pass, "FAIL: One or more SUVAT checkpoints exceeded tolerance"
    print(f"  ✓ PASSED: SUVAT position checkpoints within tolerance")


def main():
    print("=" * 60)
    print("  Phase 8.2: Physics Parity Tests")
    print("=" * 60)

    tests = [
        test_constant_velocity,
        test_trapezoidal_profile_reaches_end,
        test_multi_segment_dwell,
        test_suvat_position_at_timestamps,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {t.__name__}: {e}")
            failed += 1

    print(f"{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
