"""
test_phase3.py — Verification tests for the Physics Engine (Phase 3).

Runs 7 tests to validate:
  1. KinematicState immutability
  2. Constant-velocity CurveSimulator
  3. CurveSimulator reaches end of curve
  4. Trapezoidal acceleration profile
  5. TransitAgent multi-segment traversal
  6. DwellTime behavior
  7. TransitSimulation orchestrator

Usage:
    cd python
    python -m ocean.transit.test_phase3
"""

import numpy as np
import sys

from ocean.geometry.catmull_rom_spline import CatmullRomSpline
from ocean.simulation.kinematic_state import KinematicState
from ocean.simulation.curve_simulator import CurveSimulator
from ocean.topology.transit_network import TransitNetwork
from ocean.transit.transit_agent import TransitAgent
from ocean.transit.transit_simulation import TransitSimulation


def _make_spline(p1, p2):
    """Helper: create a 2-point Catmull-Rom spline (effectively a line)."""
    return CatmullRomSpline(np.vstack([p1, p2]))


def _build_test_network():
    """Build a 4-station, 2-line test network for agent tests."""
    net = TransitNetwork()
    # Stations: A(0,0) -> B(5,0) -> C(10,0) -> D(15,0)
    net.add_station('A', np.array([0, 0]))   # id=1
    net.add_station('B', np.array([5, 0]))   # id=2
    net.add_station('C', np.array([10, 0]))  # id=3
    net.add_station('D', np.array([15, 0]))  # id=4

    net.add_line('L1', 'Line 1', (1, 0, 0))

    s12 = _make_spline(np.array([0, 0]), np.array([5, 0]))
    s23 = _make_spline(np.array([5, 0]), np.array([10, 0]))
    s34 = _make_spline(np.array([10, 0]), np.array([15, 0]))

    net.add_route_to_line('L1', 1, 2, s12)
    net.add_route_to_line('L1', 2, 3, s23)
    net.add_route_to_line('L1', 3, 4, s34)

    return net


def test_kinematic_state_immutability():
    """Test 1: KinematicState is frozen (immutable)."""
    state = KinematicState(
        time=0.0, parameter=0.0,
        position=np.array([0.0, 0.0]),
        velocity=1.0, acceleration=0.0,
        arc_distance=0.0, tangent=np.array([1.0, 0.0]),
    )
    assert state.velocity == 1.0, "FAIL: velocity"
    assert state.time == 0.0, "FAIL: time"

    # Frozen: cannot mutate
    try:
        state.velocity = 2.0
        assert False, "FAIL: should not be able to mutate frozen dataclass"
    except AttributeError:
        pass  # expected

    print(f"  state = {state}")
    print("  ✓ PASSED")


def test_constant_velocity_simulator():
    """Test 2: CurveSimulator at constant velocity accumulates distance."""
    spline = _make_spline(np.array([0, 0]), np.array([10, 0]))
    sim = CurveSimulator(spline, initial_velocity=2.0, dt=0.01)

    sim.run_for(100)  # 100 steps * 0.01 dt * 2.0 v = 2.0 units of distance
    s = sim.get_state()

    print(f"  After 100 steps: dist={s.arc_distance:.4f}, param={s.parameter:.4f}")
    assert s.arc_distance > 0, "FAIL: no distance traveled"
    assert s.parameter > 0, "FAIL: parameter didn't advance"
    assert abs(s.velocity - 2.0) < 1e-6, f"FAIL: velocity drifted to {s.velocity}"
    print("  ✓ PASSED")


def test_simulator_reaches_end():
    """Test 3: CurveSimulator stops at end of curve (parameter >= 1)."""
    spline = _make_spline(np.array([0, 0]), np.array([5, 0]))
    sim = CurveSimulator(spline, initial_velocity=2.0, dt=0.01)

    sim.run_for(50000)  # way more than needed
    s = sim.get_state()

    print(f"  Final: param={s.parameter:.4f}, v={s.velocity:.4f}")
    assert s.parameter >= 1.0, f"FAIL: parameter={s.parameter}, didn't reach end"
    assert s.velocity == 0.0, f"FAIL: velocity={s.velocity}, should be 0 at end"
    print("  ✓ PASSED")


def test_trapezoidal_profile():
    """Test 4: Trapezoidal acceleration: velocity ramps up then down."""
    spline = _make_spline(np.array([0, 0]), np.array([20, 0]))
    L = spline.total_length()

    # Use the TransitAgent's trapezoidal profile builder
    from ocean.transit.transit_agent import _trapezoidal_accel

    v_cruise = 5.0
    a_rate = 2.0
    b_rate = 2.0
    d_accel = v_cruise ** 2 / (2 * a_rate)  # = 6.25
    d_brake = v_cruise ** 2 / (2 * b_rate)  # = 6.25
    # L ≈ 20, d_accel + d_brake = 12.5 < 20 → long track (full trapezoid)

    t_accel = v_cruise / a_rate         # = 2.5s
    d_cruise = L - d_accel - d_brake    # ≈ 7.5
    t_cruise = d_cruise / v_cruise      # ≈ 1.5s
    t_brake_start = t_accel + t_cruise  # ≈ 4.0s
    t_total = t_brake_start + v_cruise / b_rate  # ≈ 6.5s

    def accel_fn(t):
        return _trapezoidal_accel(t, t_accel, t_brake_start, t_total, a_rate, b_rate)

    sim = CurveSimulator(
        spline, initial_velocity=0.0,
        accel_profile=accel_fn, dt=0.01,
        max_velocity=v_cruise * 1.1,
    )

    # Run until past accel phase — velocity should be near cruise
    sim.run_until(t_accel + 0.5)
    v_mid = sim.get_state().velocity
    print(f"  At t={sim.get_state().time:.2f}: v={v_mid:.4f} (expect ~{v_cruise:.1f})")
    assert v_mid > v_cruise * 0.8, f"FAIL: velocity too low during cruise: {v_mid}"

    # Run to end — velocity should be near 0
    sim.run_for(50000)
    v_end = sim.get_state().velocity
    print(f"  At end: v={v_end:.4f}, param={sim.get_state().parameter:.4f}")
    assert sim.get_state().parameter >= 1.0, "FAIL: didn't reach end"
    print("  ✓ PASSED")


def test_agent_multi_segment():
    """Test 5: TransitAgent traverses all 3 segments and arrives."""
    net = _build_test_network()
    agent = TransitAgent(net, 'L1', speed=5.0, dt=0.01, dwell_time=0.0)

    agent.run_to_end()

    print(f"  Final state: {agent.get_agent_state()}")
    print(f"  Final pos:   {agent.get_position()}")
    print(f"  Progress:    {agent.get_progress():.2f}")

    assert agent.is_arrived(), "FAIL: agent not arrived"
    assert abs(agent.get_progress() - 1.0) < 1e-6, f"FAIL: progress={agent.get_progress()}"

    # Final position should be near station D(15, 0)
    final_pos = agent.get_position()
    assert abs(final_pos[0] - 15.0) < 0.5, f"FAIL: final_pos[0]={final_pos[0]}"
    print("  ✓ PASSED")


def test_dwell_time():
    """Test 6: Agent enters dwelling state at intermediate stations."""
    net = _build_test_network()
    agent = TransitAgent(net, 'L1', speed=5.0, dt=0.1, dwell_time=1.0)

    # Step until dwelling state is detected
    found_dwelling = False
    for _ in range(10000):
        agent.step(dt=0.1, real_dt=0.1)
        if agent.get_agent_state() == "dwelling":
            found_dwelling = True
            dwell_start = agent.dwell_remaining
            print(f"  Entered dwelling at seg {agent.current_seg_idx}, "
                  f"dwell_remaining={dwell_start:.2f}s")

            # Step a few more times — dwell should countdown
            agent.step(dt=0.1, real_dt=0.1)
            dwell_after = agent.dwell_remaining
            print(f"  After 1 step: dwell_remaining={dwell_after:.2f}s")
            assert dwell_after < dwell_start, "FAIL: dwell not counting down"
            break
        if agent.is_arrived():
            break

    assert found_dwelling, "FAIL: never entered dwelling state"
    print("  ✓ PASSED")


def test_simulation_orchestrator():
    """Test 7: TransitSimulation ticks multiple agents to completion."""
    net = _build_test_network()

    # Add a second line (reuse some stations)
    s24 = _make_spline(np.array([5, 0]), np.array([15, 0]))
    net.add_line('L2', 'Line 2', (0, 1, 0))
    net.add_route_to_line('L2', 2, 4, s24)

    sim = TransitSimulation(net, dt=0.05)
    sim.add_agent('L1', speed=5.0, dwell_time=0.0)
    sim.add_agent('L2', speed=3.0, dwell_time=0.0)

    assert sim.get_agent_count() == 2, "FAIL: agent count"

    sim.run_to_end()

    print(f"  Clock: {sim.get_clock():.2f}s")
    assert sim.is_all_arrived(), "FAIL: not all arrived"

    positions = sim.get_positions()
    print(f"  Final positions shape: {positions.shape}")
    assert positions.shape == (2, 2), f"FAIL: shape={positions.shape}"

    colors = sim.get_colors()
    assert colors.shape == (2, 3), f"FAIL: colors shape={colors.shape}"
    print("  ✓ PASSED")


def main():
    print("=" * 60)
    print("  Phase 3 Verification: Physics Engine")
    print("=" * 60)

    tests = [
        test_kinematic_state_immutability,
        test_constant_velocity_simulator,
        test_simulator_reaches_end,
        test_trapezoidal_profile,
        test_agent_multi_segment,
        test_dwell_time,
        test_simulation_orchestrator,
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
