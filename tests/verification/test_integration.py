"""
test_integration.py — End-to-end JSON round-trip integration test.

1. Build a 3-station network programmatically
2. Save to JSON
3. Load from JSON
4. Spawn a train
5. Verify it reaches the terminal station
"""

import sys
import os
import numpy as np

from ocean.topology.transit_network import TransitNetwork
from ocean.geometry.catmull_rom_spline import CatmullRomSpline
from ocean.io.track_loader import load_network, save_network
from ocean.transit.transit_simulation import TransitSimulation

TEMP_JSON = os.path.join(os.path.dirname(__file__), '_test_roundtrip.json')


def test_json_roundtrip():
    """Test: Build → Save → Load → Simulate → Arrive."""

    # 1. BUILD
    net = TransitNetwork()
    net.add_station("Alpha", np.array([0.0, 0.0]))
    net.add_station("Bravo", np.array([40.0, 20.0]))
    net.add_station("Charlie", np.array([100.0, 0.0]))
    net.add_line("RT", "Roundtrip Test", (0.2, 0.8, 0.4))

    spl1 = CatmullRomSpline(np.array([[0.0, 0.0], [40.0, 20.0]]))
    spl2 = CatmullRomSpline(np.array([[40.0, 20.0], [100.0, 0.0]]))
    net.add_route_to_line("RT", 1, 2, spl1)
    net.add_route_to_line("RT", 2, 3, spl2)
    net.reconcile_line_routes()

    print(f"  Built: {net.get_station_count()} stations, {net.get_route_count()} edges, {net.get_line_count()} lines")
    assert net.get_station_count() == 3
    assert net.get_route_count() == 2
    assert net.get_line_count() == 1

    # 2. SAVE
    viz_meta = {
        'edge_colors': {1: [0.2, 0.8, 0.4], 2: [0.2, 0.8, 0.4]},
        'label_offsets': {},
        'draw_orders': {},
        'map_title': ['Round-Trip Test']
    }
    save_network(TEMP_JSON, net, viz_meta)
    assert os.path.exists(TEMP_JSON), "FAIL: JSON file not created"
    size = os.path.getsize(TEMP_JSON)
    print(f"  Saved: {TEMP_JSON} ({size:,} bytes)")

    # 3. LOAD
    net2, viz2 = load_network(TEMP_JSON)
    assert net2.get_station_count() == 3, f"FAIL: Loaded {net2.get_station_count()} stations, expected 3"
    assert net2.get_line_count() == 1, f"FAIL: Loaded {net2.get_line_count()} lines, expected 1"
    print(f"  Loaded: {net2.get_station_count()} stations, {net2.get_line_count()} lines")

    # 4. SIMULATE
    sim = TransitSimulation(net2, dt=0.01)
    agent = sim.add_agent("RT", speed=5.0, accel_rate=1.0, brake_rate=1.5,
                          dwell_time=1.0, start_segment=0)
    sim.run_to_end()

    # 5. VERIFY
    assert agent.is_arrived(), "FAIL: Agent did not arrive"
    final_pos = agent.get_position()
    target = np.array([100.0, 0.0])
    error = np.linalg.norm(final_pos - target)
    assert error < 3.0, f"FAIL: Final pos {final_pos} too far from {target}"

    print(f"  Train arrived at ({final_pos[0]:.1f}, {final_pos[1]:.1f}) in {sim.get_clock():.1f}s")
    print(f"  ✓ PASSED: JSON round-trip integration test")

    # Cleanup
    if os.path.exists(TEMP_JSON):
        os.remove(TEMP_JSON)

    return True


def main():
    print("=" * 60)
    print("  Phase 8.3: Integration Test (JSON Round-Trip)")
    print("=" * 60)

    try:
        test_json_roundtrip()
        return 0
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
