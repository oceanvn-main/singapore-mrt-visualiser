"""
test_phase2.py — Verification tests for the Graph Engine & Transit Network (Phase 2).

Runs 8 tests to validate:
  1. Station CRUD
  2. Line registration
  3. Route with spline
  4. Shortest path (NetworkX Dijkstra)
  5. Interchange detection
  6. Terminal detection
  7. to_viz_data() structure
  8. Line route reconciliation

Usage:
    cd python
    python -m ocean.topology.test_phase2
"""

import numpy as np
import sys

from ocean.geometry.catmull_rom_spline import CatmullRomSpline
from ocean.topology.transit_network import TransitNetwork


def _make_spline(p1, p2):
    """Helper: create a simple 2-point spline between two positions."""
    return CatmullRomSpline(np.vstack([p1, p2]))


def test_station_crud():
    """Test 1: Station creation, retrieval, and counting."""
    net = TransitNetwork()
    id1 = net.add_station('Orchard', np.array([3, 7]))
    id2 = net.add_station('Dhoby Ghaut', np.array([4, 6]))

    print(f"  id1={id1}, id2={id2}, count={net.get_station_count()}")
    assert id1 == 1, f"FAIL: expected id1=1, got {id1}"
    assert id2 == 2, f"FAIL: expected id2=2, got {id2}"
    assert net.get_station_count() == 2, "FAIL: station count"
    assert net.get_station(id1)['name'] == 'Orchard', "FAIL: name mismatch"
    assert np.allclose(net.get_station(id1)['pos'], [3, 7]), "FAIL: pos mismatch"

    # By name lookup
    found = net.get_station_by_name('Dhoby Ghaut')
    assert found == id2, f"FAIL: get_by_name returned {found}"
    print("  ✓ PASSED")


def test_line_registration():
    """Test 2: Line creation and retrieval."""
    net = TransitNetwork()
    net.add_line('NSL', 'North-South Line', (0.85, 0.15, 0.15))
    assert net.get_line_count() == 1, "FAIL: line count"
    line = net.get_line('NSL')
    assert line.code == 'NSL', "FAIL: code mismatch"
    assert line.name == 'North-South Line', "FAIL: name mismatch"
    print("  ✓ PASSED")


def test_route_with_spline():
    """Test 3: Route creation with a CatmullRomSpline."""
    net = TransitNetwork()
    id1 = net.add_station('A', np.array([0, 0]))
    id2 = net.add_station('B', np.array([5, 5]))
    net.add_line('L1', 'Line 1', (1, 0, 0))

    spline = _make_spline(np.array([0, 0]), np.array([5, 5]))
    edge_idx = net.add_route_to_line('L1', id1, id2, spline)

    print(f"  edge_idx={edge_idx}")
    assert edge_idx == 1, f"FAIL: expected edge_idx=1, got {edge_idx}"
    assert net.get_spline(edge_idx) is spline, "FAIL: spline not stored"
    assert net.get_route_count() == 1, "FAIL: route count"
    print("  ✓ PASSED")


def test_shortest_path():
    """Test 4: Shortest path via NetworkX Dijkstra."""
    net = TransitNetwork()
    id1 = net.add_station('A', np.array([0, 0]))
    id2 = net.add_station('B', np.array([3, 0]))
    id3 = net.add_station('C', np.array([6, 0]))
    net.add_line('L1', 'Line 1', (1, 0, 0))

    s1 = _make_spline(np.array([0, 0]), np.array([3, 0]))
    s2 = _make_spline(np.array([3, 0]), np.array([6, 0]))
    net.add_route_to_line('L1', id1, id2, s1)
    net.add_route_to_line('L1', id2, id3, s2)

    path, dist = net.shortest_path(id1, id3)
    print(f"  path={path}, dist={dist:.4f}")
    assert path == [id1, id2, id3], f"FAIL: path={path}"
    assert dist > 0, f"FAIL: dist={dist}"
    print("  ✓ PASSED")


def test_interchange_detection():
    """Test 5: Interchange = station belonging to 2+ lines."""
    net = TransitNetwork()
    id1 = net.add_station('A', np.array([0, 0]))
    id2 = net.add_station('B', np.array([3, 0]))
    id3 = net.add_station('C', np.array([6, 0]))

    net.add_line('L1', 'Line 1', (1, 0, 0))
    net.add_line('L2', 'Line 2', (0, 1, 0))

    s12 = _make_spline(np.array([0, 0]), np.array([3, 0]))
    s23 = _make_spline(np.array([3, 0]), np.array([6, 0]))

    net.add_route_to_line('L1', id1, id2, s12)
    net.add_route_to_line('L2', id2, id3, s23)

    interchanges = net.get_interchanges()
    print(f"  interchanges={interchanges}")
    assert id2 in interchanges, f"FAIL: id2 not in interchanges"
    assert id1 not in interchanges, f"FAIL: id1 should not be interchange"
    print("  ✓ PASSED")


def test_terminal_detection():
    """Test 6: Terminal = degree-1 node (leaf in the graph)."""
    net = TransitNetwork()
    id1 = net.add_station('A', np.array([0, 0]))
    id2 = net.add_station('B', np.array([3, 0]))
    id3 = net.add_station('C', np.array([6, 0]))

    net.add_line('L1', 'Line 1', (1, 0, 0))
    s12 = _make_spline(np.array([0, 0]), np.array([3, 0]))
    s23 = _make_spline(np.array([3, 0]), np.array([6, 0]))
    net.add_route_to_line('L1', id1, id2, s12)
    net.add_route_to_line('L1', id2, id3, s23)

    terminals = net.get_terminals()
    print(f"  terminals={terminals}")
    assert id1 in terminals, "FAIL: id1 should be terminal"
    assert id3 in terminals, "FAIL: id3 should be terminal"
    assert id2 not in terminals, "FAIL: id2 should NOT be terminal"
    print("  ✓ PASSED")


def test_to_viz_data():
    """Test 7: to_viz_data() must return complete renderer-ready struct."""
    net = TransitNetwork()
    id1 = net.add_station('A', np.array([0, 0]))
    id2 = net.add_station('B', np.array([3, 0]))
    id3 = net.add_station('C', np.array([6, 0]))

    net.add_line('L1', 'Line 1', (1, 0, 0))
    s12 = _make_spline(np.array([0, 0]), np.array([3, 0]))
    s23 = _make_spline(np.array([3, 0]), np.array([6, 0]))
    net.add_route_to_line('L1', id1, id2, s12)
    net.add_route_to_line('L1', id2, id3, s23)

    data = net.to_viz_data()
    print(f"  positions shape: {data['positions'].shape}")
    print(f"  node_names: {data['node_names']}")
    print(f"  edge_pairs shape: {data['edge_pairs'].shape}")
    print(f"  n_splines: {len(data['splines'])}")

    assert data['positions'].shape == (3, 2), f"FAIL: positions shape {data['positions'].shape}"
    assert len(data['node_names']) == 3, "FAIL: node_names count"
    assert data['edge_pairs'].shape == (2, 2), f"FAIL: edge_pairs shape {data['edge_pairs'].shape}"
    assert len(data['splines']) == 2, "FAIL: splines count"
    assert 'interchange_ids' in data, "FAIL: missing interchange_ids"
    assert 'terminal_ids' in data, "FAIL: missing terminal_ids"
    print("  ✓ PASSED")


def test_reconciliation():
    """Test 8: Route reconciliation reorders click-order into traversal-order."""
    net = TransitNetwork()
    id1 = net.add_station('A', np.array([0, 0]))
    id2 = net.add_station('B', np.array([3, 0]))
    id3 = net.add_station('C', np.array([6, 0]))

    net.add_line('L1', 'Line 1', (1, 0, 0))
    s12 = _make_spline(np.array([0, 0]), np.array([3, 0]))
    s23 = _make_spline(np.array([3, 0]), np.array([6, 0]))
    net.add_route_to_line('L1', id1, id2, s12)
    net.add_route_to_line('L1', id2, id3, s23)

    # Scramble the route
    line = net.get_line('L1')
    line.set_route([id3, id1, id2])  # wrong order
    print(f"  Before reconcile: {line.get_route()}")

    net.reconcile_line_routes()
    route = line.get_route()
    print(f"  After reconcile:  {route}")

    # id2 must be in the middle (between id1 and id3)
    assert route[1] == id2, f"FAIL: middle station is {route[1]}, expected {id2}"
    # Route must be either [id1, id2, id3] or [id3, id2, id1]
    assert route in ([id1, id2, id3], [id3, id2, id1]), f"FAIL: route={route}"
    print("  ✓ PASSED")


def main():
    print("=" * 60)
    print("  Phase 2 Verification: Graph Engine & Transit Network")
    print("=" * 60)

    tests = [
        test_station_crud,
        test_line_registration,
        test_route_with_spline,
        test_shortest_path,
        test_interchange_detection,
        test_terminal_detection,
        test_to_viz_data,
        test_reconciliation,
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
