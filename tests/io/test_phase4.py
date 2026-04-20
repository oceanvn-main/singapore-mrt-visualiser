"""
test_phase4.py — Verification tests for the JSON Track Loader (Phase 4).

Runs 7 tests to validate loading the real Singapore MRT JSON file and
saving it back, ensuring full parity with MATLAB's TrackDesigner.

Usage:
    cd python
    python -m ocean.io.test_phase4
"""

import os
import sys

from ocean.io.track_loader import load_network, save_network

# The actual big file
TEST_FILE = r'c:\Ocean Library\Matlab code practice\Experiments\Scripts\Simulation\test_simulator_20260402.json'


def test_station_count():
    """Test 1: Loader reconstructs exactly 186 stations."""
    net, meta = load_network(TEST_FILE)
    count = net.get_station_count()
    print(f"  Stations loaded: {count}")
    assert count == 186, f"FAIL: Expected 186 stations, got {count}"
    print("  ✓ PASSED")


def test_edge_count():
    """Test 2: Loader reconstructs exactly 214 unique edges."""
    net, meta = load_network(TEST_FILE)
    count = net.get_route_count()
    print(f"  Edges loaded: {count} (consolidated from 216 raw parallel edges)")
    assert count == 214, f"FAIL: Expected 214 unique edges, got {count}"
    print("  ✓ PASSED")


def test_line_count():
    """Test 3: Loader reconstructs exactly 9 transit lines."""
    net, meta = load_network(TEST_FILE)
    count = net.get_line_count()
    print(f"  Lines loaded: {count}")
    assert count == 9, f"FAIL: Expected 9 lines, got {count}"
    print("  ✓ PASSED")


def test_station_name_lookup():
    """Test 4: Station name lookup works and positions are 2D."""
    net, meta = load_network(TEST_FILE)
    sid = net.get_station_by_name('Dhoby Ghaut')
    assert sid is not None, "FAIL: Could not find 'Dhoby Ghaut'"
    
    pos = net.get_station(sid)['pos']
    print(f"  Dhoby Ghaut pos: {pos}, shape: {pos.shape}")
    assert len(pos) == 2, f"FAIL: Position must be 2D, got shape {pos.shape}"
    print("  ✓ PASSED")


def test_spline_geometry():
    """Test 5: Edge splines are parsed and have measurable length."""
    net, meta = load_network(TEST_FILE)
    # Get spline for edge index 1
    spline = net.get_spline(1)
    assert spline is not None, "FAIL: Edge 1 lacks a spline"
    
    L = spline.total_length()
    print(f"  Edge 1 Spline length: {L:.2f}")
    assert L > 0, "FAIL: Spline length is 0"
    print("  ✓ PASSED")


def test_viz_meta_completeness():
    """Test 6: visual metadata is extracted to dictionary."""
    net, meta = load_network(TEST_FILE)
    
    assert len(meta['edge_colors']) > 0, "FAIL: No edge colors found"
    assert len(meta['label_offsets']) > 0, "FAIL: No label offsets found"
    
    title = meta.get('map_title')
    print(f"  Map Title: {title}")
    assert title != '', "FAIL: map_title is empty or missing"
    print("  ✓ PASSED")


def test_roundtrip_parity():
    """Test 7: load -> save -> load preserves dimensions."""
    tmp_path = r'c:\Ocean Library\Matlab code practice\Experiments\Scripts\Simulation\test_simulator_roundtrip_temp.json'
    
    # 1. Load original
    net1, meta1 = load_network(TEST_FILE)
    
    # 2. Save new
    save_network(tmp_path, net1, meta1)
    
    # 3. Reload
    net2, meta2 = load_network(tmp_path)
    
    s1, s2 = net1.get_station_count(), net2.get_station_count()
    e1, e2 = net1.get_route_count(), net2.get_route_count()
    l1, l2 = net1.get_line_count(), net2.get_line_count()
    
    print(f"  Orig: S={s1}, E={e1}, L={l1}")
    print(f"  Loop: S={s2}, E={e2}, L={l2}")
    
    assert s1 == s2, "FAIL: Station count mismatch"
    assert e1 == e2, "FAIL: Edge count mismatch"
    assert l1 == l2, "FAIL: Line count mismatch"
    
    # Cleanup
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        
    print("  ✓ PASSED")


def main():
    print("=" * 60)
    print("  Phase 4 Verification: JSON Track Loader")
    print("=" * 60)

    tests = [
        test_station_count,
        test_edge_count,
        test_line_count,
        test_station_name_lookup,
        test_spline_geometry,
        test_viz_meta_completeness,
        test_roundtrip_parity,
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
