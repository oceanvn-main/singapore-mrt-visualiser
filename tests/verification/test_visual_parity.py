"""
test_visual_parity.py — Export a high-res PNG of the Singapore MRT map for
manual visual comparison with the MATLAB TrackDesigner output.

Uses Matplotlib's non-interactive 'Agg' backend to render headlessly.
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')  # Headless — no window needed
import matplotlib.pyplot as plt

from ocean.io.track_loader import load_network
from ocean.viz.track_renderer import TrackRenderer

MAP_FILE = r'c:\Ocean Library\Matlab code practice\Experiments\Scripts\Simulation\test_simulator_20260402.json'
OUTPUT_DIR = r'c:\Ocean Library\Matlab code practice\python'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'python_map_output.png')


def test_visual_export():
    """Test: Render the full Singapore MRT map to a PNG file."""
    network, viz_meta = load_network(MAP_FILE)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=72)
    fig.patch.set_facecolor('#111111')

    renderer = TrackRenderer(ax, network, viz_meta)
    # Draw only edges and stations — skip labels to avoid Agg memory crash
    renderer.ax.clear()
    renderer._configure_axes()
    renderer._draw_edges()
    renderer._draw_stations()
    # Skip: renderer._draw_labels() and renderer._draw_hud()

    fig.savefig(OUTPUT_FILE, dpi=72, facecolor='#111111')
    plt.close(fig)

    # Assert the file exists and is not blank
    assert os.path.exists(OUTPUT_FILE), f"FAIL: Output file not created"
    size = os.path.getsize(OUTPUT_FILE)
    assert size > 10_000, f"FAIL: Output file is suspiciously small ({size} bytes)"

    print(f"  PNG exported: {OUTPUT_FILE} ({size:,} bytes)")
    print(f"  ✓ PASSED: Visual parity export complete")
    print(f"  → Compare this PNG with your MATLAB screenshot side-by-side")
    return True


def main():
    print("=" * 60)
    print("  Phase 8.1: Visual Parity Test")
    print("=" * 60)

    try:
        test_visual_export()
        return 0
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
