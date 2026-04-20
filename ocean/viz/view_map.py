"""
view_map.py — Display a transit map with line legends and station types.

Usage:
    python -m ocean.viz.view_map
    python -m ocean.viz.view_map path/to/map.json
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from ocean.io.track_loader import load_network
from ocean.viz.track_renderer import TrackRenderer

import os
DEFAULT_MAP = os.path.join(os.path.dirname(__file__), '..', 'data', 'singapore_mrt.json')


def view_map(filepath: str = None, label_mode: str = 'major',
             show_weights: bool = True, weight_unit: str = 'km',
             scale_factor: float = 10.0, theme: str = 'dark'):
    """Load and display a transit map with legends.

    Parameters
    ----------
    filepath : str
        Path to the JSON map file.
    label_mode : str
        'all', 'major', or 'none'.
    show_weights : bool
        If True, display distance labels on each edge.
    weight_unit : str
        'raw', 'km', or 'miles'.
    scale_factor : float
        Conversion: 100 internal units = scale_factor km.
        e.g. 10 means 100 units = 10 km.
    theme : str
        'dark' or 'light'.
    """
    filepath = filepath or DEFAULT_MAP

    network, viz_meta = load_network(filepath)
    
    # ── Task 2: Network Statistics Extraction ──
    G = network.get_graph()
    num_stations = network.get_station_count()
    num_edges = network.get_route_count()
    num_lines = network.get_line_count()
    
    # Calculate physical distances
    total_internal = sum(float(d.get('weight', 0)) for u, v, k, d in G.edges(keys=True, data=True))
    total_km = total_internal * (scale_factor / 100.0)
    total_miles = total_km * 0.621371
    
    avg_km = total_km / num_edges if num_edges else 0
    avg_miles = total_miles / num_edges if num_edges else 0
    
    # Print via pandas for coursework requirement
    import pandas as pd
    stats_df = pd.DataFrame([
        {'Metric': 'Total Network Length', 'Value (km)': f'{total_km:.2f}', 'Value (miles)': f'{total_miles:.2f}'},
        {'Metric': 'Avg Inter-Station Distance', 'Value (km)': f'{avg_km:.2f}', 'Value (miles)': f'{avg_miles:.2f}'},
        {'Metric': 'Number of Stations', 'Value (km)': str(num_stations), 'Value (miles)': '-'},
        {'Metric': 'Number of Edges', 'Value (km)': str(num_edges), 'Value (miles)': '-'},
        {'Metric': 'Number of Lines', 'Value (km)': str(num_lines), 'Value (miles)': '-'}
    ])
    
    print("\n" + "="*50)
    print(" TASK 2: NETWORK STATISTICS")
    print("="*50)
    print(stats_df.to_string(index=False))
    print("="*50 + "\n")

    # Resolve theme colors for figure-level elements
    th = TrackRenderer.THEMES.get(theme, TrackRenderer.THEMES['dark'])

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(th['canvas_bg'])
    fig.canvas.manager.set_window_title("OceanViz — Map Viewer")
    ax.set_position([0.01, 0.01, 0.98, 0.98])

    renderer = TrackRenderer(ax, network, viz_meta, theme=theme)
    renderer.draw_network(label_mode=label_mode, show_weights=show_weights,
                          weight_unit=weight_unit, scale_factor=scale_factor)

    # ── Build 3-column legend: Train Services | Station Types | Path Notes ──
    # We use THREE separate legends positioned side by side at the bottom

    # Column 1: Train Services
    svc_handles = []
    for code in network.get_line_names():
        lo = network.get_line(code)
        n_st = len(lo.get_route()) if lo.get_route() else 0
        svc_handles.append(mlines.Line2D([], [], color=lo.color, lw=3.5,
                           label=f'{code}  {lo.name} ({n_st})'))

    leg1 = ax.legend(handles=svc_handles, loc='lower left',
                     title='Train Services', title_fontproperties={'weight': 'bold', 'size': 9},
                     fontsize=7, frameon=True, facecolor=th['legend_bg'],
                     edgecolor=th['legend_edge'], labelcolor=th['legend_label'], framealpha=0.92,
                     borderpad=0.8, handlelength=2.5,
                     bbox_to_anchor=(0.0, 0.0))
    leg1.set_zorder(500)
    leg1.get_title().set_color(th['legend_label'])
    ax.add_artist(leg1)  # Keep this legend when adding the next

    # Column 2: Station Types
    type_handles = [
        mlines.Line2D([], [], color='w', marker='o', markersize=8,
                      markeredgecolor='k', markeredgewidth=2,
                      linestyle='None', label='Interchange'),
        mlines.Line2D([], [], color='w', marker='D', markersize=7,
                      markeredgecolor='#333', markeredgewidth=1.5,
                      linestyle='None', label='Terminus'),
        mlines.Line2D([], [], color='w', marker='o', markersize=5,
                      markeredgecolor='#666', markeredgewidth=1,
                      linestyle='None', label='Standard'),
    ]


    leg2 = ax.legend(handles=type_handles, loc='lower right',
                     title='Station Types', title_fontproperties={'weight': 'bold', 'size': 9},
                     fontsize=7, frameon=True, facecolor=th['legend_bg'],
                     edgecolor=th['legend_edge'], labelcolor=th['legend_label'], framealpha=0.92,
                     borderpad=0.8,
                     bbox_to_anchor=(1.0, 0.0))
    leg2.set_zorder(500)
    leg2.get_title().set_color(th['legend_label'])
    ax.add_artist(leg2)

    # Column 3: Path Notes
    from matplotlib.legend_handler import HandlerBase

    class SharedPathHandler(HandlerBase):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            y_c = ydescent + height / 2.0
            gap = height * 0.25
            l1 = mlines.Line2D([xdescent, xdescent+width], [y_c + gap, y_c + gap], color='#FF4444', lw=2.5)
            l2 = mlines.Line2D([xdescent, xdescent+width], [y_c - gap, y_c - gap], color='#4444FF', lw=2.5)
            l1.set_transform(trans)
            l2.set_transform(trans)
            return [l1, l2]

    class LineCrossingHandler(HandlerBase):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            y_c = ydescent + height / 2.0
            gap = height * 0.4
            l1 = mlines.Line2D([xdescent, xdescent+width], [y_c - gap, y_c + gap], color='#FF4444', lw=2.5)
            l2 = mlines.Line2D([xdescent, xdescent+width], [y_c + gap, y_c - gap], color='#4444FF', lw=2.5)
            l1.set_transform(trans)
            l2.set_transform(trans)
            return [l1, l2]

    class SharedPathObj: pass
    class LineCrossingObj: pass

    path_handles = [SharedPathObj(), LineCrossingObj()]
    path_labels = ['Shared Path', 'Line Crossing']

    leg3 = ax.legend(handles=path_handles, labels=path_labels, loc='lower center',
                     handler_map={SharedPathObj: SharedPathHandler(), LineCrossingObj: LineCrossingHandler()},
                     title='Path Notes', title_fontproperties={'weight': 'bold', 'size': 9},
                     fontsize=7, frameon=True, facecolor=th['legend_bg'],
                     edgecolor=th['legend_edge'], labelcolor=th['legend_label'], framealpha=0.92,
                     borderpad=0.8,
                     bbox_to_anchor=(0.5, 0.0))
    leg3.set_zorder(500)
    leg3.get_title().set_color(th['legend_label'])
    ax.add_artist(leg3)

    # ── Unit annotation (top-right, non-overlapping) ──
    if show_weights:
        unit_labels = {'km': 'Distances in Kilometers (km)',
                       'miles': 'Distances in Miles (mi)',
                       'raw': 'Distances in Internal Units'}
        unit_text = unit_labels.get(weight_unit, weight_unit)
        ax.text(0.98, 0.98, unit_text,
                color=th['unit_badge_color'], fontsize=9, fontweight='bold',
                ha='right', va='top',
                transform=ax.transAxes,
                bbox=dict(facecolor=th['unit_badge_bg'], edgecolor=th['unit_badge_edge'],
                          alpha=0.92, pad=5, boxstyle='round,pad=0.4'),
                zorder=500)

    ax.autoscale_view()
    ax.margins(0.02)
    plt.show()
import argparse


def main():
    parser = argparse.ArgumentParser(description='OceanViz — Transit Map Viewer')
    parser.add_argument('filepath', nargs='?', default=None, help='Path to JSON map file')
    parser.add_argument('--unit', choices=['km', 'miles', 'raw'], default='km',
                        help='Distance unit for edge labels (default: km)')
    parser.add_argument('--scale', type=float, default=10.0,
                        help='Scale factor: 100 internal units = SCALE km (default: 10)')
    parser.add_argument('--no-weights', action='store_true',
                        help='Hide edge distance labels')
    parser.add_argument('--labels', choices=['all', 'major', 'none'], default='major',
                        help='Station label mode (default: major)')
    parser.add_argument('--theme', choices=['dark', 'light'], default='dark',
                        help='Color theme (default: dark)')
    args = parser.parse_args()

    view_map(filepath=args.filepath,
             label_mode=args.labels,
             show_weights=not args.no_weights,
             weight_unit=args.unit,
             scale_factor=args.scale,
             theme=args.theme)
    return 0


if __name__ == '__main__':
    sys.exit(main())
