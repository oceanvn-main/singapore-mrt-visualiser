"""
track_renderer.py — Static map plotting for TransitNetworks.

Port of: +OceanViz/+Designer/TrackRenderer.m

Takes a mathematical TransitNetwork and visual metadata dict (from JSON)
and plots them onto a Matplotlib Axes object with a default Dark Theme.
Supports station type markers (interchange, terminal, standard).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.patheffects as pe

from ocean.topology.transit_network import TransitNetwork


class TrackRenderer:
    """Renders track state onto a matplotlib Axes."""

    # ── Theme Presets ──
    THEMES = {
        'dark': {
            'canvas_bg':        '#0D0D0D',
            'text_color':       '#EEEEEE',
            'halo_color':       '#0D0D0D',
            'spine_color':      '#333333',
            'tick_color':       '#666666',
            'ix_face':          '#FFFFFF',
            'ix_edge':          '#000000',
            'tm_face':          '#FFFFFF',
            'tm_edge':          '#222222',
            'st_face':          '#FFFFFF',
            'st_edge':          '#444444',
            'weight_stroke':    '#0F0F0F',
            'legend_bg':        '#1A1A1A',
            'legend_edge':      '#333333',
            'legend_label':     '#DDDDDD',
            'unit_badge_bg':    '#1A1A1A',
            'unit_badge_edge':  '#333333',
            'unit_badge_color': '#FFB74D',
        },
        'light': {
            'canvas_bg':        '#FFFFFF',
            'text_color':       '#1A1A1A',
            'halo_color':       '#FFFFFF',
            'spine_color':      '#BBBBBB',
            'tick_color':       '#888888',
            'ix_face':          '#FFFFFF',
            'ix_edge':          '#000000',
            'tm_face':          '#FFFFFF',
            'tm_edge':          '#333333',
            'st_face':          '#FFFFFF',
            'st_edge':          '#666666',
            'weight_stroke':    '#FFFFFF',
            'legend_bg':        '#F5F5F5',
            'legend_edge':      '#CCCCCC',
            'legend_label':     '#222222',
            'unit_badge_bg':    '#F5F5F5',
            'unit_badge_edge':  '#CCCCCC',
            'unit_badge_color': '#C67200',
        },
    }

    # Layout constants (theme-independent)
    CURVE_WIDTH = 4.2          # matches MATLAB CurveLineWidth
    HALO_EXTRA = 4.0           # matches MATLAB HaloExtraWidth
    SAMPLES_PER_CURVE = 200
    COORD_SCALE = 1.3  # Scale all positions for larger visualization

    # Station markers (matching MATLAB TrackRenderer.updateStationStyle)
    INTERCHANGE_SIZE = 55
    INTERCHANGE_EDGE_W = 2.4
    TERMINAL_SIZE = 45
    TERMINAL_EDGE_W = 1.8
    STANDARD_SIZE = 15
    STANDARD_EDGE_W = 1.0

    def __init__(self, ax: plt.Axes, network: TransitNetwork, viz_meta: dict = None,
                 unit_scale: float = 0.01, theme: str = 'dark'):
        self.ax = ax
        self.network = network
        self.viz_meta = viz_meta or {}
        self.unit_scale = unit_scale
        self._theme = self.THEMES.get(theme, self.THEMES['dark'])

        if 'edge_colors' not in self.viz_meta:
            self.viz_meta['edge_colors'] = {}
        if 'label_offsets' not in self.viz_meta:
            self.viz_meta['label_offsets'] = {}
        if 'default_label_offset' not in self.viz_meta:
            self.viz_meta['default_label_offset'] = [0.0, 0.3]
        if 'draw_orders' not in self.viz_meta:
            self.viz_meta['draw_orders'] = {}

        self._configure_axes()

    def _configure_axes(self) -> None:
        import matplotlib.ticker as ticker
        self.ax.set_facecolor(self._theme['canvas_bg'])
        self.ax.margins(0.02)
        
        # Enable spines
        for spine in self.ax.spines.values():
            spine.set_color(self._theme['spine_color'])
            spine.set_visible(True)
            
        # Format ticks dynamically to explicitly show internal Units
        self.ax.tick_params(colors=self._theme['tick_color'], labelsize=7)
        self.ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=7))
        self.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=7))
        self.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f} u"))
        self.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{y:.0f} u"))
        
        self.ax.set_aspect('equal', adjustable='box')

    def draw_network(self, label_mode: str = 'all', show_weights: bool = False,
                     weight_unit: str = 'raw', scale_factor: float = 1.0) -> None:
        """Main entry point.

        Parameters
        ----------
        label_mode : str
            'all', 'major' (interchanges+terminals), or 'none'.
        show_weights : bool
            If True, display distance labels on each edge.
        weight_unit : str
            'raw' (internal units), 'km', or 'miles'.
        scale_factor : float
            Conversion factor: 100 internal units = scale_factor km.
            e.g. scale_factor=10 means 100 units = 10 km.
        """
        self.ax.clear()
        self._configure_axes()
        self._draw_edges()
        self._draw_stations()
        self._draw_labels(label_mode=label_mode)
        if show_weights:
            self._draw_weights(weight_unit=weight_unit, scale_factor=scale_factor)
        self._draw_hud()
        self._fit_axis_to_labels(label_mode=label_mode)

    def _draw_edges(self) -> None:
        """Draw spline curves sorted by drawOrder."""
        G = self.network.get_graph()
        
        # Group by station pair to detect shared corridors
        from collections import defaultdict
        pair_map = defaultdict(list)

        for u, v, d in G.edges(data=True):
            edge_idx = d.get('edge_idx')
            if edge_idx is None:
                continue
            spline = self.network.get_spline(edge_idx)
            if not spline:
                continue
            z_order = self.viz_meta['draw_orders'].get(edge_idx, 0)
            color = self.viz_meta['edge_colors'].get(edge_idx, [0.4, 0.8, 1.0])
            pair_map[frozenset([u, v])].append((z_order, spline, tuple(color)))

        t = np.linspace(0.0, 1.0, self.SAMPLES_PER_CURVE)
        S = self.COORD_SCALE

        base_edges = []
        elevated = []

        for pair, edges in pair_map.items():
            # Sort edges so they stack consistently (e.g. by color hash)
            edges.sort(key=lambda item: sum(item[2])) 
            n_edges = len(edges)

            base_width = self.CURVE_WIDTH  # 4.2
            step = 5.0  # Matches MATLAB's step size for candy-shell

            for i, (z_order, spline, color) in enumerate(edges):
                pts = spline.evaluate(t) * S
                
                # Candy-shell layering: thickest on bottom, thinnest on top
                target_width = base_width + (n_edges - 1 - i) * step
                
                if z_order == 0:
                    # zorder spaced slightly to ensure stack order
                    layer_z = 2 + (i * 0.1)
                    base_edges.append((layer_z, pts, color, target_width))
                else:
                    elevated.append((z_order, i, pts, color, target_width))

        # Pass 1 — base layer (no halo)
        base_edges.sort(key=lambda item: item[0])
        for layer_z, pts, color, width in base_edges:
            self.ax.plot(pts[:, 0], pts[:, 1],
                         color=color,
                         linewidth=width,
                         zorder=layer_z,
                         solid_capstyle='round')

        # Pass 2 — elevated (halo + color, on top of everything)
        for z_order, i, pts, color, width in elevated:
            halo_z = 10 + (z_order * 2) + (i * 0.1)
            halo_w = width + self.HALO_EXTRA

            # Halo: thick background line masks base layer underneath
            self.ax.plot(pts[:, 0], pts[:, 1],
                         color=self._theme['halo_color'],
                         linewidth=halo_w,
                         zorder=halo_z,
                         solid_capstyle='round')

            # Color line on top of halo
            self.ax.plot(pts[:, 0], pts[:, 1],
                         color=color,
                         linewidth=width,
                         zorder=halo_z + 0.05,
                         solid_capstyle='round')

    def _draw_stations(self) -> None:
        """Draw stations with type-specific markers: interchange=●, terminal=◇, standard=○."""
        G = self.network.get_graph()
        nodes = sorted(G.nodes())
        S = self.COORD_SCALE

        # Classify all stations
        interchanges = set(self.network.get_interchanges())
        terminals = set(self.network.get_terminals())

        # Separate into three groups for efficient batch scatter
        ix_pos, tm_pos, st_pos = [], [], []
        for nid in nodes:
            pos = np.array(G.nodes[nid].get('pos', [0, 0])) * S
            if nid in interchanges:
                ix_pos.append(pos)
            elif nid in terminals:
                tm_pos.append(pos)
            else:
                st_pos.append(pos)

        # Interchanges — large circle, thick black edge (like MATLAB)
        if ix_pos:
            arr = np.array(ix_pos)
            self.ax.scatter(arr[:, 0], arr[:, 1],
                            s=self.INTERCHANGE_SIZE,
                            marker='o',
                            facecolors=self._theme['ix_face'],
                            edgecolors=self._theme['ix_edge'],
                            linewidths=self.INTERCHANGE_EDGE_W,
                            zorder=100)

        # Terminals — diamond marker
        if tm_pos:
            arr = np.array(tm_pos)
            self.ax.scatter(arr[:, 0], arr[:, 1],
                            s=self.TERMINAL_SIZE,
                            marker='D',
                            facecolors=self._theme['tm_face'],
                            edgecolors=self._theme['tm_edge'],
                            linewidths=self.TERMINAL_EDGE_W,
                            zorder=100)

        # Standard — small hollow circle
        if st_pos:
            arr = np.array(st_pos)
            self.ax.scatter(arr[:, 0], arr[:, 1],
                            s=self.STANDARD_SIZE,
                            marker='o',
                            facecolors=self._theme['st_face'],
                            edgecolors=self._theme['st_edge'],
                            linewidths=self.STANDARD_EDGE_W,
                            zorder=100)

    def _draw_labels(self, label_mode: str = 'all') -> None:
        """Draw text labels with filtering."""
        if label_mode == 'none':
            return

        G = self.network.get_graph()
        nodes_sorted = sorted(G.nodes())
        S = self.COORD_SCALE

        major_ids = set()
        if label_mode == 'major':
            major_ids = set(self.network.get_interchanges()) | set(self.network.get_terminals())

        offsets = self.viz_meta['label_offsets']
        def_offset = self.viz_meta['default_label_offset']

        interchanges = set(self.network.get_interchanges())
        terminals = set(self.network.get_terminals())

        for nid in nodes_sorted:
            name = G.nodes[nid].get('name', '')
            if not name:
                continue
            if label_mode == 'major' and nid not in major_ids:
                continue

            pos = np.array(G.nodes[nid].get('pos', [0.0, 0.0])) * S
            offset = offsets.get(nid, def_offset)
            dx, dy = offset[0] * S, offset[1] * S
            label_x = pos[0] + dx
            label_y = pos[1] + dy

            ha = 'center'
            if dx > 0.05: ha = 'left'
            elif dx < -0.05: ha = 'right'

            va = 'center'
            if dy > 0.05: va = 'bottom'
            elif dy < -0.05: va = 'top'

            # Bold for interchanges/terminals
            fw = 'bold' if (nid in interchanges or nid in terminals) else 'light'

            self.ax.text(label_x, label_y, name,
                         color=self._theme['text_color'],
                         fontsize=7,
                         ha=ha, va=va,
                         zorder=110,
                         fontweight=fw,
                         clip_on=False)

    def _fit_axis_to_labels(self, label_mode: str = 'all') -> None:
        """Expand axis limits so every label is fully visible."""
        G = self.network.get_graph()
        S = self.COORD_SCALE
        offsets = self.viz_meta['label_offsets']
        def_offset = self.viz_meta['default_label_offset']

        major_ids = set()
        if label_mode == 'major':
            major_ids = set(self.network.get_interchanges()) | set(self.network.get_terminals())

        # Estimate character width in data coordinates
        # Use the axes transform to convert font points → data units
        fig = self.ax.get_figure()
        fig.canvas.draw()  # flush layout so transforms are valid

        # Get data-per-pixel scale from current limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        ax_bbox = self.ax.get_window_extent()
        dx_per_px = (xlim[1] - xlim[0]) / max(ax_bbox.width, 1)
        dy_per_px = (ylim[1] - ylim[0]) / max(ax_bbox.height, 1)

        # Approximate text width: ~5.5 px per char at fontsize 7
        CHAR_PX = 5.5

        x_min, x_max = xlim
        y_min, y_max = ylim

        for nid in G.nodes():
            name = G.nodes[nid].get('name', '')
            if not name:
                continue
            if label_mode == 'major' and nid not in major_ids:
                continue

            pos = np.array(G.nodes[nid].get('pos', [0, 0])) * S
            offset = offsets.get(nid, def_offset)
            dx, dy = offset[0] * S, offset[1] * S
            lx = pos[0] + dx
            ly = pos[1] + dy

            text_w = len(name) * CHAR_PX * dx_per_px
            text_h = 12 * dy_per_px  # ~12px line height

            # Determine alignment to compute bounding box
            ha = 'center'
            if dx > 0.05: ha = 'left'
            elif dx < -0.05: ha = 'right'

            if ha == 'left':
                lx_right = lx + text_w
                lx_left = lx
            elif ha == 'right':
                lx_right = lx
                lx_left = lx - text_w
            else:
                lx_right = lx + text_w / 2
                lx_left = lx - text_w / 2

            x_min = min(x_min, lx_left)
            x_max = max(x_max, lx_right)
            y_min = min(y_min, ly - text_h)
            y_max = max(y_max, ly + text_h)

        # Apply padded limits
        pad_x = (x_max - x_min) * 0.01
        pad_y = (y_max - y_min) * 0.01
        self.ax.set_xlim(x_min - pad_x, x_max + pad_x)
        self.ax.set_ylim(y_min - pad_y, y_max + pad_y)

    def _draw_weights(self, weight_unit: str = 'raw', scale_factor: float = 1.0) -> None:
        """Draw edge weights at the midpoint of each curve.

        Uses per-line deduplication: if consecutive edges on the same
        transit line have the same rounded distance, only one label is shown.

        Parameters
        ----------
        weight_unit : str
            'raw' shows internal units, 'km' converts to kilometers,
            'miles' converts to miles.
        scale_factor : float
            100 internal units = scale_factor km.
            So: km = weight * scale_factor / 100
                miles = km * 0.621371
        """
        KM_PER_MILE = 0.621371
        G = self.network.get_graph()
        S = self.COORD_SCALE

        # ── Step 1: Build per-line edge lists in route order ──
        from collections import defaultdict
        line_edges = defaultdict(list)  # line_code -> [(edge_idx, u, v, weight), ...]

        for line_code in self.network.get_line_names():
            line = self.network.get_line(line_code)
            route = line.get_route()
            if not route or len(route) < 2:
                continue
            for i in range(len(route) - 1):
                src, tgt = route[i], route[i + 1]
                # Find this edge in the graph
                for u, v, d in G.edges(data=True):
                    eidx = d.get('edge_idx')
                    if eidx is None:
                        continue
                    if (u == src and v == tgt) or (u == tgt and v == src):
                        w = d.get('weight', 1.0)
                        line_edges[line_code].append((eidx, u, v, w))
                        break

        # Collect all edge indices that belong to at least one line
        shown_edges = set()

        # ── Step 2: Deduplicate per line ──
        for line_code, edges in line_edges.items():
            prev_label = None
            for eidx, u, v, weight in edges:
                # Convert
                if weight_unit == 'km':
                    val = round(weight * scale_factor / 100.0)
                elif weight_unit == 'miles':
                    val = round(weight * scale_factor / 100.0 * KM_PER_MILE)
                else:
                    val = round(weight, 1)

                label = str(int(val)) if weight_unit != 'raw' else f'{val:.1f}'

                # Skip if same as previous on this line
                if label == prev_label:
                    prev_label = label
                    continue
                prev_label = label
                shown_edges.add((eidx, label))

        # ── Step 3: Also handle edges not on any line (orphan edges) ──
        line_edge_ids = set()
        for edges in line_edges.values():
            for eidx, _, _, _ in edges:
                line_edge_ids.add(eidx)

        for u, v, d in G.edges(data=True):
            eidx = d.get('edge_idx')
            if eidx is None or eidx in line_edge_ids:
                continue
            weight = d.get('weight', 1.0)
            if weight_unit == 'km':
                val = round(weight * scale_factor / 100.0)
                label = str(int(val))
            elif weight_unit == 'miles':
                val = round(weight * scale_factor / 100.0 * KM_PER_MILE)
                label = str(int(val))
            else:
                label = f'{weight:.1f}'
            shown_edges.add((eidx, label))

        # ── Step 4: Draw ──
        for eidx, label in shown_edges:
            spline = self.network.get_spline(eidx)
            if not spline:
                continue
            pt = spline.evaluate(np.array([0.5]))[0] * S

            edge_color = self.viz_meta['edge_colors'].get(eidx, [1.0, 0.72, 0.3])
            
            # Contrast logic: dark text on white stroke (light mode), light text on dark stroke (dark mode)
            if self._theme.get('canvas_bg') == '#FFFFFF':
                text_c = [c * 0.45 for c in edge_color]
            else:
                text_c = [min(1.0, c * 0.6 + 0.4) for c in edge_color]

            self.ax.text(pt[0], pt[1], label,
                         color=text_c,
                         fontsize=5.5,
                         fontweight='bold',
                         ha='center', va='center',
                         zorder=120,
                         path_effects=[pe.withStroke(linewidth=2.0, foreground=self._theme['weight_stroke'])])

    def _draw_hud(self) -> None:
        """Draw map title and legends."""
        title = self.viz_meta.get('map_title', '')
        if isinstance(title, list):
            title = '\n'.join(title)
        if title:
            pos = self.viz_meta.get('title_pos', [0.05, 0.95])
            self.ax.text(pos[0], pos[1], title,
                         color=self._theme['text_color'],
                         fontsize=14,
                         fontweight='bold',
                         transform=self.ax.transAxes,
                         ha='left', va='top')

        pos1 = self.viz_meta.get('legend_pos1')
        if pos1 is not None:
            handles = self.build_legend_handles()
            self.ax.legend(handles=handles, loc='lower left',
                           bbox_to_anchor=(0.02, 0.02),
                           ncol=2, columnspacing=0.8, handletextpad=0.4,
                           facecolor=self._theme['legend_bg'], edgecolor='none',
                           labelcolor=self._theme['legend_label'], fontsize=6, framealpha=0.5)

    def build_legend_handles(self) -> list:
        """Build legend handles: Train Services + Station Types + Path Notes."""
        handles = []

        # ── Train Services ──
        handles.append(mlines.Line2D([], [], color='none', label=''))
        handles.append(mlines.Line2D([], [], color='none',
                       label='$\\bf{Train\\ Services}$'))
        for code in self.network.get_line_names():
            lo = self.network.get_line(code)
            n_stations = len(lo.get_route()) if lo.get_route() else 0
            h = mlines.Line2D([], [], color=lo.color, linewidth=3.5,
                               label=f'{code}  {lo.name} ({n_stations})')
            handles.append(h)

        # ── Station Types ──
        handles.append(mlines.Line2D([], [], color='none', label=''))
        handles.append(mlines.Line2D([], [], color='none',
                       label='$\\bf{Station\\ Types}$'))
        handles.append(mlines.Line2D([], [], color='w', marker='o',
                       markersize=8, markeredgecolor='k', markeredgewidth=2,
                       linestyle='None', label='Interchange'))
        handles.append(mlines.Line2D([], [], color='w', marker='D',
                       markersize=7, markeredgecolor='#333', markeredgewidth=1.5,
                       linestyle='None', label='Terminus'))
        handles.append(mlines.Line2D([], [], color='w', marker='o',
                       markersize=5, markeredgecolor='#666', markeredgewidth=1,
                       linestyle='None', label='Standard'))

        # ── Path Notes ──
        handles.append(mlines.Line2D([], [], color='none', label=''))
        handles.append(mlines.Line2D([], [], color='none',
                       label='$\\bf{Path\\ Notes}$'))
        # Shared path: two colored lines stacked
        handles.append(mlines.Line2D([0, 1], [0, 0], color='#FF4444',
                       linewidth=4, label='Shared Path',
                       marker='|', markersize=6, markeredgecolor='#4444FF'))
        # Line crossing: halo gap
        handles.append(mlines.Line2D([0, 1], [0, 0], color='#4444FF',
                       linewidth=4, label='Line Crossing',
                       marker='x', markersize=5, markeredgecolor='#444'))

        # Force column breaks dynamically so all Train Services stay in the left
        left_col_count = 2 + len(self.network.get_line_names())
        while len(handles) < left_col_count * 2:
            handles.append(mlines.Line2D([], [], color='none', label=''))

        return handles
