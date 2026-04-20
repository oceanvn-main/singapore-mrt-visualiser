"""
transit_simulator_app.py — Interactive Transit Simulator with Control Panel.
Uses ONLY: NumPy, NetworkX, Matplotlib.
# Roles: AppContext + LayoutBuilder + InteractionHandler + Commands + TimelineController + OverlayRenderer
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider, RadioButtons, TextBox

from ocean.io.track_loader import load_network
from ocean.transit.transit_simulation import TransitSimulation
from ocean.viz.track_renderer import TrackRenderer


class TransitSimulatorApp:
    import os
    DEFAULT_MAP = os.path.join(os.path.dirname(__file__), '..', 'data', 'singapore_mrt.json')

    BG    = '#0C0C0C'
    FG    = '#D0D0D0'
    DIM   = '#666666'
    CARD  = '#1A1A1A'
    GREEN = '#3DDC84'
    RED   = '#FF5252'
    CYAN  = '#40C4FF'
    AMBER = '#FFB74D'
    PURP  = '#B388FF'

    def __init__(self):
        self.network = None
        self.simulation = None
        self.viz_meta = None
        self._label_mode = 'all'
        self.is_playing = False
        self.time_warp = 1.0
        self.unit_scale = 0.01
        self.real_clock = 0.0
        self._station_names = {}
        self._name_to_id = {}

        self.fig = None
        self.ax_map = None
        self.renderer = None
        self._scat = None
        self._anim = None
        self._w = {}
        self._tele_texts = []
        self._last_real_time = time.time()
        self._table_last_update = 0.0

    def launch(self):
        self.fig = plt.figure(figsize=(14, 9), facecolor=self.BG)
        self.fig.canvas.manager.set_window_title("OceanViz — Transit Simulator")
        self.ax_map = self.fig.add_axes([0.05, 0.05, 0.69, 0.9])
        self.ax_ui = self.fig.add_axes([0, 0, 1, 1], zorder=1000)
        self.ax_ui.axis('off')
        self._setup_ax()
        self._build_panel()
        
        # Pre-allocate an empty scatter for FuncAnimation blitting requirement
        # when running without a loaded network
        self._scat = self.ax_map.scatter([], [])
        
        self._anim = animation.FuncAnimation(
            self.fig, self._tick, init_func=self._init_anim,
            interval=33, blit=False, cache_frame_data=False) #1000 ms / 33 ~ 30 FPS
        plt.show()

    def _setup_ax(self):
        self.ax_map.set_facecolor(self.BG)
        self.ax_map.set_xticks([]); self.ax_map.set_yticks([])
        for s in self.ax_map.spines.values(): s.set_visible(False)
        self.ax_map.set_aspect('equal', adjustable='datalim')

    def _get_anim_artists(self):
        artists = [self._scat] if self._scat is not None else []
        if hasattr(self, '_w') and 'lbl_clock' in self._w:
            artists.append(self._w['lbl_clock'])
        if hasattr(self, '_tele_texts'):
            artists.extend(self._tele_texts)
        if hasattr(self, '_w') and 'lbl_tele' in self._w:
            artists.append(self._w['lbl_tele'])
        return artists

    def _init_anim(self):
        return self._get_anim_artists()

    # ==================================================================
    #  Panel — separator-based layout (no card overlaps)
    # ==================================================================
    def _build_panel(self):
        L = 0.755;  W = 0.235;  R = L + W

        def sep(y):
            self.fig.add_artist(plt.Line2D([L, R], [y, y],
                transform=self.fig.transFigure, color='#333', lw=0.5, clip_on=False))

        def hdr(y, text, color=None):
            self.fig.text(L + 0.005, y, text, color=color or self.FG,
                          fontsize=9, fontweight='bold')
            sep(y - 0.008)

        def lbl(x, y, text, sz=7):
            return self.ax_ui.text(x, y, text, color=self.DIM, fontsize=sz)

        def val(x, y, text, color=None, sz=8, ha='left'):
            return self.ax_ui.text(x, y, text, color=color or self.FG,
                                fontsize=sz, ha=ha, fontweight='bold')

        def btn(rect, text, cb, bg=None, fg=None):
            ax = self.fig.add_axes(rect)
            b = Button(ax, text, color=bg or self.CARD, hovercolor='#333')
            b.label.set_color(fg or self.FG); b.label.set_fontweight('bold')
            b.on_clicked(lambda _: cb())
            return b

        def tbox(rect, initial='', fsz=8):
            ax = self.fig.add_axes(rect, facecolor=self.CARD)
            t = TextBox(ax, '', initial=initial, color=self.CARD, hovercolor='#333333')
            t.text_disp.set_color(self.FG); t.text_disp.set_fontsize(fsz)
            return t

        def sld(rect, lo, hi, v, cb, color=None):
            ax = self.fig.add_axes(rect, facecolor='#181818')
            s = Slider(ax, '', lo, hi, valinit=v, color=color or self.CYAN)
            s.valtext.set_visible(False)
            s.on_changed(cb)
            return s

        hw = W / 2 - 0.005

        # ── TRANSPORT CONTROL ──
        hdr(0.97, 'TRANSPORT CONTROL', self.GREEN)

        self._w['btn_load'] = btn([L, 0.935, W, 0.028], 'LOAD MAP',
                                   self._on_load, '#1A1A1A', self.FG)
        self._w['btn_play'] = btn([L, 0.895, hw, 0.03], '▶  PLAY',
                                   self._on_play, '#1A2A1A', self.GREEN)
        self._w['btn_reset'] = btn([L+hw+0.01, 0.895, hw, 0.03], '⟳  RESET',
                                    self._on_reset, '#2A1A1A', self.RED)

        lbl(L+0.005, 0.875, 'Time Warp')
        self._w['lbl_warp'] = val(R-0.005, 0.875, '1.0x', self.GREEN, ha='right')
        self._w['sld_warp'] = sld([L, 0.855, W, 0.015], 0.1, 100, 1.0,
                                   self._on_warp, self.GREEN)

        self._w['lbl_clock'] = self.ax_ui.text(L+0.005, 0.825,
            'Sim  00:00:00.0\nReal 00:00:00.0',
            color=self.GREEN, fontsize=9, fontweight='bold',
            family='monospace', linespacing=1.5)

        # Labels
        lbl(L+0.005, 0.775, 'Labels')
        self._w['btn_lbl_all'] = btn([L+0.045, 0.765, 0.05, 0.025], 'All', 
                                      lambda: self._on_labels('All'), '#1A2A3A', self.CYAN)
        self._w['btn_lbl_maj'] = btn([L+0.100, 0.765, 0.05, 0.025], 'Major', 
                                      lambda: self._on_labels('Major'), '#1A2A3A', self.CYAN)
        self._w['btn_lbl_non'] = btn([L+0.155, 0.765, 0.05, 0.025], 'None', 
                                      lambda: self._on_labels('None'), '#1A2A3A', self.CYAN)

        lbl(L+0.005, 0.742, 'Weights')
        self._w['btn_wt'] = btn([L+0.045, 0.732, 0.08, 0.025], 'Toggle',
                                 self._on_weights, '#1A2A3A', self.AMBER)
        
        lbl(L+0.13, 0.742, '100u=')
        self._w['txt_scale'] = tbox([L+0.165, 0.732, 0.035, 0.025], f'{100*self.unit_scale:.1f}', 8)
        lbl(L+0.205, 0.742, 'km')
        self._w['txt_scale'].on_submit(self._on_scale)

        # ── PATHFINDING ──
        sep(0.70)
        hdr(0.685, 'PATHFINDING', self.CYAN)

        lbl(L+0.005, 0.655, 'From')
        lbl(L+0.125, 0.655, 'To')
        self._w['btn_src'] = btn([L, 0.625, 0.11, 0.025], 'Jurong East', 
                                  lambda: self._show_popup('src'), '#162233', self.FG)
        self._w['btn_dst'] = btn([L+0.125, 0.625, 0.11, 0.025], 'Changi Airport', 
                                  lambda: self._show_popup('dst'), '#162233', self.FG)

        self._w['btn_find'] = btn([L, 0.590, hw, 0.03], '⚡ Find',
                                   self._on_find_path, '#162233', self.CYAN)
        self._w['btn_clear'] = btn([L+hw+0.01, 0.590, hw, 0.03], 'Clear',
                                    self._on_clear_path)

        self._w['lbl_path'] = self.fig.text(L+0.005, 0.565, '',
                                             color=self.CYAN, fontsize=5.5)

        # ── TRAIN SPAWNER ──
        sep(0.535)
        hdr(0.520, 'TRAIN SPAWNER', self.AMBER)

        lbl(L+0.005, 0.495, 'Line')
        self._w['btn_line'] = btn([L, 0.473, 0.09, 0.02], 'CCL', 
                                  lambda: self._show_popup('line'), '#162233', self.FG)
        lbl(L+0.10, 0.495, 'Seg')
        self._w['txt_seg'] = tbox([L+0.10, 0.473, 0.05, 0.02], '0', 7)
        lbl(L+0.16, 0.495, 'N')
        self._w['txt_count'] = tbox([L+0.16, 0.473, 0.035, 0.02], '1', 7)

        lbl(L+0.005, 0.445, 'Speed')
        self._w['lbl_spd'] = val(R-0.005, 0.445, '0.5 u/s', self.AMBER, 7, 'right')
        self._w['sld_speed'] = sld([L, 0.425, W, 0.015], 0.05, 2.0, 0.5,
                                    self._on_speed, self.AMBER)

        lbl(L+0.005, 0.405, 'Accel')
        self._w['lbl_acc'] = val(R-0.005, 0.405, '0.1 u/s²', self.AMBER, 7, 'right')
        self._w['sld_accel'] = sld([L, 0.385, W, 0.015], 0.01, 1.0, 0.1,
                                    self._on_accel, self.AMBER)

        self._w['btn_spawn'] = btn([L, 0.345, W, 0.03], '⚡ SPAWN TRAINS',
                                    self._on_spawn, '#2A2211', self.AMBER)

        # ── TELEMETRY ──
        sep(0.315)
        hdr(0.300, 'LIVE TELEMETRY', self.PURP)

        self.fig.text(L+0.005, 0.280,
                      f"{'#':>2} {'Line':<5} {'State':<9} {'km/h':>6} {'Prog':>5}",
                      color='#555', fontsize=6.5, family='monospace')

        for i in range(16):
            y = 0.265 - i * 0.016
            txt = self.ax_ui.text(L+0.005, y, '', color='#BBB',
                                fontsize=6.5, family='monospace')
            self._tele_texts.append(txt)

    # ==================================================================
    def _load_map(self, filepath):
        try:
            self.network, self.viz_meta = load_network(filepath)
        except Exception as e:
            print(f"[Load Error] {e}"); return

        G = self.network.get_graph()
        self._station_names = {}
        self._name_to_id = {}
        for nid in G.nodes():
            name = G.nodes[nid].get('name', '')
            self._station_names[nid] = name
            if name:
                self._name_to_id[name.lower().strip()] = nid

        self.simulation = TransitSimulation(self.network, dt=0.02)
        self._redraw_map()
        if self.network.get_line_count() > 0:
            line_codes = self.network.get_line_names()
            self._name_to_id['line'] = line_codes
            self._w['btn_line'].label.set_text(line_codes[0])
        
        # Populate physics readout math from Scale explicitly
        self._on_speed(self._w['sld_speed'].val)
        self._on_accel(self._w['sld_accel'].val)
        self.is_playing = False
        self.fig.canvas.draw_idle()
        print(f"[Loaded] {self.network.get_station_count()} stations, "
              f"{self.network.get_line_count()} lines")

    def _redraw_map(self):
        anim_running = False
        if self._anim and self._anim.event_source:
            anim_running = getattr(self, 'is_playing', False)
            self._anim.event_source.stop()
            self._anim = None

        self.ax_map.clear(); self._setup_ax()
        self.renderer = TrackRenderer(self.ax_map, self.network, self.viz_meta, unit_scale=self.unit_scale)
        self.renderer.draw_network(label_mode=self._label_mode,
                                    show_weights=getattr(self, '_show_weights', False),
                                    weight_unit=getattr(self, '_weight_unit', 'km'),
                                    scale_factor=getattr(self, '_scale_factor', 10.0))
        
        self._scat = self.ax_map.scatter([], [], s=50, zorder=200, edgecolors='#FFFFFF', linewidths=1.5)
        self._update_agents()
        self.fig.canvas.draw_idle()

        # Re-attach animation strictly with double-buffered draw
        import matplotlib.animation as animation
        self._anim = animation.FuncAnimation(
            self.fig, self._tick, init_func=self._init_anim,
            interval=33, blit=False, cache_frame_data=False)

        if not anim_running:
            self._anim.event_source.stop()

    def _resolve(self, text):
        text = text.strip()
        try:
            nid = int(text)
            if nid in self._station_names: return nid
        except ValueError: pass
        key = text.lower()
        if key in self._name_to_id: return self._name_to_id[key]
        for name, nid in self._name_to_id.items():
            if key in name: return nid
        return -1

    # ==================================================================
    def _on_load(self): 
        import tkinter as tk
        from tkinter import filedialog
        
        # Open a native OS file dialog box
        root = tk.Tk()
        root.withdraw() # Hide the main root window
        root.attributes('-topmost', True) # Bring dialog to front
        
        filepath = filedialog.askopenfilename(
            title="Select Transit Network JSON",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if filepath:
            self._load_map(filepath)

    def _on_play(self):
        self.is_playing = not self.is_playing
        self._last_real_time = time.time()
        self._w['btn_play'].label.set_text('|| PAUSE' if self.is_playing else '▶  PLAY')
        self.fig.canvas.draw_idle()

    def _on_reset(self):
        self.is_playing = False
        self._w['btn_play'].label.set_text('▶  PLAY')
        if self.simulation: self.simulation.reset()
        self.real_clock = 0.0
        self._update_clock(); self._update_agents(); self._update_tele()
        self.fig.canvas.draw_idle()

    def _on_warp(self, v):
        self.time_warp = v
        self._w['lbl_warp'].set_text(f'{v:.1f}x')

    def _on_speed(self, val):
        speed = float(val)
        kmh = speed * 3600.0 * self.unit_scale
        self._w['lbl_spd'].set_text(f'{speed:.1f} u/s ({kmh:.0f} km/h)')

    def _on_accel(self, val):
        a = float(val)
        kmh_s = a * 3600.0 * self.unit_scale
        self._w['lbl_acc'].set_text(f'{a:.1f} u/s² ({kmh_s:.0f} km/h/s)')

    def _on_scale(self, text):
        try:
            val = float(text)
            if val <= 0: raise ValueError
            self.unit_scale = val / 100.0
            self._on_speed(self._w['sld_speed'].val)
            self._on_accel(self._w['sld_accel'].val)
            self._redraw_map()
        except ValueError:
            self._w['txt_scale'].set_val(f'{100*self.unit_scale:.1f}')

    def _on_labels(self, label):
        if label in ['All', 'Major', 'None']:
            self._label_mode = label.lower()
        if self.renderer:
            self._redraw_map()
            self._scat = self.ax_map.scatter([], [], s=50, zorder=200,
                                             edgecolors='#FFFFFF', linewidths=1.5)
            self._update_agents(); self.fig.canvas.draw_idle()

    def _on_weights(self):
        self._show_weights = not getattr(self, '_show_weights', False)
        if self.renderer:
            self._redraw_map()

    def _on_spawn(self):
        if not self.simulation:
            print("[Spawn] No simulation active — press PLAY first.")
            return
        code = self._w['btn_line'].label.get_text().strip()
        try: seg = int(self._w['txt_seg'].text.strip())
        except: seg = 0
        try: n = int(self._w['txt_count'].text.strip())
        except: n = 1
        spd = self._w['sld_speed'].val; acc = self._w['sld_accel'].val
        print(f"[Spawn] Line={code}, Seg={seg}, N={n}, Speed={spd:.2f}, Accel={acc:.2f}")
        for _ in range(n):
            try:
                self.simulation.add_agent(line_code=code, speed=spd,
                    accel_rate=acc, brake_rate=acc*1.5, start_segment=seg)
            except Exception as e:
                print(f"[Spawn] {e}"); break
        self._update_agents(); self._update_tele()
        self.fig.canvas.draw_idle()

    def _on_find_path(self):
        if not self.network: return
        src = self._resolve(self._w['btn_src'].label.get_text())
        dst = self._resolve(self._w['btn_dst'].label.get_text())
        if src < 0 or dst < 0:
            self._w['lbl_path'].set_text('Station not found!')
            self.fig.canvas.draw_idle(); return
        path, dist = self.network.shortest_path(src, dst)
        if path:
            import matplotlib.lines as mlines
            import matplotlib.collections as mcoll
            import matplotlib.text as mtext
            # Dim all existing interactive map elements
            for child in self.ax_map.get_children():
                if isinstance(child, (mlines.Line2D, mcoll.PathCollection, mtext.Text)):
                    a = child.get_alpha()
                    child.set_alpha((a if a is not None else 1.0) * 0.15)
                    
            names = [self._station_names.get(n, str(n)) for n in path]
            self._w['lbl_path'].set_text(
                f"{' → '.join(names)}\n({len(path)} stops, d={dist:.1f})")
            t = np.linspace(0, 1, 100); S = TrackRenderer.COORD_SCALE
            for i in range(len(path)-1):
                eidx = self.network.find_edge_index(path[i], path[i+1])
                if eidx:
                    spl = self.network.get_spline(eidx)
                    if spl:
                        pts = spl.evaluate(t) * S
                        # Draw high-contrast solid cyan path core without the muddy glow
                        self.ax_map.plot(pts[:,0], pts[:,1], color='#00FFFF',
                                         lw=6, alpha=1.0, zorder=100)
            
            # Re-draw the stations and labels for the path overlay
            G = self.network.get_graph()
            pos_x, pos_y = [], []
            offsets = self.viz_meta.get('label_offsets', {})
            def_offset = self.viz_meta.get('default_label_offset', [0.6, -0.6])
            
            for n in path:
                p = np.array(G.nodes[n].get('pos', [0, 0])) * S
                pos_x.append(p[0])
                pos_y.append(p[1])
                
            self.ax_map.scatter(pos_x, pos_y, s=35, marker='o', 
                                facecolors='#FFFFFF', edgecolors='#00FFFF', 
                                linewidths=1.5, zorder=110, alpha=1.0)
                                
            for i, n in enumerate(path):
                name = self._station_names.get(n, '')
                if not name: continue
                offset = offsets.get(n, def_offset)
                dx, dy = offset[0] * S, offset[1] * S
                
                ha = 'left' if dx > 0.05 else ('right' if dx < -0.05 else 'center')
                va = 'bottom' if dy > 0.05 else ('top' if dy < -0.05 else 'center')
                
                self.ax_map.text(pos_x[i] + dx, pos_y[i] + dy, name, 
                                 color='#FFFFFF', fontsize=7, fontweight='bold', 
                                 ha=ha, va=va, zorder=120)
                                 
            self.fig.canvas.draw_idle()
        else:
            self._w['lbl_path'].set_text('No path found!')
            self.fig.canvas.draw_idle()

    def _on_clear_path(self):
        self._w['lbl_path'].set_text('')
        if self.renderer:
            self._redraw_map()
            self._scat = self.ax_map.scatter([], [], s=50, zorder=200,
                                             edgecolors='#FFF', linewidths=1.5)
            self._update_agents(); self.fig.canvas.draw_idle()

    # ==================================================================
    def _tick(self, frame):
        if not self.is_playing or not self.simulation:
            return self._get_anim_artists()
        now = time.time()
        real_dt = min(now - self._last_real_time, 0.1)
        self._last_real_time = now; self.real_clock += real_dt
        total = real_dt * self.time_warp
        n = max(1, int(np.ceil(total / (1/30))))
        dt = total / n; rdt = real_dt / n
        for _ in range(n):
            self.simulation.step(sim_dt=dt, real_dt=rdt)
        self._update_agents()
        if (self.real_clock - self._table_last_update) >= 0.2:
            self._update_clock(); self._update_tele()
            self._table_last_update = self.real_clock
            self.fig.canvas.draw_idle()
        return self._get_anim_artists()

    def _update_agents(self):
        if not self.simulation or not self._scat: return
        pos = self.simulation.get_positions()
        col = self.simulation.get_colors()
        if len(pos) > 0:
            pos = pos * TrackRenderer.COORD_SCALE
            self._scat.set_offsets(pos); self._scat.set_facecolor(col)
        else:
            self._scat.set_offsets(np.empty((0, 2)))

    def _update_clock(self):
        if not self.simulation: return
        def f(t):
            return f'{int(t//3600):02d}:{int((t%3600)//60):02d}:{t%60:04.1f}'
        st = self.simulation.get_clock()
        self._w['lbl_clock'].set_text(f'Sim  {f(st)}\nReal {f(self.real_clock)}')

    # ==================================================================
    #  Station Selection Popup
    # ==================================================================
    def _show_popup(self, target, page=0):
        """Show overlay popup restricted to the UI Panel bounding box."""
        if hasattr(self, '_popup_bg') and self._popup_bg:
            self.fig.canvas.mpl_disconnect(self._cid_click)
            self._popup_bg.remove()
            
        self._popup_target = target
        
        # Hide underlying panel widgets to prevent overlap rendering and block interaction
        if not hasattr(self, '_popup_hidden_axes') or not self._popup_hidden_axes:
            self._popup_hidden_axes = []
            for w in self._w.values():
                if hasattr(w, 'ax') and w.ax.get_visible():
                    w.ax.set_visible(False)
                    self._popup_hidden_axes.append(w.ax)
            
        L = 0.755; W = 0.235
        # Create an opaque overlay across ONLY the side panel area
        self._popup_bg = self.fig.add_axes([L, 0, W, 1], facecolor='#09090E')
        self._popup_bg.patch.set_alpha(0.98)
        self._popup_bg.set_zorder(1000)
        self._popup_bg.set_xticks([]); self._popup_bg.set_yticks([])
        
        if target in ('src', 'dst'):
            target_name = "SOURCE" if target == 'src' else "DESTINATION"
            title = f'SELECT {target_name}'
            names = sorted([n for n in set(self._station_names.values()) if n])
        elif target == 'line':
            title = 'SELECT LINE'
            names = sorted(self.network.get_line_names())
            
        self._popup_bg.text(0.5, 0.96, title, color=self.CYAN, 
                            fontsize=12, fontweight='bold', ha='center', va='center')
        
        ITEMS_PER_PAGE = 25
        start_idx = page * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, len(names))
        view_names = names[start_idx:end_idx]
        
        self._popup_hitboxes = []
        self._popup_page = page
        self._popup_max_page = (len(names) - 1) // ITEMS_PER_PAGE
        
        # Render a single column natively constrained to the panel bounds to prevent clipping!
        for i, name in enumerate(view_names):
            x = 0.5
            y = 0.91 - i * 0.031
            
            txt = self._popup_bg.text(x, y, name, color=self.FG, fontsize=9, ha='center', va='center',
                                      bbox=dict(facecolor='#1A1A22', edgecolor='#444455', boxstyle='round,pad=0.4'))
            self._popup_hitboxes.append((txt, name))
            
        self._popup_bg.text(0.5, 0.12, '[ CLICK OUTSIDE TO CANCEL ]', color=self.DIM, 
                            fontsize=8, ha='center', va='center')
                            
        if self._popup_page > 0:
            t_prev = self._popup_bg.text(0.5, 0.07, '▲ PREVIOUS PAGE ▲', color=self.AMBER, fontsize=10, fontweight='bold',
                                         ha='center', va='center', bbox=dict(facecolor='#221100', boxstyle='round,pad=0.5'))
            self._popup_hitboxes.append((t_prev, '__PREV__'))
            
        if self._popup_page < self._popup_max_page:
            t_next = self._popup_bg.text(0.5, 0.03, '▼ NEXT PAGE ▼', color=self.AMBER, fontsize=10, fontweight='bold',
                                         ha='center', va='center', bbox=dict(facecolor='#221100', boxstyle='round,pad=0.5'))
            self._popup_hitboxes.append((t_next, '__NEXT__'))
                            
        self._cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_popup_click)
        self.fig.canvas.draw_idle()

    def _on_popup_click(self, event):
        renderer = self.fig.canvas.get_renderer()
        for txt, name in self._popup_hitboxes:
            box = txt.get_window_extent(renderer)
            if box.contains(event.x, event.y):
                if name == '__PREV__':
                    self._show_popup(self._popup_target, self._popup_page - 1)
                    return
                elif name == '__NEXT__':
                    self._show_popup(self._popup_target, self._popup_page + 1)
                    return
                else:
                    if self._popup_target == 'src':
                        self._w['btn_src'].label.set_text(name)
                    elif self._popup_target == 'dst':
                        self._w['btn_dst'].label.set_text(name)
                    elif self._popup_target == 'line':
                        self._w['btn_line'].label.set_text(name)
                    self._hide_popup()
                    return
        
        # If click wasn't on any valid target, cancel the popup globally
        self._hide_popup()

    def _hide_popup(self):
        if hasattr(self, '_cid_click'):
            self.fig.canvas.mpl_disconnect(self._cid_click)
        if hasattr(self, '_popup_bg') and self._popup_bg:
            self._popup_bg.remove()
            self._popup_bg = None
            
        # Restore underlying widgets
        if hasattr(self, '_popup_hidden_axes') and self._popup_hidden_axes:
            for ax in self._popup_hidden_axes:
                ax.set_visible(True)
            self._popup_hidden_axes = []
            
        self._popup_hitboxes = []
        self.fig.canvas.draw_idle()

    def _update_tele(self):
        if not self.simulation: return
        agents = self.simulation.get_agents()
        for i, txt in enumerate(self._tele_texts):
            if i < len(agents):
                a = agents[i]
                st = a.get_agent_state()
                kin = a.get_state()
                v = kin.velocity if kin else 0.0
                kmh = v * self.unit_scale * 3600.0
                p = a.get_progress()
                txt.set_text(f"{i+1:>2} {a._line_code:<5} {st:<9} {kmh:6.0f} {p*100:4.0f}%")
            else:
                txt.set_text('')


if __name__ == '__main__':
    app = TransitSimulatorApp()
    app.launch()

