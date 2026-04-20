"""
transit_network.py — Domain wrapper around NetworkX for transit systems.

Port of: +OceanMath/+Transit/TransitNetwork.m (795 lines)

Adds transit vocabulary (stations, routes, splines, lines) on top of
a NetworkX graph. NetworkX handles all graph algorithms (Dijkstra, adjacency).

The MATLAB DigraphCus (397 lines), Node (48 lines), Edge (51 lines), and
Dijkstra (109 lines) are ALL replaced by NetworkX — saving ~600 lines.

Usage:
    net = TransitNetwork()
    net.add_line('NSL', 'North-South Line', (0.85, 0.15, 0.15))
    id1 = net.add_station('Orchard', np.array([3, 7]))
    id2 = net.add_station('Dhoby Ghaut', np.array([4, 6]))
    net.add_route_to_line('NSL', id1, id2, spline)
    path, dist = net.shortest_path(id1, id2)
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import networkx as nx

from ocean.topology.transit_line import TransitLine
from ocean.geometry.parametric_curve import ParametricCurve


class TransitNetwork:
    """Domain wrapper around nx.Graph for transit systems."""

    def __init__(self, directed: bool = False):
        """Construct an empty transit network.

        Parameters
        ----------
        directed : bool
            If True, use nx.DiGraph. Default False (undirected).
        """
        self._graph: nx.Graph = nx.MultiDiGraph() if directed else nx.MultiGraph()
        self._lines: dict[str, TransitLine] = {}
        self._splines: dict[int, ParametricCurve] = {}
        self._node_counter: int = 0   # 1-indexed auto-increment
        self._edge_counter: int = 0   # 1-indexed auto-increment

        # Lazy cache for station-to-line lookups
        self._station_line_cache: dict[int, list[str]] = {}
        self._cache_valid: bool = False

    # ==================================================================
    #  Station Operations
    # ==================================================================

    def reserve_stations(self, n: int) -> None:
        """Pre-hint for bulk loading. NetworkX handles allocation natively."""
        pass  # NetworkX does not need pre-allocation

    def add_station(self, name: str, position: np.ndarray = None,
                    metadata: dict = None) -> int:
        """Add a station to the transit network. Returns 1-indexed ID."""
        if position is None:
            position = np.array([0.0, 0.0])
        if metadata is None:
            metadata = {}
        self._node_counter += 1
        sid = self._node_counter
        self._graph.add_node(sid, name=name, pos=np.asarray(position, dtype=float),
                             metadata=metadata)
        return sid

    def add_stations(self, station_list: list) -> list[int]:
        """Bulk-add stations from a list of (name, position) tuples.

        Parameters
        ----------
        station_list : list of (name, position) pairs

        Returns
        -------
        list of assigned 1-indexed IDs
        """
        ids = []
        for name, pos in station_list:
            ids.append(self.add_station(name, np.asarray(pos)))
        return ids

    def get_station(self, sid: int) -> dict:
        """Retrieve a station's attribute dict by ID.

        Returns dict with keys: 'name', 'pos', 'metadata'
        """
        if sid not in self._graph:
            raise KeyError(f"Station ID {sid} does not exist.")
        return dict(self._graph.nodes[sid])

    def get_station_by_name(self, name: str) -> Optional[int]:
        """Find a station ID by name. Returns None if not found."""
        for nid, data in self._graph.nodes(data=True):
            if data.get('name') == name:
                return nid
        return None

    def get_station_count(self) -> int:
        """Number of stations in the network."""
        return self._graph.number_of_nodes()

    def get_station_names(self) -> list[str]:
        """Return list of all station names, ordered by node ID."""
        nodes_sorted = sorted(self._graph.nodes())
        return [self._graph.nodes[n].get('name', '') for n in nodes_sorted]

    def update_station_name(self, sid: int, new_name: str) -> None:
        """Rename a station."""
        if sid not in self._graph:
            raise KeyError(f"Station ID {sid} does not exist.")
        self._graph.nodes[sid]['name'] = new_name

    def update_station_position(self, sid: int, new_pos: np.ndarray) -> None:
        """Move a station to a new position."""
        if sid not in self._graph:
            raise KeyError(f"Station ID {sid} does not exist.")
        self._graph.nodes[sid]['pos'] = np.asarray(new_pos, dtype=float)

    def get_station_positions(self) -> np.ndarray:
        """Return (V, 2) array of station positions, ordered by node ID."""
        nodes_sorted = sorted(self._graph.nodes())
        if not nodes_sorted:
            return np.empty((0, 2))
        positions = []
        for n in nodes_sorted:
            pos = self._graph.nodes[n].get('pos', np.array([0.0, 0.0]))
            positions.append(pos)
        return np.array(positions)

    # ==================================================================
    #  Line Management
    # ==================================================================

    def add_line(self, code: str, name: str, color: tuple) -> TransitLine:
        """Register a new transit line."""
        line = TransitLine(code, name, color)
        self._lines[code] = line
        self._invalidate_cache()
        return line

    def get_line(self, code: str) -> TransitLine:
        """Retrieve a TransitLine by its code."""
        if code not in self._lines:
            raise KeyError(f'Line "{code}" not registered.')
        return self._lines[code]

    def get_line_names(self) -> list[str]:
        """Return all registered line codes."""
        return list(self._lines.keys())

    def get_lines_for_station(self, station_id: int) -> list[str]:
        """Return line codes serving this station. Uses O(1) cache."""
        if not self._cache_valid:
            self._rebuild_line_cache()
        return self._station_line_cache.get(station_id, [])

    def get_line_count(self) -> int:
        """Number of registered lines."""
        return len(self._lines)

    def get_interchanges(self) -> list[int]:
        """Return node IDs of interchange stations (served by 2+ lines)."""
        if not self._cache_valid:
            self._rebuild_line_cache()
        return [nid for nid, codes in self._station_line_cache.items()
                if len(codes) >= 2]

    def get_terminals(self) -> list[int]:
        """Return node IDs of terminal stations (degree 1 in graph)."""
        return [n for n in self._graph.nodes() if self._graph.degree(n) == 1]

    def classify_station(self, node_id: int) -> str:
        """Classify a station as 'interchange', 'terminal', or 'standard'."""
        if not self._cache_valid:
            self._rebuild_line_cache()
        codes = self._station_line_cache.get(node_id, [])
        if len(codes) >= 2:
            return 'interchange'
        if self._graph.degree(node_id) == 1:
            return 'terminal'
        return 'standard'

    def build_line(self, code: str, name: str, color: tuple,
                   station_ids: list[int],
                   splines: list[ParametricCurve] = None) -> TransitLine:
        """One-call line builder: create + register + connect.

        Parameters
        ----------
        code : str
            Line code (e.g. 'NSL')
        name : str
            Full name
        color : tuple
            RGB color
        station_ids : list[int]
            Ordered station IDs (must already exist via add_station)
        splines : list[ParametricCurve], optional
            One spline per segment. If None, auto-generates straight lines.
        """
        if len(station_ids) < 2:
            raise ValueError("build_line requires at least 2 stations.")

        line = self.add_line(code, name, color)
        n_segs = len(station_ids) - 1

        if splines is not None and len(splines) != n_segs:
            raise ValueError(f"Expected {n_segs} splines, got {len(splines)}.")

        for i in range(n_segs):
            src, tgt = station_ids[i], station_ids[i + 1]
            if splines is not None:
                spl = splines[i]
            else:
                # Auto-generate straight line between station positions
                p_src = self._graph.nodes[src]['pos']
                p_tgt = self._graph.nodes[tgt]['pos']
                from ocean.geometry.catmull_rom_spline import CatmullRomSpline
                spl = CatmullRomSpline(np.vstack([p_src, p_tgt]))
            self.add_route_to_line(code, src, tgt, spl)

        return line

    # ==================================================================
    #  Route Operations
    # ==================================================================

    def add_route(self, src_id: int, tgt_id: int,
                  spline: ParametricCurve, label: str = '') -> int:
        """Connect two stations with a spline-shaped route.

        Weight is auto-computed from the spline's total arc length.
        Returns the 1-indexed edge index.
        """
        weight = spline.total_length()
        self._edge_counter += 1
        idx = self._edge_counter

        self._graph.add_edge(src_id, tgt_id, weight=weight, label=label,
                             edge_idx=idx,
                             draw_src=src_id, draw_tgt=tgt_id)
        self._splines[idx] = spline
        return idx

    def add_route_to_line(self, line_code: str, src_id: int, tgt_id: int,
                          spline: ParametricCurve) -> int:
        """Add a spline route AND register it on a line.

        This does three things:
          1. Adds the edge to the graph (weight from arc length)
          2. Stores the spline for rendering
          3. Registers both stations on the TransitLine
        """
        idx = self.add_route(src_id, tgt_id, spline, label=line_code)
        line = self.get_line(line_code)
        line.register_station(src_id)
        line.register_station(tgt_id)
        self._invalidate_cache()
        return idx

    def add_simple_route(self, src_id: int, tgt_id: int,
                         weight: float, label: str = '') -> int:
        """Connect two stations without a spline (weight-only)."""
        self._edge_counter += 1
        idx = self._edge_counter
        self._graph.add_edge(src_id, tgt_id, weight=weight, label=label,
                             edge_idx=idx)
        return idx

    def get_spline(self, edge_idx: int) -> Optional[ParametricCurve]:
        """Return the ParametricCurve for an edge index, or None."""
        return self._splines.get(edge_idx)

    def update_route_spline(self, edge_idx: int,
                            spline: ParametricCurve) -> None:
        """Replace a route's spline and update edge weight."""
        self._splines[edge_idx] = spline
        new_weight = spline.total_length()
        # Find the edge with this index and update weight
        for u, v, k, data in self._graph.edges(keys=True, data=True):
            if data.get('edge_idx') == edge_idx:
                self._graph[u][v][k]['weight'] = new_weight
                return

    def get_route_count(self) -> int:
        """Number of routes (edges) in the network."""
        return self._graph.number_of_edges()

    def remove_route(self, src_id: int, tgt_id: int) -> None:
        """Remove a route (edge + spline) between two stations."""
        if self._graph.has_edge(src_id, tgt_id):
            edges = list(self._graph[src_id][tgt_id].values())
            for d in edges:
                edge_idx = d.get('edge_idx')
                if edge_idx and edge_idx in self._splines:
                    del self._splines[edge_idx]
            while self._graph.has_edge(src_id, tgt_id):
                self._graph.remove_edge(src_id, tgt_id)

    def find_edge_index(self, src_id: int, tgt_id: int) -> int:
        """Return the first edge index for the given station pair."""
        if self._graph.has_edge(src_id, tgt_id):
            return list(self._graph[src_id][tgt_id].values())[0].get('edge_idx', 0)
        return 0

    # ==================================================================
    #  Graph Queries (powered by NetworkX)
    # ==================================================================

    def get_graph(self) -> nx.Graph:
        """Return the inner NetworkX graph."""
        return self._graph

    def shortest_path(self, src_id: int, tgt_id: int) -> tuple:
        """Shortest path via Dijkstra. Returns (path, distance).

        path : list[int] — node IDs from src to tgt, or [] if no path.
        distance : float — total weight, or inf if no path.
        """
        try:
            path = nx.dijkstra_path(self._graph, src_id, tgt_id, weight='weight')
            dist = nx.dijkstra_path_length(self._graph, src_id, tgt_id, weight='weight')
            return path, dist
        except nx.NetworkXNoPath:
            return [], float('inf')

    def get_adjacency_matrix(self) -> np.ndarray:
        """Return the (V x V) weight matrix."""
        nodes_sorted = sorted(self._graph.nodes())
        return nx.to_numpy_array(self._graph, nodelist=nodes_sorted, weight='weight')

    # ==================================================================
    #  Interchange & Terminal Queries
    # ==================================================================

    def get_interchanges(self) -> list[int]:
        """Station IDs that belong to 2+ lines."""
        station_line_counts: dict[int, int] = {}
        for code, line in self._lines.items():
            for sid in line.get_route():
                station_line_counts[sid] = station_line_counts.get(sid, 0) + 1
        return sorted([sid for sid, count in station_line_counts.items() if count >= 2])

    def get_station_lines(self, station_id: int) -> list[str]:
        """Line codes passing through a station."""
        codes = []
        for code, line in self._lines.items():
            if line.has_station(station_id):
                codes.append(code)
        return codes

    def is_interchange(self, station_id: int) -> bool:
        """True if station belongs to 2+ lines."""
        return len(self.get_station_lines(station_id)) >= 2

    def get_terminals(self) -> list[int]:
        """Terminal stations: degree-1 nodes (leaf nodes in the graph)."""
        terminals = []
        for n in sorted(self._graph.nodes()):
            if self._graph.degree(n) == 1:
                terminals.append(n)
        return terminals

    # ==================================================================
    #  Analytics
    # ==================================================================

    def total_network_length(self) -> float:
        """Sum of all edge weights."""
        return sum(d.get('weight', 0.0) for _, _, d in self._graph.edges(data=True))

    def average_route_length(self) -> float:
        """Average edge weight."""
        n = self._graph.number_of_edges()
        if n == 0:
            return 0.0
        return self.total_network_length() / n

    # ==================================================================
    #  Route Reconciliation
    # ==================================================================

    def reconcile_line_routes(self) -> None:
        """Reorder each line's route to match actual edge topology.

        The user may have registered stations in arbitrary click order.
        This method walks the actual graph edges to produce the correct
        physical traversal sequence for each line.

        Algorithm:
          1. For each line, build local adjacency among registered stations
          2. Find degree-1 nodes (terminals) in the local subgraph
          3. Walk from one terminal, collecting stations in order
          4. Circular lines (no degree-1): start from first station
        """
        G = self._graph

        for code, line in self._lines.items():
            S = line.get_route()  # current (possibly unordered) station set
            n = len(S)
            if n < 2:
                continue

            s_set = set(S)  # fast membership test

            # Build local adjacency within S
            # local_adj[i] = [indices into S that are neighbors of S[i]]
            local_adj: list[list[int]] = [[] for _ in range(n)]

            for a in range(n):
                for nb_id in G.neighbors(S[a]):
                    if nb_id in s_set:
                        # Find index of nb_id in S
                        try:
                            idx = S.index(nb_id)
                            local_adj[a].append(idx)
                        except ValueError:
                            pass

            # Find degree-1 stations (terminals) in local subgraph
            local_deg = [len(adj) for adj in local_adj]
            terminals = [i for i, d in enumerate(local_deg) if d == 1]

            if not terminals:
                start_idx = 0  # circular line
            else:
                start_idx = terminals[0]

            # Walk the chain
            visited = [False] * n
            ordered = []
            current = start_idx

            for _ in range(n):
                ordered.append(S[current])
                visited[current] = True
                next_idx = None
                for nb in local_adj[current]:
                    if not visited[nb]:
                        next_idx = nb
                        break
                if next_idx is None:
                    break
                current = next_idx

            # Warn about orphaned stations
            if len(ordered) < n:
                orphans = [S[i] for i in range(n) if not visited[i]]
                orphan_names = []
                for oid in orphans:
                    data = self._graph.nodes.get(oid, {})
                    orphan_names.append(data.get('name', f'#{oid}'))
                print(f'[Reconcile] Line "{code}": {len(ordered)}/{n} '
                      f'stations connected. Orphaned: {", ".join(orphan_names)}')

            line.set_route(ordered)

    # ==================================================================
    #  Visualization Export
    # ==================================================================

    def to_viz_data(self) -> dict:
        """Export struct for TransitGraphStrategy renderer.

        Returns dict with keys:
            positions      : (V, 2) ndarray
            node_names     : list[str]
            splines        : list[ParametricCurve or None], indexed by edge_idx
            edge_pairs     : (E, 2) ndarray of [src, tgt]
            edge_labels    : list[str]
            lines          : dict of line structs
            interchange_ids: list[int]
            terminal_ids   : list[int]
        """
        nodes_sorted = sorted(self._graph.nodes())

        data = {
            'positions': self.get_station_positions(),
            'node_names': self.get_station_names(),
        }

        # Collect splines, edge pairs, and labels ordered by edge_idx
        edge_list = []
        for u, v, d in self._graph.edges(data=True):
            edge_list.append((d.get('edge_idx', 0), u, v, d.get('label', '')))
        edge_list.sort(key=lambda x: x[0])  # sort by edge_idx

        n_edges = len(edge_list)
        spline_list = [None] * n_edges
        edge_pairs = np.zeros((n_edges, 2), dtype=int)
        edge_labels = [''] * n_edges

        for i, (eidx, u, v, lbl) in enumerate(edge_list):
            spline_list[i] = self._splines.get(eidx)
            edge_pairs[i] = [u, v]
            edge_labels[i] = lbl

        data['splines'] = spline_list
        data['edge_pairs'] = edge_pairs
        data['edge_labels'] = edge_labels

        # Line info
        data['lines'] = {code: line.to_struct() for code, line in self._lines.items()}
        data['interchange_ids'] = self.get_interchanges()
        data['terminal_ids'] = self.get_terminals()

        return data

    # ==================================================================
    #  Serialization
    # ==================================================================

    def to_struct(self) -> dict:
        """Export full network for JSON saving."""
        nodes_sorted = sorted(self._graph.nodes())
        node_data = []
        for nid in nodes_sorted:
            d = self._graph.nodes[nid]
            node_data.append({
                'Id': nid,
                'Name': d.get('name', ''),
                'Position': d.get('pos', np.array([0, 0])).tolist(),
                'Metadata': d.get('metadata', {}),
            })

        edge_data = []
        spline_data = []
        for u, v, d in self._graph.edges(data=True):
            eidx = d.get('edge_idx', 0)
            edge_data.append({
                'Source': u,
                'Target': v,
                'Weight': d.get('weight', 0.0),
                'Label': d.get('label', ''),
            })
            spl = self._splines.get(eidx)
            if spl is not None:
                spline_data.append(spl.get_control_points().tolist())
            else:
                spline_data.append(None)

        line_data = [line.to_struct() for line in self._lines.values()]

        return {
            'Nodes': node_data,
            'Edges': edge_data,
            'SplineData': spline_data,
            'LineData': line_data,
        }

    # ==================================================================
    #  Display
    # ==================================================================

    def describe(self) -> str:
        """Print a human-readable network summary."""
        desc = "TransitNetwork:\n"
        desc += f"  Stations: {self.get_station_count()}\n"
        desc += f"  Routes:   {self.get_route_count()}\n"
        desc += f"  Lines:    {self.get_line_count()}\n"
        if self.get_route_count() > 0:
            desc += f"  Total Length: {self.total_network_length():.4f}\n"
            desc += f"  Avg Route:   {self.average_route_length():.4f}\n"

        all_names = self.get_station_names()
        nodes_sorted = sorted(self._graph.nodes())
        id_to_idx = {nid: i for i, nid in enumerate(nodes_sorted)}

        for code, line in self._lines.items():
            route = line.get_route()
            route_names = [all_names[id_to_idx[r]] for r in route if r in id_to_idx]
            desc += f"  [{line.code}] {line.name} ({line.get_station_count()} stations)\n"
            desc += f"    Route: {' -> '.join(route_names)}\n"

        ix_ids = self.get_interchanges()
        if ix_ids:
            ix_names = [all_names[id_to_idx[i]] for i in ix_ids if i in id_to_idx]
            desc += f"  Interchanges: {', '.join(ix_names)}\n"

        print(desc)
        return desc

    # ==================================================================
    #  Private helpers
    # ==================================================================

    def _invalidate_cache(self) -> None:
        """Mark station-to-line cache as dirty."""
        self._cache_valid = False

    def _rebuild_line_cache(self) -> None:
        """One-time O(L*S) scan to power O(1) lookups."""
        cache: dict[int, list[str]] = {}
        for code, line in self._lines.items():
            for sid in line.get_route():
                if sid not in cache:
                    cache[sid] = []
                if code not in cache[sid]:
                    cache[sid].append(code)
        self._station_line_cache = cache
        self._cache_valid = True
