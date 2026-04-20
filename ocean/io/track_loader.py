"""
track_loader.py — Read and write TransitNetworks to/from JSON.

Port of: +OceanViz/+Designer/+Persistence/DesignerSerializer.m (lines 578-692, 261-407)

Functions:
    load_network(filepath) -> (TransitNetwork, viz_meta)
    save_network(filepath, network, viz_meta)

Reconstructs the pure math TransitNetwork and extracts visual metadata
for the renderer (edge colors, label offsets, etc.).
"""

import json
import logging
import numpy as np

from ocean.topology.transit_network import TransitNetwork
from ocean.geometry.catmull_rom_spline import CatmullRomSpline
from ocean.geometry.bezier_curve import BezierCurve

logger = logging.getLogger(__name__)


def load_network(filepath: str) -> tuple[TransitNetwork, dict]:
    """Load a TransitNetwork from a TrackDesigner JSON file.
    
    Parameters
    ----------
    filepath : str
        Path to the JSON file.
        
    Returns
    -------
    network : TransitNetwork
        Fully reconstructed transit network with stations, splines, and lines.
    viz_meta : dict
        Visual metadata for rendering:
            edge_colors         : dict[int, list[float]]  — {edge_idx: [r,g,b]}
            label_offsets       : dict[int, list[float]]  — {station_id: [dx,dy]}
            default_label_offset: list[float]              — [dx, dy]
            draw_orders         : dict[int, int]           — {edge_idx: z_order}
            legend_pos1         : list[float] or None
            legend_pos2         : list[float] or None
            title_pos           : list[float] or None
            map_title           : str
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    net = TransitNetwork()
    id_map = {}  # json_id -> network_id
    
    viz_meta = {
        'edge_colors': {},
        'label_offsets': {},
        'default_label_offset': data.get('meta', {}).get('defaultLabelOffset', [0.0, 0.3]),
        'draw_orders': {},
        'legend_pos1': data.get('legendPos1'),
        'legend_pos2': data.get('legendPos2'),
        'title_pos': data.get('titlePos'),
        'map_title': data.get('mapTitle', ''),
    }
    
    # 1. Rebuild Stations
    for s in data.get('stations', []):
        pos = np.array(s['position'][:2], dtype=float)  # strip Z if present
        new_id = net.add_station(s['name'], pos)
        
        # Determine mapping key
        if 'id' in s:
            id_map[s['id']] = new_id
        else:
            # Fallback for old JSONs without explicit IDs
            id_map[len(id_map) + 1] = new_id
            
        # Capture label offset
        if 'labelOffset' in s:
            viz_meta['label_offsets'][new_id] = s['labelOffset']
            
    # 2. Rebuild Edges
    for e in data.get('edges', []):
        src_json = e.get('source')
        tgt_json = e.get('target')
        
        if src_json not in id_map or tgt_json not in id_map:
            logger.warning(f"Edge {e.get('id', '?')} references unknown stations {src_json}->{tgt_json}")
            continue
            
        src = id_map[src_json]
        tgt = id_map[tgt_json]
        
        cp = e.get('controlPoints')
        edge_idx = 0
        if cp and len(cp) > 0:
            cp_arr = np.array(cp, dtype=float)
            if cp_arr.shape[1] >= 3:
                cp_arr = cp_arr[:, :2]  # strip Z
                
            curve_type = e.get('curveType', 'catmullrom').lower()
            if curve_type == 'bezier':
                spline = BezierCurve(cp_arr)
            else:
                spline = CatmullRomSpline(cp_arr)
                
            edge_idx = net.add_route(src, tgt, spline)
        else:
            edge_idx = net.add_simple_route(src, tgt, e.get('weight', 1.0))
            
        # Capture visual metadata
        if 'color' in e:
            viz_meta['edge_colors'][edge_idx] = e['color']
        if 'drawOrder' in e:
            viz_meta['draw_orders'][edge_idx] = e['drawOrder']
            
    # 3. Rebuild Lines
    for li in data.get('lines', []):
        # Default color if missing
        color = li.get('color', [0.4, 0.8, 1.0])
        net.add_line(li['code'], li['name'], tuple(color))
        line_obj = net.get_line(li['code'])
        
        for sid_json in li.get('route', []):
            if sid_json in id_map:
                line_obj.register_station(id_map[sid_json])
                
    return net, viz_meta


def save_network(filepath: str, network: TransitNetwork, viz_meta: dict = None) -> None:
    """Export a TransitNetwork to a TrackDesigner-compatible JSON file.
    
    Parameters
    ----------
    filepath : str
        Output JSON file path.
    network : TransitNetwork
        The transit network to serialize.
    viz_meta : dict, optional
        Visual metadata (edge_colors, label_offsets, etc.). 
        If None, uses defaults.
    """
    viz_meta = viz_meta or {}
    
    # 1. Build stations array
    stations = []
    # Sort strictly by node ID
    for sid in sorted(network.get_graph().nodes()):
        s = network.get_station(sid)
        entry = {
            'id': sid,
            'name': s.get('name', ''),
            'position': s.get('pos', np.array([0., 0.])).tolist()
        }
        offsets = viz_meta.get('label_offsets', {})
        if sid in offsets:
            entry['labelOffset'] = offsets[sid]
        stations.append(entry)
        
    # 2. Build edges array
    edges = []
    # Iterate through edges (might not be sorted by ID inherently, sort them below)
    edge_list = []
    for u, v, d in network.get_graph().edges(data=True):
        edge_list.append((u, v, d))
        
    # Sort by edge_idx
    edge_list.sort(key=lambda item: item[2].get('edge_idx', 0))
    
    edge_counter = 1
    for u, v, d in edge_list:
        edge_idx = d.get('edge_idx', 0)
        entry = {
            'id': edge_counter,
            'source': u,
            'target': v,
            'weight': d.get('weight', 1.0)
        }
        
        spline = network.get_spline(edge_idx)
        if spline:
            cp = spline.get_control_points()
            # Pad to 3D for MATLAB compatibility: [x, y] -> [x, y, 0]
            if cp.shape[1] == 2:
                cp_3d = np.column_stack([cp, np.zeros(len(cp))])
            else:
                cp_3d = cp
            entry['controlPoints'] = cp_3d.tolist()
            if isinstance(spline, BezierCurve):
                entry['curveType'] = 'bezier'
            else:
                entry['curveType'] = 'catmullrom'
        else:
            entry['controlPoints'] = []
            entry['curveType'] = 'none'
            
        edge_colors = viz_meta.get('edge_colors', {})
        if edge_idx in edge_colors:
            entry['color'] = edge_colors[edge_idx]
            
        draw_orders = viz_meta.get('draw_orders', {})
        if edge_idx in draw_orders:
            entry['drawOrder'] = draw_orders[edge_idx]
            
        edges.append(entry)
        edge_counter += 1
        
    # Reconcile lines before saving
    network.reconcile_line_routes()
        
    # 3. Build lines array
    lines = []
    for code in network.get_line_names():
        line = network.get_line(code)
        lines.append({
            'code': line.code,
            'name': line.name,
            'color': list(line.color),
            'route': line.get_route(),
        })
        
    # 4. Assemble full structure
    data = {
        'meta': {
            'defaultLabelOffset': viz_meta.get('default_label_offset', [0.0, 0.3])
        },
        'stations': stations,
        'edges': edges,
        'lines': lines,
    }
    
    # Overlay stuff
    if viz_meta.get('legend_pos1') is not None:
        data['legendPos1'] = viz_meta['legend_pos1']
    if viz_meta.get('legend_pos2') is not None:
        data['legendPos2'] = viz_meta['legend_pos2']
    if viz_meta.get('title_pos') is not None:
        data['titlePos'] = viz_meta['title_pos']
    if viz_meta.get('map_title'):
        data['mapTitle'] = viz_meta['map_title']
        
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
