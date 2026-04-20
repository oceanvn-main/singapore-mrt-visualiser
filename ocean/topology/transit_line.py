"""
transit_line.py — Manager for one transit line (e.g. a single MRT line).

Port of: +OceanMath/+Transit/TransitLine.m (228 lines)

Acts as a "highway authority": owns the line's identity (code, name, color),
maintains an ordered station sequence (the route), and holds extensible
operational policies (speed limits, ticket zones, frequency).

Usage:
    line = TransitLine('NSL', 'North-South Line', (0.85, 0.15, 0.15))
    line.register_stations([1, 2, 3, 4])
    line.set_policy('SpeedLimit', 80)
    route = line.get_route()        # [1, 2, 3, 4]
    line.get_policy('SpeedLimit')   # 80
"""

from __future__ import annotations
from typing import Any
import numpy as np


class TransitLine:
    """Manager for a single transit line."""

    def __init__(self, code: str, name: str, color: tuple):
        """Create a new line manager.

        Parameters
        ----------
        code : str
            Short identifier: 'NSL', 'EWL'
        name : str
            Full name: 'North-South Line'
        color : tuple of 3 floats
            RGB visual identity, each in [0, 1]
        """
        self.code = code
        self.name = name
        self.color = tuple(color)
        self._station_ids: list[int] = []
        self._policy: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Route Management
    # ------------------------------------------------------------------

    def register_station(self, station_id: int) -> None:
        """Append a station to end of the route. Skips duplicates."""
        if station_id not in self._station_ids:
            self._station_ids.append(station_id)

    def register_stations(self, station_ids: list[int]) -> None:
        """Bulk-register an ordered list of stations. Skips duplicates."""
        for sid in station_ids:
            self.register_station(sid)

    def insert_station(self, position: int, station_id: int) -> None:
        """Insert a station at a specific 0-based position in the route.

        Parameters
        ----------
        position : int
            0-based insertion index.
        station_id : int
            Station ID to insert.
        """
        if self.has_station(station_id):
            return
        position = min(position, len(self._station_ids))
        self._station_ids.insert(position, station_id)

    def remove_station(self, station_id: int) -> None:
        """Remove a station from the route."""
        self._station_ids = [s for s in self._station_ids if s != station_id]

    def set_route(self, station_ids: list[int]) -> None:
        """Replace the route with an explicitly ordered sequence.

        Used by reconciliation to fix click-order mismatches.
        """
        self._station_ids = list(station_ids)

    def get_route(self) -> list[int]:
        """Return the ordered station IDs (copy)."""
        return list(self._station_ids)

    def get_station_count(self) -> int:
        """Number of stations on this line."""
        return len(self._station_ids)

    def has_station(self, station_id: int) -> bool:
        """Check if a station belongs to this line."""
        return station_id in self._station_ids

    def get_length(self, G) -> float:
        """Total line length (sum of edge weights along route).

        Parameters
        ----------
        G : nx.Graph
            The NetworkX graph providing edge weights.
        """
        total = 0.0
        route = self._station_ids
        for i in range(len(route) - 1):
            s, t = route[i], route[i + 1]
            if G.has_edge(s, t):
                total += G[s][t].get('weight', 0.0)
        return total

    # ------------------------------------------------------------------
    # Policy Management
    # ------------------------------------------------------------------

    def set_policy(self, key: str, value: Any) -> None:
        """Set an operational attribute (e.g. SpeedLimit, TicketZone)."""
        self._policy[key] = value

    def get_policy(self, key: str) -> Any:
        """Get an operational attribute. Returns None if not set."""
        return self._policy.get(key)

    def get_all_policies(self) -> dict:
        """Return the full policy dict (copy)."""
        return dict(self._policy)

    def has_policy(self, key: str) -> bool:
        """Check if a policy key exists."""
        return key in self._policy

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_struct(self) -> dict:
        """Export for JSON serialization."""
        return {
            'Code': self.code,
            'Name': self.name,
            'Color': list(self.color),
            'StationIds': list(self._station_ids),
            'Policy': dict(self._policy),
        }

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Print a human-readable summary."""
        desc = f"  [{self.code}] {self.name} ({self.get_station_count()} stations)\n"
        if self._station_ids:
            ids_str = ' -> '.join(str(s) for s in self._station_ids)
            desc += f"    Route: {ids_str}\n"
        for key, val in self._policy.items():
            desc += f"    {key}: {val}\n"
        print(desc)
        return desc
