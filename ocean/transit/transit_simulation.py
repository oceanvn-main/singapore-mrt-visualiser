"""
transit_simulation.py — Orchestrator that runs multiple TransitAgents in lockstep.

Port of: +OceanMath/+Transit/TransitSimulation.m (197 lines)

Creates and ticks agents across a TransitNetwork. Provides a per-frame
snapshot of all agent positions for rendering.
Read only (Render Static image)
Usage:
    sim = TransitSimulation(net, dt=0.02)
    sim.add_agent('NSL', speed=3)
    sim.add_agent('EWL', speed=4)
    while not sim.is_all_arrived():
        sim.step()
        positions = sim.get_positions()
"""

from __future__ import annotations
import numpy as np

from ocean.topology.transit_network import TransitNetwork
from ocean.transit.transit_agent import TransitAgent


class TransitSimulation:
    """Orchestrator that runs multiple TransitAgents in lockstep."""

    def __init__(self, network: TransitNetwork, dt: float = 0.02):
        """Construct the simulation orchestrator.

        Parameters
        ----------
        network : TransitNetwork
        dt : float
            Default time step (seconds).
        """
        self._network = network
        self._agents: list[TransitAgent] = []
        self._dt = dt
        self._clock = 0.0

    # ------------------------------------------------------------------
    # Agent Management
    # ------------------------------------------------------------------

    def add_agent(
        self,
        line_code: str,
        speed: float = 1.0,
        dwell_time: float = 0.5,
        accel_rate: float = 0.0,
        brake_rate: float = 0.0,
        start_segment: int = 0,
    ) -> TransitAgent:
        """Create a TransitAgent on a given line and add to the simulation.

        Returns the created agent.
        """
        agent = TransitAgent(
            self._network, line_code,
            speed=speed, dt=self._dt,
            dwell_time=dwell_time,
            accel_rate=accel_rate,
            brake_rate=brake_rate,
            start_segment=start_segment,
        )
        self._agents.append(agent)
        return agent

    def get_agent_count(self) -> int:
        return len(self._agents)

    def get_agent(self, idx: int) -> TransitAgent:
        return self._agents[idx]

    def get_agents(self) -> list[TransitAgent]:
        return list(self._agents)

    def reset(self) -> None:
        """Clear all agents and reset clock to zero."""
        self._agents.clear()
        self._clock = 0.0

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def step(self, sim_dt: float = None, real_dt: float = None) -> None:
        """Tick all agents by one dt."""
        dt = sim_dt if sim_dt is not None else self._dt
        effective_real_dt = real_dt if real_dt is not None else dt

        self._clock += dt
        for agent in self._agents:
            agent.step(dt, effective_real_dt)

    def run_for(self, n_steps: int) -> None:
        """Step N times."""
        for _ in range(n_steps):
            self.step()
            if self.is_all_arrived():
                break

    def run_until(self, target_time: float) -> None:
        """Step until simulation clock reaches target_time."""
        while self._clock < target_time:
            self.step()
            if self.is_all_arrived():
                break

    def run_to_end(self) -> None:
        """Run until all agents have arrived."""
        max_iter = int(1e6)
        for _ in range(max_iter):
            if self.is_all_arrived():
                break
            self.step()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_positions(self) -> np.ndarray:
        """Return (N, 2) array of current agent positions."""
        n = len(self._agents)
        positions = np.zeros((n, 2))
        for i, agent in enumerate(self._agents):
            positions[i] = agent.get_position()
        return positions

    def get_colors(self) -> np.ndarray:
        """Return (N, 3) array of agent line colors."""
        n = len(self._agents)
        colors = np.zeros((n, 3))
        for i, agent in enumerate(self._agents):
            colors[i] = agent.get_color()
        return colors

    def is_all_arrived(self) -> bool:
        """True when every agent has arrived."""
        return all(agent.is_arrived() for agent in self._agents)

    def get_clock(self) -> float:
        return self._clock

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def describe(self) -> str:
        desc = "TransitSimulation:\n"
        desc += f"  Agents: {len(self._agents)} | Clock: {self._clock:.2f} s | dt: {self._dt:.4f} s\n"
        print(desc)
        for agent in self._agents:
            agent.describe()
        return desc
