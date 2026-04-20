"""
agent_states.py — State Pattern FSM for TransitAgent.

Port of: +OceanMath/+Transit/+AgentStates/ (4 files, 176 lines)

Three concrete states:
    RunningState  — Agent is moving along a spline segment.
    DwellingState — Agent is paused at a station (dwell countdown).
    ArrivedState  — Agent has reached the terminal (no-op).

Each state implements: step(), get_position(), get_tangent(), get_progress(), 
is_arrived(), get_name().
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

# TYPE_CHECKING avoids circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ocean.transit.transit_agent import TransitAgent


class IAgentState(ABC):
    """Abstract interface for TransitAgent state objects (State Pattern)."""

    @abstractmethod
    def step(self, agent: TransitAgent, dt: float, real_dt: float) -> None:
        """Advance the agent by one time step. May trigger state transitions."""
        ...

    @abstractmethod
    def get_position(self, agent: TransitAgent) -> np.ndarray:
        """Return [x, y] position in this state."""
        ...

    @abstractmethod
    def get_tangent(self, agent: TransitAgent) -> np.ndarray:
        """Return heading direction vector."""
        ...

    @abstractmethod
    def get_progress(self, agent: TransitAgent) -> float:
        """Return route completion fraction [0, 1]."""
        ...

    @abstractmethod
    def is_arrived(self) -> bool:
        """True only for ArrivedState."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return state name string for display/debug."""
        ...


class RunningState(IAgentState):
    """Agent is actively moving along a curve segment.

    Steps the CurveSimulator each tick. Detects segment end and
    transitions to DwellingState or ArrivedState.
    """

    def step(self, agent: TransitAgent, dt: float, real_dt: float) -> None:
        agent.simulator.step(agent.dt)
        s = agent.simulator.get_state()
        agent.record_history(s)

        # Check if segment ended
        if s.parameter >= 1.0:
            n_seg = agent.get_segment_count()
            if agent.current_seg_idx >= n_seg - 1:
                # Last segment — terminal station reached
                agent.set_state(ArrivedState())
            else:
                # Intermediate station — dwell or advance
                if agent.dwell_time > 0:
                    agent.set_state(DwellingState())
                    agent.dwell_remaining = agent.dwell_time
                else:
                    agent.advance_to_next_segment()

    def get_position(self, agent: TransitAgent) -> np.ndarray:
        return agent.simulator.get_state().position

    def get_tangent(self, agent: TransitAgent) -> np.ndarray:
        return agent.simulator.get_state().tangent

    def get_progress(self, agent: TransitAgent) -> float:
        n_seg = agent.get_segment_count()
        seg_progress = agent.simulator.get_state().parameter
        return (agent.current_seg_idx + seg_progress) / n_seg

    def is_arrived(self) -> bool:
        return False

    def get_name(self) -> str:
        return "running"


class DwellingState(IAgentState):
    """Agent is paused at a station (dwell countdown).

    Decrements dwell_remaining by real_dt (wall-clock time, NOT sim time).
    When countdown reaches zero, transitions to RunningState on next segment.
    """

    def step(self, agent: TransitAgent, dt: float, real_dt: float) -> None:
        # Drain by REAL WORLD seconds, independent of TimeWarp
        agent.dwell_remaining -= real_dt
        if agent.dwell_remaining <= 0:
            agent.dwell_remaining = 0.0
            agent.advance_to_next_segment()

    def get_position(self, agent: TransitAgent) -> np.ndarray:
        # Agent has reached the end of current_seg_idx and is waiting there
        station_id = agent.route[agent.current_seg_idx + 1]
        return agent.network.get_station(station_id)['pos']

    def get_tangent(self, agent: TransitAgent) -> np.ndarray:
        return np.array([0.0, 0.0])  # stationary

    def get_progress(self, agent: TransitAgent) -> float:
        n_seg = agent.get_segment_count()
        return (agent.current_seg_idx + 1) / n_seg

    def is_arrived(self) -> bool:
        return False

    def get_name(self) -> str:
        return "dwelling"


class ArrivedState(IAgentState):
    """Agent has reached the terminal station (no-op).

    All queries return terminal values. step() does nothing.
    """

    def step(self, agent: TransitAgent, dt: float, real_dt: float) -> None:
        pass  # no-op

    def get_position(self, agent: TransitAgent) -> np.ndarray:
        last_id = agent.route[-1]
        return agent.network.get_station(last_id)['pos']

    def get_tangent(self, agent: TransitAgent) -> np.ndarray:
        return np.array([0.0, 0.0])  # stationary

    def get_progress(self, agent: TransitAgent) -> float:
        return 1.0

    def is_arrived(self) -> bool:
        return True

    def get_name(self) -> str:
        return "arrived"
