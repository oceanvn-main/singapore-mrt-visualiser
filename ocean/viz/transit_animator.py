"""
transit_animator.py — Real-time physics simulation using FuncAnimation.

Port of: +OceanViz/+Simulator/TransitSimulatorApp.m
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ocean.viz.track_renderer import TrackRenderer
from ocean.transit.transit_simulation import TransitSimulation


class TransitAnimator:
    """Animates a TransitSimulation on top of a TrackRenderer."""

    def __init__(self, fig: plt.Figure, ax: plt.Axes, 
                 renderer: TrackRenderer, simulation: TransitSimulation, 
                 fps: int = 30, time_warp: float = 1.0):
        """Construct the animator.
        
        Parameters
        ----------
        fig : plt.Figure
            The matplotlib figure.
        ax : plt.Axes
            The axes to draw on.
        renderer : TrackRenderer
            The renderer responsible for drawing the background map.
        simulation : TransitSimulation
            The physics engine orchestrator (already populated with agents).
        fps : int
            Frames per second for visually updating the screen.
        time_warp : float
            Simulation speed multiplier.
        """
        self.fig = fig
        self.ax = ax
        self.renderer = renderer
        self.simulation = simulation
        
        self.fps = fps
        self.interval_ms = 1000.0 / fps
        self.time_warp = time_warp
        
        # Real-time state
        self._last_real_time = time.time()
        
        # Graphics handles
        self._scat = None
        self._text_clock = None
        
        # Internals
        self._anim = None

    def _init_frame(self):
        """Called once before animation starts."""
        self._last_real_time = time.time()
        
        # 1. Draw static background
        self.renderer.draw_network()
        
        # 2. Setup dynamic train scatter plot
        # 15.0 markersize = 225 area
        self._scat = self.ax.scatter([], [], s=40, zorder=200, 
                                     edgecolors='#FFFFFF', linewidths=1.5)
                                     
        # 3. Setup dynamic HUD clock
        self._text_clock = self.ax.text(0.05, 0.05, 'T: 0.00s',
                                        color='#44FF44', 
                                        fontsize=12,
                                        fontweight='bold',
                                        transform=self.ax.transAxes,
                                        ha='left', va='bottom')
                                        
        return self._scat, self._text_clock

    def _update_frame(self, frame):
        """Called every frame by FuncAnimation."""
        now = time.time()
        real_dt = now - self._last_real_time
        self._last_real_time = now
        
        # Prevent huge jumps if animation is lagging
        real_dt = min(real_dt, 0.1)
        
        # Mathematical simulation step
        sim_dt = real_dt * self.time_warp
        
        # Physics Engine update
        #   real_dt is passed explicitly for DwellTime consistency (which ticks in real seconds)
        #   sim_dt is used for physical SUVAT movement scaling
        self.simulation.step(sim_dt=sim_dt, real_dt=real_dt)
        
        # Query Engine for visual state
        pos = self.simulation.get_positions()
        colors = self.simulation.get_colors()
        
        if len(pos) > 0:
            self._scat.set_offsets(pos)
            # Scatter expects 2D color array ([R, G, B])
            self._scat.set_facecolor(colors)
            
        # Update HUD
        t = self.simulation.get_clock()
        active = sum(1 for agent in self.simulation.get_agents() if not agent.is_arrived())
        total = self.simulation.get_agent_count()
        self._text_clock.set_text(f'T: {t:.1f}s  |  Trains: {active}/{total}')
        
        return self._scat, self._text_clock

    def start(self):
        """Invoke the FuncAnimation loop and show the window."""
        self._anim = animation.FuncAnimation(
            self.fig, self._update_frame, 
            init_func=self._init_frame,
            interval=self.interval_ms,
            blit=True,
            cache_frame_data=False
        )
        
        # This is blocking!
        plt.show()
