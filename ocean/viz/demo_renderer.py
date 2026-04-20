"""
demo_renderer.py — The Grand Culmination Test.

Verifies Phases 1 through 5 working in perfect harmony:
1. Loads the Singapore MRT track JSON.
2. Spawns 50 trains randomly across the network.
3. Renders the mathematical graph to a Matplotlib window.
4. Animates the SUVAT physics engine live.

Usage:
    python -m ocean.viz.demo_renderer
"""

import sys
import random
import matplotlib.pyplot as plt

from ocean.io.track_loader import load_network
from ocean.transit.transit_simulation import TransitSimulation
from ocean.viz.track_renderer import TrackRenderer
from ocean.viz.transit_animator import TransitAnimator

TEST_FILE = r'c:\Ocean Library\Matlab code practice\Experiments\Scripts\Simulation\test_simulator_20260402.json'


def main():
    print("============================================================")
    print("  Phase 5 Grand Culmination: Live Simulation Renderer")
    print("============================================================")
    
    # 1. Load Data (Phase 4, Phase 2, Phase 1)
    print("Loading track map...")
    network, viz_meta = load_network(TEST_FILE)
    print(f"  Stations: {network.get_station_count()}")
    print(f"  Lines:    {network.get_line_count()}")
    
    # 2. Physics Orchestrator (Phase 3)
    # We set dt=0 defaults because the Animator injects real_dt live.
    sim = TransitSimulation(network, dt=0.01)

    # 3. Spawn trains randomly on lines
    lines = network.get_line_names()
    train_count = 50
    print(f"Spawning {train_count} trains...")
    
    random.seed(42)  # For consistent demo
    
    agents = []
    for i in range(train_count):
        # Pick a random line
        line_code = random.choice(lines)
        line_obj = network.get_line(line_code)
        
        route = line_obj.get_route()
        if len(route) < 2:
            continue
            
        # Start random segment along the route
        seg_idx = random.randint(0, len(route) - 2)
        
        # Give them realistic speeds (e.g. 5 to 15 grid units per second)
        speed = random.uniform(5.0, 15.0)
        
        # Real-world physics: accel/brake
        # 0.5 to 2.0 u/s^2
        accel = random.uniform(0.5, 2.0)
        brake = accel * 1.5  # Typical train brakes stronger than it accelerates
        
        # Dwell time at stations
        dwell = random.uniform(2.0, 5.0)
        
        agent = sim.add_agent(
            line_code=line_code,
            speed=speed,
            dwell_time=dwell,
            accel_rate=accel,
            brake_rate=brake,
            start_segment=seg_idx
        )
        agents.append(agent)

    print("Opening Graphics Window...")
    
    # 4. Create Matplotlib Architecture (Phase 5)
    # Give it a nice ultra-wide 16:9 cinematic aspect
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.canvas.manager.set_window_title("OceanViz Python - Transit Simulator")

    # The background static renderer
    renderer = TrackRenderer(ax, network, viz_meta)

    # The loop animator (with 2x Time Warp)
    animator = TransitAnimator(fig, ax, renderer, sim, fps=60, time_warp=5.0)

    # BLOCKS until window is closed!
    animator.start()
    
    print("Simulation Terminated.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
