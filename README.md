# Singapore MRT Transit Simulator

A highly interactive Python-based topological graph visualizer and simulation system for the Singapore MRT transit network, featuring Catmull-Rom spline routing and Dijkstra's pathfinding algorithm. This project was originally designed for the COMP1844 curriculum.

## Features

- **Interactive Topology Rendering**: Full NetworkX-backed geometric processing rendered with Matplotlib.
- **Topological Theme Modes**: Launch the network mapping interface in either high-contrast `dark` or clean `light` aesthetics.
- **Dynamic Extensibility**: Parses tracks and routing nodes directly from JSON logic files without hardcoded paths.
- **Live Simulator**: Built-in Kinematics simulation tracking spawned trains via real-time telemetry overlays.
- **Shortest Path Analysis**: Highlight topological traversal between dual-nodes efficiently via Dijkstra.

## Environment Setup

This project uses standard scientific packages. Python 3.9+ is highly recommended. Ensure you install the required dependencies:

```bash
pip install numpy pandas matplotlib networkx
```

## Running the Map Viewer

Launch the static mapping utility via terminal explicitly from the `Python` working directory:

```bash
# Standard Output (Kilometres, Dark Mode)
python -m ocean.viz.view_map

# Light Mode interface
python -m ocean.viz.view_map --theme light

# Dynamic parameter translation (Miles output)
python -m ocean.viz.view_map --theme light --unit miles
```

## Running the Transit Simulator

To view the active transit simulation grid, launch the Application UI:

```bash
python -m ocean.simulator.transit_simulator_app
```

*Note: The native Simulator UI actively enforces a Dark Mode framework to manage telemetry text contrast properly.*
