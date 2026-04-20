"""
demo_simulator.py — Entry point for the full interactive Transit Simulator.

Usage:
    python -m ocean.simulator.demo_simulator
"""

import sys
from ocean.simulator.transit_simulator_app import TransitSimulatorApp


def main():
    app = TransitSimulatorApp()
    app.launch()
    return 0

if __name__ == '__main__':
    sys.exit(main())
