"""
Autonomous Vehicle Control System - Simulation Module.

This module provides simulation interfaces and tools for testing and validating
autonomous vehicle control algorithms in virtual environments. It includes:

- CARLASimulator: Interface to the CARLA autonomous driving simulator
- SUMOInterface: Interface to SUMO microscopic traffic simulation
- ScenarioRunner: Framework for executing and evaluating test scenarios
- DataRecorder: Recording and playback of simulation data streams

Usage:
    from simulation import CARLASimulator, SUMOInterface, ScenarioRunner, DataRecorder
"""

from simulation.carla_simulator import CARLASimulator
from simulation.sumo_interface import SUMOInterface
from simulation.scenario_runner import ScenarioRunner
from simulation.data_recorder import DataRecorder

__all__ = [
    "CARLASimulator",
    "SUMOInterface",
    "ScenarioRunner",
    "DataRecorder",
]

__version__ = "1.0.0"
