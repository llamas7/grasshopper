from .core import (
    AnnealData,
    AnnealRecord,
    AnnealSchedule,
    DeltaType,
    Engine,
    Neighborhood,
    NeighborhoodData,
    NonAntipodalNeighborhood,
    NonAntipodalNeighborhoodCSR,
    OccupancyState,
    SystemState,
    SystemType,
)
from .greedy import GreedyResult, greedy_optimization
from .metrics import DistanceMetric
from .parallel_tempering import ParallelTemperingResult, parallel_tempering

__all__ = [
    "AnnealData",
    "AnnealRecord",
    "AnnealSchedule",
    "DeltaType",
    "DistanceMetric",
    "Engine",
    "GreedyResult",
    "Neighborhood",
    "NeighborhoodData",
    "NonAntipodalNeighborhood",
    "NonAntipodalNeighborhoodCSR",
    "OccupancyState",
    "ParallelTemperingResult",
    "System",
    "SystemState",
    "SystemType",
    "greedy_optimization",
    "parallel_tempering",
]


def __getattr__(name: str):
    if name == "System":
        from .system import System

        return System
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
