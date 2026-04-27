from .core import (
    AnnealData,
    AnnealRecord,
    AnnealSchedule,
    ConstraintModel,
    DeltaType,
    Geometry,
    GreedyHistory,
    GreedyRecord,
    InteractionModel,
    Neighborhood,
    NeighborhoodData,
    NonAntipodalNeighborhood,
    NonAntipodalNeighborhoodCSR,
    OccupancyState,
    ParallelTemperingHistory,
    ParallelTemperingRecord,
    SystemConfig,
    UpdateRule,
)
from .greedy import GreedyResult
from .metrics import DistanceMetric
from .parallel_tempering import ParallelTemperingResult

__all__ = [
    "AnnealData",
    "AnnealRecord",
    "AnnealSchedule",
    "ConstraintModel",
    "DeltaType",
    "DistanceMetric",
    "Geometry",
    "GreedyHistory",
    "GreedyRecord",
    "GreedyResult",
    "InteractionModel",
    "Neighborhood",
    "NeighborhoodData",
    "NonAntipodalNeighborhood",
    "NonAntipodalNeighborhoodCSR",
    "OccupancyState",
    "ParallelTemperingHistory",
    "ParallelTemperingRecord",
    "ParallelTemperingResult",
    "SystemConfig",
    "System",
    "UpdateRule",
]


def __getattr__(name: str):
    if name == "System":
        from .system import System

        return System
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
