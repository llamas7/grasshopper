from .core import (
    AnnealData,
    AnnealRecord,
    AnnealSchedule,
    config_from_system_type,
    ConstraintModel,
    DeltaType,
    Engine,
    Geometry,
    GreedyHistory,
    GreedyRecord,
    InteractionModel,
    NeighborhoodModel,
    Neighborhood,
    NeighborhoodData,
    NonAntipodalNeighborhood,
    NonAntipodalNeighborhoodCSR,
    OptimizationRuntime,
    OccupancyState,
    ParallelTemperingHistory,
    ParallelTemperingRecord,
    SystemConfig,
    SystemDefinition,
    SystemState,
    SystemType,
    system_type_from_config,
    UpdateRule,
)
from .greedy import GreedyResult, greedy_optimization
from .metrics import DistanceMetric
from .parallel_tempering import ParallelTemperingResult, parallel_tempering

__all__ = [
    "AnnealData",
    "AnnealRecord",
    "AnnealSchedule",
    "config_from_system_type",
    "ConstraintModel",
    "DeltaType",
    "DistanceMetric",
    "Engine",
    "Geometry",
    "GreedyHistory",
    "GreedyRecord",
    "GreedyResult",
    "InteractionModel",
    "NeighborhoodModel",
    "Neighborhood",
    "NeighborhoodData",
    "NonAntipodalNeighborhood",
    "NonAntipodalNeighborhoodCSR",
    "OptimizationRuntime",
    "OccupancyState",
    "ParallelTemperingHistory",
    "ParallelTemperingRecord",
    "ParallelTemperingResult",
    "SystemConfig",
    "System",
    "SystemDefinition",
    "SystemState",
    "SystemType",
    "system_type_from_config",
    "UpdateRule",
    "greedy_optimization",
    "parallel_tempering",
]


def __getattr__(name: str):
    if name == "System":
        from .system import System

        return System
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
