"""
grasshopper public package API.

Euclidean systems are fixed-density systems: candidate lattice sites may
outnumber occupied sites, and physical length scale is set by the occupied-site
count (`System.num_spins`).
"""

from .core import (
    AnnealData,
    AnnealRecord,
    AnnealSchedule,
    ConstraintModel,
    Geometry,
    GreedyHistory,
    GreedyRecord,
    InteractionModel,
    KernelSpec,
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
from .lattices import (
    fibonacci_lattice,
    hexagonal_lattice,
    honeycomb_lattice,
    rectangular_lattice,
    regular_lattice,
    square_lattice,
    triangular_lattice,
)
from .metrics import DistanceMetric
from .parallel_tempering import ParallelTemperingResult

__version__ = "0.4.0"

__all__ = [
    "AnnealData",
    "AnnealRecord",
    "AnnealSchedule",
    "ConstraintModel",
    "DistanceMetric",
    "fibonacci_lattice",
    "Geometry",
    "GreedyHistory",
    "GreedyRecord",
    "GreedyResult",
    "hexagonal_lattice",
    "honeycomb_lattice",
    "InteractionModel",
    "KernelSpec",
    "Neighborhood",
    "NeighborhoodData",
    "NonAntipodalNeighborhood",
    "NonAntipodalNeighborhoodCSR",
    "OccupancyState",
    "ParallelTemperingHistory",
    "ParallelTemperingRecord",
    "ParallelTemperingResult",
    "rectangular_lattice",
    "regular_lattice",
    "square_lattice",
    "SystemConfig",
    "System",
    "triangular_lattice",
    "UpdateRule",
    "__version__",
]


def __getattr__(name: str):
    if name == "System":
        from .system import System

        return System
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
