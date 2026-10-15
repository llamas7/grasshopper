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
    GreedyHistory,
    GreedyRecord,
    KernelSpec,
    Neighborhood,
    NeighborhoodData,
    FixedDensityNeighborhood,
    FixedDensityNeighborhoodCSR,
    OccupancyState,
    ParallelTemperingHistory,
    ParallelTemperingRecord,
    SystemKind,
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

__version__ = "1.0.0"

__all__ = [
    "AnnealData",
    "AnnealRecord",
    "AnnealSchedule",
    "DistanceMetric",
    "fibonacci_lattice",
    "GreedyHistory",
    "GreedyRecord",
    "GreedyResult",
    "hexagonal_lattice",
    "honeycomb_lattice",
    "KernelSpec",
    "Neighborhood",
    "NeighborhoodData",
    "FixedDensityNeighborhood",
    "FixedDensityNeighborhoodCSR",
    "OccupancyState",
    "ParallelTemperingHistory",
    "ParallelTemperingRecord",
    "ParallelTemperingResult",
    "rectangular_lattice",
    "regular_lattice",
    "square_lattice",
    "System",
    "SystemKind",
    "triangular_lattice",
    "__version__",
]


def __getattr__(name: str):
    if name == "System":
        from .system import System

        return System
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
