"""
Euclidean grid utilities and neighborhood builders.

System-level Euclidean scaling is based on occupied sites, not total candidate
sites. Low-level helpers expose both pieces: candidate-grid normalization uses a
target candidate volume, while neighborhood construction accepts
`occupied_sites` to set interaction cell length.
"""

from ..metrics import DistanceMetric, resolve_distance_metric
from ..lattices import (
    hexagonal_lattice,
    honeycomb_lattice,
    rectangular_lattice,
    regular_lattice,
    square_lattice,
    triangular_lattice,
)
from .build_neighbors import (
    build_euclidean_neighborhood,
    estimate_bounding_box_volume,
    normalize_euclidean_grid,
)

__all__ = [
    "DistanceMetric",
    "build_euclidean_neighborhood",
    "estimate_bounding_box_volume",
    "hexagonal_lattice",
    "honeycomb_lattice",
    "rectangular_lattice",
    "regular_lattice",
    "resolve_distance_metric",
    "normalize_euclidean_grid",
    "square_lattice",
    "triangular_lattice",
]
