from __future__ import annotations

import numpy as np

from ..kernels import (
    calculate_contributions,
    resolve_delta_kernel,
)
from ..core import NonAntipodalNeighborhood
from ..metrics import DistanceMetric, resolve_distance_metric, validate_distances

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def estimate_bounding_box_volume(points: np.ndarray) -> float:
    """
    Estimate domain volume as axis-aligned bounding-box volume.

    Dimensions with zero extent are treated as extent 1 so lower-dimensional
    point sets do not collapse to zero cell length.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("points must be a non-empty 2D array")

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = np.maximum(maxs - mins, 1.0)
    return float(np.prod(extents))


def calculate_prob_prefactor_euclidean(
    n_points: int,
    jump_distance: float,
    dim: int,
    *,
    domain_volume: float | None = None,
    constant: float = 1.0,
) -> float:
    """
    Probability prefactor for Euclidean systems.

    The exact physical constant is model-dependent; this keeps the old
    placeholder scaling explicit and centralized.
    """
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")
    return float(constant / n_points)


def build_euclidean_neighborhood(
    grid_points: np.ndarray,
    jump_distance: float,
    delta: str = "two",
    silent: bool = True,
    distance_metric: DistanceMetric = "euclidean",
) -> list[NonAntipodalNeighborhood]:
    """
    Build a non-antipodal neighborhood using a modular distance metric.

    `distance_metric` may be one of the built-ins: "euclidean", "manhattan",
    or "chebyshev". It may also be a callable accepting `(points, spin)` and
    returning one distance per point, or accepting `(point, spin)` and returning
    one scalar distance.
    """
    grid_points = np.asarray(grid_points, dtype=float)
    if grid_points.ndim != 2 or grid_points.shape[0] == 0:
        raise ValueError("grid_points must be a non-empty 2D array")

    discrete_function, interaction_spread, exact_scan = resolve_delta_kernel(delta)

    num_spins = int(grid_points.shape[0])
    dim = int(grid_points.shape[1])
    domain_volume = estimate_bounding_box_volume(grid_points)
    cell_length = (domain_volume / num_spins) ** (1.0 / dim)

    jump_distance = float(abs(jump_distance))
    width = interaction_spread * cell_length
    radius = jump_distance + width

    resolved_metric = resolve_distance_metric(distance_metric)
    grid_tree = None
    if resolved_metric.kd_tree_p is not None and not exact_scan:
        import scipy as sci

        grid_tree = sci.spatial.cKDTree(grid_points)
    all_spins = np.arange(num_spins, dtype=np.int64)

    neighborhood: list[NonAntipodalNeighborhood] = []
    for i in tqdm(range(num_spins), disable=silent):
        if grid_tree is None:
            pruned_spins = all_spins.copy()
        else:
            pruned_spins = np.array(
                grid_tree.query_ball_point(
                    grid_points[i], r=radius, p=resolved_metric.kd_tree_p
                ),
                dtype=np.int64,
            )
        pruned_spins = pruned_spins[pruned_spins != i]

        if pruned_spins.size == 0:
            neighborhood.append(
                NonAntipodalNeighborhood(
                    neighbor_spins=np.array([], dtype=np.int64),
                    contributions=np.array([], dtype=np.float64),
                    spin_potential=0.0,
                )
            )
            continue

        distances = validate_distances(
            resolved_metric.distance_fn(grid_points[pruned_spins], grid_points[i])
        )
        contributions, contributing_mask, spin_potential = calculate_contributions(
            jump_distance, cell_length, distances, discrete_function
        )
        neighborhood.append(
            NonAntipodalNeighborhood(
                neighbor_spins=pruned_spins[contributing_mask],
                contributions=contributions,
                spin_potential=spin_potential,
            )
        )

    return neighborhood
