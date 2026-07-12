from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..kernels import (
    calculate_contributions,
    resolve_kernel,
)
from ..core import KernelSpec, NonAntipodalNeighborhood
from ..metrics import DistanceMetric, resolve_distance_metric, validate_distances

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def estimate_bounding_box_volume(points: np.ndarray) -> float:
    """
    Estimate candidate-grid volume from an axis-aligned bounding box.

    The estimate includes one inferred lattice spacing beyond the observed
    coordinate extent along each varying axis. For regularly spaced points,
    this approximates the total volume occupied by all candidate site cells,
    not just the min-to-max coordinate span. Dimensions with zero extent are
    treated as extent 1 so lower-dimensional point sets do not collapse to zero
    volume.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("points must be a non-empty 2D array")

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = maxs - mins
    for axis in range(points.shape[1]):
        values = np.unique(points[:, axis])
        if values.size <= 1:
            extents[axis] = max(extents[axis], 1.0)
            continue
        diffs = np.diff(np.sort(values))
        positive_diffs = diffs[diffs > 100.0 * np.finfo(float).eps]
        if positive_diffs.size:
            extents[axis] += float(np.min(positive_diffs))
        else:
            extents[axis] = max(extents[axis], 1.0)
    return float(np.prod(extents))


def normalize_euclidean_grid(
    points: np.ndarray,
    *,
    target_volume: float = 1.0,
) -> np.ndarray:
    """
    Scale Euclidean coordinates to an estimated candidate-grid volume.

    This helper only knows about the points it receives. `System.euclidean()`
    chooses `target_volume = site_count / num_spins`, so the candidate lattice
    can contain more sites than are occupied while each occupied spin still
    represents area/volume `1 / num_spins`.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("points must be a non-empty 2D array")
    target_volume = float(target_volume)
    if not np.isfinite(target_volume) or target_volume <= 0:
        raise ValueError("target_volume must be a positive finite number.")

    current_volume = estimate_bounding_box_volume(points)
    if not np.isfinite(current_volume) or current_volume <= 0:
        raise ValueError("Could not estimate a positive Euclidean grid volume.")
    scale = (target_volume / current_volume) ** (1.0 / points.shape[1])
    return np.ascontiguousarray(points * scale, dtype=np.float64)


def _resolve_occupied_site_count(
    occupied_sites: int | None,
    *,
    site_count: int,
) -> int:
    """Validate the occupied-site count used to set the Euclidean cell scale."""
    if occupied_sites is None:
        return site_count
    if isinstance(occupied_sites, bool) or not isinstance(
        occupied_sites, (int, np.integer)
    ):
        raise TypeError("occupied_sites must be an integer.")
    resolved = int(occupied_sites)
    if resolved <= 0:
        raise ValueError("occupied_sites must be positive.")
    if resolved > site_count:
        raise ValueError(
            f"occupied_sites cannot exceed the number of grid sites "
            f"({site_count}), got {resolved}."
        )
    return resolved


def _ring_pruned_indices_2d(
    *,
    grid_tree,
    center: np.ndarray,
    jump_distance: float,
    width: float,
    cell_length: float,
) -> np.ndarray:
    """Return candidate sites near a 2D Euclidean jump ring."""
    approx_num_circle_points = int(
        np.ceil((2.0 * np.pi * jump_distance) / max(cell_length, 1e-15))
    )
    num_circle_points = max(8, approx_num_circle_points)
    theta = np.linspace(0.0, 2.0 * np.pi, num_circle_points, endpoint=False)
    ring_points = center + jump_distance * np.column_stack(
        (np.cos(theta), np.sin(theta))
    )
    midpoint_offset = jump_distance * np.sin(np.pi / (2.0 * num_circle_points))
    search_radius = np.sqrt(width**2 + midpoint_offset**2)
    raw_indices = grid_tree.query_ball_point(ring_points, search_radius * 1.05)
    nonempty = [
        np.asarray(indices, dtype=np.int64) for indices in raw_indices if len(indices) > 0
    ]
    return (
        np.unique(np.concatenate(nonempty))
        if nonempty
        else np.empty(0, dtype=np.int64)
    )


def _ball_pruned_indices(
    *,
    grid_tree,
    center: np.ndarray,
    radius: float,
    p: float,
) -> np.ndarray:
    """Return candidate sites from a padded KD-tree metric ball."""
    return np.array(
        grid_tree.query_ball_point(center, r=radius, p=p),
        dtype=np.int64,
    )


def build_euclidean_neighborhood(
    grid_points: np.ndarray,
    jump_distance: float,
    kernel: KernelSpec | str | Callable = "cosine",
    silent: bool = True,
    distance_metric: DistanceMetric = "euclidean",
    occupied_sites: int | None = None,
) -> list[NonAntipodalNeighborhood]:
    """
    Build a non-antipodal neighborhood using a modular distance metric.

    `grid_points` are candidate lattice sites. `occupied_sites` controls the
    physical cell length used by the interaction kernel:
    `(1.0 / occupied_sites) ** (1 / dimension)`. When called through
    `System.compute_neighborhood()`, this is `system.num_spins`, so changing the
    number of candidate sites does not change the physical length scale unless
    the number of occupied spins changes.

    `distance_metric` may be one of the built-ins: "euclidean", "manhattan",
    or "chebyshev". It may also be a callable accepting `(points, spin)` and
    returning one distance per point, or accepting `(point, spin)` and returning
    one scalar distance.
    """
    grid_points = np.asarray(grid_points, dtype=float)
    if grid_points.ndim != 2 or grid_points.shape[0] == 0:
        raise ValueError("grid_points must be a non-empty 2D array")

    site_count = int(grid_points.shape[0])
    dim = int(grid_points.shape[1])
    occupied_site_count = _resolve_occupied_site_count(
        occupied_sites,
        site_count=site_count,
    )
    cell_length = (1.0 / occupied_site_count) ** (1.0 / dim)
    resolved_kernel = resolve_kernel(kernel, cell_length=cell_length)
    discrete_function = resolved_kernel.function

    jump_distance = float(jump_distance)
    if not np.isfinite(jump_distance) or jump_distance < 0:
        raise ValueError("jump_distance must be a nonnegative finite number.")
    width = resolved_kernel.support_width
    if width is None:
        width = 0.0
    radius = jump_distance + width

    resolved_metric = resolve_distance_metric(distance_metric)
    grid_tree = None
    if (
        resolved_metric.kd_tree_p is not None
        and not resolved_kernel.exact_scan_required
        and resolved_kernel.support_width is not None
    ):
        import scipy as sci

        grid_tree = sci.spatial.cKDTree(grid_points)
    all_spins = np.arange(site_count, dtype=np.int64)

    neighborhood: list[NonAntipodalNeighborhood] = []
    for i in tqdm(range(site_count), disable=silent):
        if grid_tree is None:
            pruned_spins = all_spins.copy()
        elif resolved_metric.name == "euclidean" and dim == 2:
            pruned_spins = _ring_pruned_indices_2d(
                grid_tree=grid_tree,
                center=grid_points[i],
                jump_distance=jump_distance,
                width=width,
                cell_length=cell_length,
            )
        else:
            pruned_spins = _ball_pruned_indices(
                grid_tree=grid_tree,
                center=grid_points[i],
                radius=radius,
                p=resolved_metric.kd_tree_p,
            )
        pruned_spins = pruned_spins[pruned_spins != i]

        if pruned_spins.size == 0:
            neighborhood.append(
                NonAntipodalNeighborhood(
                    neighbor_spins=np.array([], dtype=np.int64),
                    contributions=np.array([], dtype=np.float64),
                    total_weight=0.0,
                )
            )
            continue

        distances = validate_distances(
            resolved_metric.distance_fn(grid_points[pruned_spins], grid_points[i])
        )
        contributions, contributing_mask, total_weight = calculate_contributions(
            jump_distance, cell_length, distances, discrete_function
        )
        neighborhood.append(
            NonAntipodalNeighborhood(
                neighbor_spins=pruned_spins[contributing_mask],
                contributions=contributions,
                total_weight=total_weight,
            )
        )

    return neighborhood
