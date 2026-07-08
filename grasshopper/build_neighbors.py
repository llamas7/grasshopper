from __future__ import annotations

import numpy as np
import scipy as sci
from tqdm import tqdm
from collections.abc import Callable, Sequence

from .core import KernelSpec, Neighborhood, NonAntipodalNeighborhood
from .kernels import calculate_contributions, resolve_kernel
from .metrics import DistanceMetric, resolve_distance_metric, validate_distances


def neighbor_pruning(
    search_points: np.ndarray,
    grid_points: np.ndarray,
    jump_distance: float,
    support_width: float,
    silent: bool = True,
):
    """
    Returns a list of arrays, where each array contains the indices
    of potential neighbors on `grid_points` for each point in `search_points`,
    using a spherical pruning radius determined by `jump_distance`.
    """
    search_points = np.asarray(search_points, dtype=float)
    grid_points = np.asarray(grid_points, dtype=float)

    if search_points.ndim != 2 or search_points.shape[1] != 3:
        raise ValueError("search_points must have shape (N, 3)")
    if grid_points.ndim != 2 or grid_points.shape[1] != 3:
        raise ValueError("grid_points must have shape (M, 3)")
    if grid_points.shape[0] == 0:
        raise ValueError("grid_points must be non-empty")
    if support_width < 0:
        raise ValueError("support_width must be nonnegative")
    if not np.isfinite(jump_distance):
        raise ValueError("jump_distance must be finite")
    if not np.all(np.isfinite(search_points)) or not np.all(np.isfinite(grid_points)):
        raise ValueError("inputs must contain only finite values")

    search_norms = np.linalg.norm(search_points, axis=1, keepdims=True)
    grid_norms = np.linalg.norm(grid_points, axis=1, keepdims=True)
    if np.any(search_norms == 0) or np.any(grid_norms == 0):
        raise ValueError("search_points and grid_points must not contain zero vectors")

    search_points = search_points / search_norms
    grid_points = grid_points / grid_norms

    grid_tree = sci.spatial.cKDTree(grid_points)
    num_grid_points = grid_points.shape[0]
    cell_length = np.sqrt(4.0 * np.pi / num_grid_points)
    ring_radius = np.sin(jump_distance)

    approx_num_circle_points = int(
        np.ceil((2.0 * np.pi * abs(ring_radius)) / max(cell_length, 1e-15))
    )
    num_circle_points = max(8, approx_num_circle_points)
    midpoint_offset = abs(ring_radius) * np.sin(np.pi / (2.0 * num_circle_points))
    search_radius = np.sqrt(
        support_width**2 + midpoint_offset**2
    )

    theta = np.linspace(0.0, 2.0 * np.pi, num_circle_points, endpoint=False)[:, None]
    neighbor_indices = []
    buffer = 1.05

    for center in tqdm(search_points, disable=silent):
        ref = (
            np.array([1.0, 0.0, 0.0])
            if abs(center[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        v = np.cross(center, ref)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            ref = np.array([0.0, 0.0, 1.0])
            v = np.cross(center, ref)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                raise ValueError("failed to construct tangent basis for search point")

        v /= v_norm
        z = np.cross(center, v)
        z /= np.linalg.norm(z)

        circle_points = (
            ring_radius * (np.cos(theta) * v + np.sin(theta) * z)
            + np.cos(jump_distance) * center
        )

        raw_indices = grid_tree.query_ball_point(circle_points, search_radius * buffer)
        nonempty = [
            np.asarray(idx, dtype=np.int64) for idx in raw_indices if len(idx) > 0
        ]
        flattened = (
            np.unique(np.concatenate(nonempty))
            if nonempty
            else np.empty(0, dtype=np.int64)
        )
        neighbor_indices.append(flattened)

    return neighbor_indices


def build_neighborhood(
    grid_points: np.ndarray,
    jump_distance: float | Sequence[float],
    antipodal: bool = True,
    kernel: KernelSpec | str | Callable = "cosine",
    silent: bool = True,
    distance_metric: DistanceMetric = "spherical",
    jump_weights: Sequence[float] | None = None,
):
    return (
        build_antipodal_neighborhood(
            grid_points, jump_distance, kernel, silent, distance_metric, jump_weights
        )
        if antipodal
        else build_nonantipodal_neighborhood(
            grid_points, jump_distance, kernel, silent, distance_metric, jump_weights
        )
    )


def _coerce_shells(
    jump_distance: float | Sequence[float],
    jump_weights: Sequence[float] | None,
    *,
    spherical: bool,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.asarray(jump_distance, dtype=np.float64)
    if distances.ndim == 0:
        distances = distances.reshape(1)
    if distances.ndim != 1 or distances.size == 0:
        raise ValueError(
            "jump_distance must be a finite scalar or non-empty 1D sequence."
        )
    if not np.all(np.isfinite(distances)):
        raise ValueError("jump_distance values must be finite.")
    if np.any(distances < 0):
        raise ValueError("jump_distance values must be nonnegative.")
    if spherical and np.any(distances > np.pi):
        raise ValueError("Spherical jump_distance values must lie in [0, pi].")

    if jump_weights is None:
        weights = np.ones(distances.shape, dtype=np.float64)
    else:
        weights = np.asarray(jump_weights, dtype=np.float64)
        if weights.ndim == 0:
            weights = weights.reshape(1)
        if weights.ndim != 1 or weights.shape != distances.shape:
            raise ValueError(
                "jump_weights must have one finite value per jump distance."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("jump_weights values must be finite.")
    return distances, weights


def _candidate_rows(
    grid_points: np.ndarray,
    search_points: np.ndarray,
    num_candidates: int,
    distances: np.ndarray,
    support_width: float | None,
    exact_scan_required: bool,
    metric_name: str,
    silent: bool,
) -> list[np.ndarray]:
    if (
        metric_name != "spherical"
        or exact_scan_required
        or support_width is None
    ):
        return [
            np.arange(num_candidates, dtype=np.int64)
            for _ in range(len(search_points))
        ]

    per_shell = [
        neighbor_pruning(
            search_points,
            grid_points,
            float(distance),
            support_width,
            silent=silent,
        )
        for distance in distances
    ]
    rows: list[np.ndarray] = []
    for row_index in range(len(search_points)):
        pieces = [
            shell_rows[row_index]
            for shell_rows in per_shell
            if shell_rows[row_index].size
        ]
        rows.append(
            np.unique(np.concatenate(pieces))
            if pieces
            else np.empty(0, dtype=np.int64)
        )
    return rows


def _weighted_contributions(
    distances_to_candidates: np.ndarray,
    shell_distances: np.ndarray,
    shell_weights: np.ndarray,
    cell_length: float,
    discrete_function: Callable,
) -> tuple[np.ndarray, np.ndarray, float]:
    totals = np.zeros(distances_to_candidates.shape, dtype=np.float64)
    for shell_distance, shell_weight in zip(shell_distances, shell_weights):
        contributions, contributing_mask, _ = calculate_contributions(
            float(shell_distance),
            cell_length,
            distances_to_candidates,
            discrete_function,
        )
        if contributions.size:
            totals[contributing_mask] += float(shell_weight) * contributions
    contributing_mask = np.abs(totals) > 100.0 * np.finfo(float).eps
    contributions = totals[contributing_mask]
    return contributions, contributing_mask, float(np.sum(contributions))


def build_antipodal_neighborhood(
    grid_points: np.ndarray,
    jump_distance: float | Sequence[float],
    kernel: KernelSpec | str | Callable = "cosine",
    silent: bool = True,
    distance_metric: DistanceMetric = "spherical",
    jump_weights: Sequence[float] | None = None,
):
    num_points = len(grid_points)
    if num_points % 2 != 0:
        raise ValueError("Number of points must be even for antipodal neighborhood.")
    num_spins = num_points // 2
    cell_length = np.sqrt((4 * np.pi) / num_points)
    resolved_kernel = resolve_kernel(kernel, cell_length=cell_length)
    discrete_function = resolved_kernel.function
    resolved_metric = resolve_distance_metric(distance_metric)
    shell_distances, shell_weights = _coerce_shells(
        jump_distance,
        jump_weights,
        spherical=resolved_metric.name == "spherical",
    )

    neighborhood = []
    pruned_neighborhood = _candidate_rows(
        grid_points,
        grid_points[: len(grid_points) // 2],
        num_points,
        shell_distances,
        resolved_kernel.support_width,
        resolved_kernel.exact_scan_required,
        resolved_metric.name,
        silent,
    )
    for i in tqdm(range(num_spins), disable=silent):
        pruned_spins = pruned_neighborhood[i]
        first_half = pruned_spins[pruned_spins < num_spins]
        second_half = pruned_spins[pruned_spins >= num_spins]
        aligned_distances = validate_distances(
            resolved_metric.distance_fn(grid_points[first_half], grid_points[i])
        )
        positive_contributions, positive_bool, total_positive = _weighted_contributions(
            aligned_distances,
            shell_distances,
            shell_weights,
            cell_length,
            discrete_function,
        )

        anti_aligned_distances = validate_distances(
            resolved_metric.distance_fn(grid_points[second_half], grid_points[i])
        )
        negative_contributions, negative_bool, total_negative = _weighted_contributions(
            anti_aligned_distances,
            shell_distances,
            shell_weights,
            cell_length,
            discrete_function,
        )

        neighborhood.append(
            Neighborhood(
                positive_spins=np.array(first_half[positive_bool]),
                positive_contributions=positive_contributions,
                negative_spins=np.array(second_half[negative_bool] - num_spins),
                negative_contributions=negative_contributions,
                spin_potential=total_positive + total_negative,
            )
        )
    return neighborhood


def build_nonantipodal_neighborhood(
    grid_points: np.ndarray,
    jump_distance: float | Sequence[float],
    kernel: KernelSpec | str | Callable = "cosine",
    silent: bool = True,
    distance_metric: DistanceMetric = "spherical",
    jump_weights: Sequence[float] | None = None,
):
    num_spins = len(grid_points)
    cell_length = np.sqrt((4 * np.pi) / num_spins)
    resolved_kernel = resolve_kernel(kernel, cell_length=cell_length)
    discrete_function = resolved_kernel.function
    resolved_metric = resolve_distance_metric(distance_metric)
    shell_distances, shell_weights = _coerce_shells(
        jump_distance,
        jump_weights,
        spherical=resolved_metric.name == "spherical",
    )

    neighborhood = []
    pruned_neighborhood = _candidate_rows(
        grid_points,
        grid_points,
        num_spins,
        shell_distances,
        resolved_kernel.support_width,
        resolved_kernel.exact_scan_required,
        resolved_metric.name,
        silent,
    )
    for i in tqdm(range(num_spins), disable=silent):
        pruned_spins = pruned_neighborhood[i]
        pruned_spins = pruned_spins[pruned_spins != i]
        distances = validate_distances(
            resolved_metric.distance_fn(grid_points[pruned_spins], grid_points[i])
        )
        contributions, contributing_mask, spin_potential = _weighted_contributions(
            distances,
            shell_distances,
            shell_weights,
            cell_length,
            discrete_function,
        )
        neighborhood.append(
            NonAntipodalNeighborhood(
                neighbor_spins=pruned_spins[contributing_mask],
                contributions=contributions,
                spin_potential=spin_potential,
            )
        )

    return neighborhood
