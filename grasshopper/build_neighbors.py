from __future__ import annotations

import numpy as np
import scipy as sci
from tqdm import tqdm
from collections.abc import Callable

from .core import Neighborhood, NonAntipodalNeighborhood
from .kernels import calculate_contributions, resolve_delta_kernel
from .metrics import DistanceMetric, resolve_distance_metric, validate_distances


def neighbor_pruning(
    search_points: np.ndarray,
    grid_points: np.ndarray,
    jump_distance: float,
    interaction_spread: float,
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
    if interaction_spread < 0:
        raise ValueError("interaction_spread must be nonnegative")
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
        (interaction_spread * cell_length) ** 2 + midpoint_offset ** 2
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
    jump_distance: float,
    antipodal: bool = True,
    delta: str | Callable = "two",
    silent: bool = True,
    distance_metric: DistanceMetric = "spherical",
):
    return (
        build_antipodal_neighborhood(
            grid_points, jump_distance, delta, silent, distance_metric
        )
        if antipodal
        else build_nonantipodal_neighborhood(
            grid_points, jump_distance, delta, silent, distance_metric
        )
    )


def build_antipodal_neighborhood(
    grid_points: np.ndarray,
    jump_distance: float,
    delta: str | Callable = "two",
    silent: bool = True,
    distance_metric: DistanceMetric = "spherical",
):
    discrete_function, interaction_spread, exact_scan = resolve_delta_kernel(delta)

    num_points = len(grid_points)
    if num_points % 2 != 0:
        raise ValueError("Number of points must be even for antipodal neighborhood.")
    num_spins = num_points // 2
    cell_length = np.sqrt((4 * np.pi) / num_points)
    resolved_metric = resolve_distance_metric(distance_metric)
    jump_distance = float(abs(jump_distance))
    if resolved_metric.name == "spherical":
        jump_distance = jump_distance % np.pi

    neighborhood = []
    if resolved_metric.name == "spherical" and not exact_scan:
        pruned_neighborhood = neighbor_pruning(
            grid_points[: len(grid_points) // 2],
            grid_points,
            jump_distance,
            interaction_spread,
            silent=silent,
        )
    else:
        pruned_neighborhood = [
            np.arange(num_points, dtype=np.int64) for _ in range(num_spins)
        ]
    for i in tqdm(range(num_spins), disable=silent):
        pruned_spins = pruned_neighborhood[i]
        first_half = pruned_spins[pruned_spins < num_spins]
        second_half = pruned_spins[pruned_spins >= num_spins]
        aligned_distances = validate_distances(
            resolved_metric.distance_fn(grid_points[first_half], grid_points[i])
        )
        positive_contributions, positive_bool, total_positive = calculate_contributions(
            jump_distance, cell_length, aligned_distances, discrete_function
        )

        anti_aligned_distances = validate_distances(
            resolved_metric.distance_fn(grid_points[second_half], grid_points[i])
        )
        negative_contributions, negative_bool, total_negative = calculate_contributions(
            jump_distance,
            cell_length,
            anti_aligned_distances,
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
    jump_distance: float,
    delta: str | Callable = "two",
    silent: bool = True,
    distance_metric: DistanceMetric = "spherical",
):
    discrete_function, interaction_spread, exact_scan = resolve_delta_kernel(delta)

    num_spins = len(grid_points)
    cell_length = np.sqrt((4 * np.pi) / num_spins)
    resolved_metric = resolve_distance_metric(distance_metric)
    jump_distance = float(abs(jump_distance))
    if resolved_metric.name == "spherical":
        jump_distance = jump_distance % np.pi

    neighborhood = []
    if resolved_metric.name == "spherical" and not exact_scan:
        pruned_neighborhood = neighbor_pruning(
            grid_points, grid_points, jump_distance, interaction_spread, silent=silent
        )
    else:
        pruned_neighborhood = [
            np.arange(num_spins, dtype=np.int64) for _ in range(num_spins)
        ]
    for i in tqdm(range(num_spins), disable=silent):
        pruned_spins = pruned_neighborhood[i]
        pruned_spins = pruned_spins[pruned_spins != i]
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
