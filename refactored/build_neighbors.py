from __future__ import annotations
from .core import Neighborhood, NonAntipodalNeighborhood
import numpy as np
import scipy as sci
from tqdm import tqdm
from numba import njit, jit
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import System


def calculate_prob_prefactor(N: int, theta: float):
    return 4 / (np.sin(theta) * N ** (3 / 2) * np.sqrt(np.pi))


def delta_two(
    spin_distance: np.ndarray, cell_length: float
) -> tuple[np.ndarray, np.ndarray]:
    r = spin_distance / cell_length
    r_bool = r < 2
    return (1 + np.cos(np.pi * r[r_bool] / 2)) / 4.0, r_bool


def delta_one_star(
    spin_distance: np.ndarray, cell_length: float
) -> tuple[np.ndarray, np.ndarray]:
    def calculate_contributions(r: np.ndarray) -> np.ndarray:
        contributions = np.zeros(len(r))
        bool_mask = r <= 0.5
        contributions[bool_mask] = (3 / 4) - r[bool_mask] ** 2
        bool_mask = np.invert(bool_mask)
        contributions[bool_mask] = (
            (9 / 8) - 3 * r[bool_mask] / 2 + (r[bool_mask] ** 2) / 2
        )
        return contributions

    r = spin_distance / cell_length
    r_bool = r <= 1.5
    return calculate_contributions(r[r_bool]), r_bool


def delta_two_star(spin_distance, cell_length):
    def calculate_contributions(r):
        contributions = np.zeros(len(r))
        bool_mask = r <= 1.5
        contributions[bool_mask] = (1 / 4 * np.pi) * (
            np.pi
            + 2 * np.sin((np.pi / 4) * (2 * r[bool_mask] + 1))
            - 2 * np.sin((np.pi / 4) * (2 * r[bool_mask] - 1))
        )
        bool_mask = np.invert(bool_mask)
        contributions[bool_mask] = -(1 / (8 * np.pi)) * (
            -5 * np.pi
            + 2 * np.pi * r[bool_mask]
            + 4 * np.sin((np.pi / 4) * (2 * r[bool_mask] - 1))
        )
        return contributions

    r = spin_distance / cell_length
    r_bool = r <= 2.5
    return calculate_contributions(r[r_bool]), r_bool


def calculate_contributions(
    jump_distance: float,
    cell_length: float,
    distances: np.ndarray,
    delta_func: Callable,
) -> tuple:
    abs_distances = np.abs(jump_distance - distances)
    contributions, interacting_bool = delta_func(abs_distances, cell_length)
    return (
        contributions,
        interacting_bool,
        np.sum(contributions),
    )


def get_distances(grid_points: np.ndarray, spin: np.ndarray) -> np.ndarray:
    distances = np.abs(grid_points - spin)
    distances = np.linalg.norm(distances, axis=1)
    distances = 2 * np.arcsin(np.clip(distances / 2, 0, 1))
    return distances


def neighbor_chord_pruning(
    search_points: np.ndarray,
    grid_points: np.ndarray,
    jump_distance: float,
    silent: bool = True,
):
    """
    Fast neighbor pruning using KD-tree ball queries with angular-to-chord conversion,
    followed by angular filtering to approximate a delta-function kernel centered
    at `jump_distance`.

    Parameters:
        search_points: points for which to find neighbors
        grid_points: full grid of candidate points
        jump_distance: angular distance target (radians)
        angular_width: allowable deviation from jump_distance (± window)
        silent: suppress tqdm progress bar

    Returns:
        List of index arrays corresponding to pruned neighbors per search point.
    """
    cos_target = np.cos(jump_distance)
    cell_length = np.sqrt(4 * np.pi / grid_points.shape[0])
    angular_width = 2 * cell_length  # Angular width for pruning
    cos_min = np.cos(jump_distance + angular_width)
    cos_max = np.cos(jump_distance - angular_width)

    # Convert angular width to Euclidean (chord) radius for KD-tree query
    chord_radius = np.sqrt(2 * (1 - np.cos(jump_distance + angular_width)))
    grid_tree = sci.spatial.cKDTree(grid_points)

    total_points = 0
    total_filtered = 0
    pruned_neighbors = []
    for center in tqdm(search_points, disable=silent):
        candidate_idxs = grid_tree.query_ball_point(center, r=chord_radius)
        candidate_vecs = grid_points[candidate_idxs]

        # Compute cos(angle) via dot product (points are on sphere)
        dot_products = candidate_vecs @ center
        mask = (dot_products >= cos_min) & (dot_products <= cos_max)

        filtered_idxs = np.array(candidate_idxs)[mask]
        pruned_neighbors.append(filtered_idxs)
        total_points += len(candidate_idxs)
        total_filtered += len(filtered_idxs)
    print(
        f"Total points to filter: {total_points}, "
        f"Total pre-filtered: {total_filtered}"
    )

    return pruned_neighbors


def neighbor_pruning(
    search_points: np.ndarray,
    grid_points: np.ndarray,
    jump_distance: float,
    interaction_spread: float,
    silent: bool = True,
):
    """
    Return a list of index arrays. Each array contains candidate neighbor
    indices on `grid_points` for the corresponding point in `search_points`.

    Geometry / assumptions
    ----------------------
    - `search_points` and `grid_points` are 3D points on the unit sphere.
    - `jump_distance` is an angular distance in radians.
    - This is a pruning heuristic: it returns a superset of plausible
      neighbors by sampling the ring at angular distance `jump_distance`
      and querying a KD-tree around those samples.
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

    # Normalize onto the unit sphere.
    search_norms = np.linalg.norm(search_points, axis=1, keepdims=True)
    grid_norms = np.linalg.norm(grid_points, axis=1, keepdims=True)

    if np.any(search_norms == 0) or np.any(grid_norms == 0):
        raise ValueError("search_points and grid_points must not contain zero vectors")

    search_points = search_points / search_norms
    grid_points = grid_points / grid_norms

    grid_tree = sci.spatial.cKDTree(grid_points)

    num_grid_points = grid_points.shape[0]
    cell_length = np.sqrt(4.0 * np.pi / num_grid_points)

    # Ring radius in ambient 3D for a unit-sphere angular jump.
    ring_radius = np.sin(jump_distance)

    # Efficient enough sampling of the ring: Balances search radius with number of samples.
    # The previous code could produce zero samples for small jump_distance.
    approx_num_circle_points = int(
        np.ceil((2.0 * np.pi * abs(ring_radius)) / max(cell_length, 1e-15))
    )
    num_circle_points = max(8, approx_num_circle_points)

    # Distance from the midpoint between adjacent query centers to the nearest
    # query center, measured as a chord in 3D.
    midpoint_offset = abs(ring_radius) * np.sin(np.pi / (2.0 * num_circle_points))

    # Combined tolerance: physical interaction spread + sampling discretization.
    search_radius = np.sqrt(
        (interaction_spread * cell_length) ** 2 + midpoint_offset**2
    )

    # Small safety margin for numerical / geometric slack.
    buffer = 1.05

    theta = np.linspace(0.0, 2.0 * np.pi, num_circle_points, endpoint=False)[:, None]

    neighbor_indices = []

    for center in tqdm(search_points, disable=silent):
        # Deterministic perpendicular vector.
        if abs(center[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])

        v = np.cross(center, ref)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            # Extremely defensive fallback.
            ref = np.array([0.0, 0.0, 1.0])
            v = np.cross(center, ref)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                raise ValueError("failed to construct tangent basis for search point")

        v /= v_norm
        z = np.cross(center, v)
        z /= np.linalg.norm(z)

        # Sample the ring at angular distance jump_distance from center.
        circle_points = (
            ring_radius * (np.cos(theta) * v + np.sin(theta) * z)
            + np.cos(jump_distance) * center
        )

        raw_indices = grid_tree.query_ball_point(circle_points, search_radius * buffer)

        if len(raw_indices) == 0:
            flattened = np.empty(0, dtype=np.int64)
        else:
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
    delta: str = "two",
    silent: bool = True,
):
    return (
        build_antipodal_neighborhood(grid_points, jump_distance, delta, silent)
        if antipodal
        else build_nonantipodal_neighborhood(grid_points, jump_distance, delta, silent)
    )


def build_antipodal_neighborhood(
    grid_points: np.ndarray,
    jump_distance: float,
    delta: str = "two",
    silent: bool = True,
):
    discrete_functions = {
        "two": delta_two,
        "one_star": delta_one_star,
        "two_star": delta_two_star,
    }

    interaction_spreads = {
        "two": 2.0,
        "one_star": 1.5,
        "two_star": 2.5,
    }

    delta_key = (
        delta if isinstance(delta, str) and delta in discrete_functions else "two"
    )
    discrete_function = discrete_functions[delta_key]
    interaction_spread = interaction_spreads[delta_key]

    num_points = len(grid_points)
    if num_points % 2 != 0:
        raise ValueError("Number of points must be even for antipodal neighborhood.")
    num_spins = num_points // 2  # Every antipodal pair is converted to a single spin
    cell_length = np.sqrt((4 * np.pi) / num_points)
    jump_distance = np.abs(jump_distance) % np.pi

    neighborhood = []
    pruned_neighborhood = neighbor_pruning(
        grid_points[: len(grid_points) // 2],
        grid_points,
        jump_distance,
        interaction_spread,
        silent=silent,
    )
    for i in tqdm(range(num_spins), disable=silent):
        pruned_spins = pruned_neighborhood[i]
        first_half = pruned_spins[pruned_spins < num_spins]
        second_half = pruned_spins[pruned_spins >= num_spins]
        aligned_distances = get_distances(grid_points[first_half], grid_points[i])
        positive_contributions, positive_bool, total_positive = calculate_contributions(
            jump_distance, cell_length, aligned_distances, discrete_function
        )

        anti_aligned_distances = get_distances(grid_points[second_half], grid_points[i])
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
    delta: str = "two",
    silent: bool = True,
):
    discrete_functions = {
        "two": delta_two,
        "one_star": delta_one_star,
        "two_star": delta_two_star,
    }

    interaction_spreads = {
        "two": 2.0,
        "one_star": 1.5,
        "two_star": 2.5,
    }

    delta_key = (
        delta if isinstance(delta, str) and delta in discrete_functions else "two"
    )
    discrete_function = discrete_functions[delta_key]
    interaction_spread = interaction_spreads[delta_key]

    num_spins = len(grid_points)
    cell_length = np.sqrt((4 * np.pi) / num_spins)
    jump_distance = np.abs(jump_distance) % np.pi

    neighborhood = []
    pruned_neighborhood = neighbor_pruning(
        grid_points, grid_points, jump_distance, interaction_spread, silent=silent
    )
    for i in tqdm(range(num_spins), disable=silent):
        pruned_spins = pruned_neighborhood[i]
        distances = get_distances(grid_points[pruned_spins], grid_points[i])
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


def euclid_distances(grid_points: np.ndarray, spin: np.ndarray) -> np.ndarray:
    distances = np.abs(grid_points - spin)
    distances = np.linalg.norm(distances, axis=1)
    return distances


def neighbor_pruning_2d(
    search_points: np.ndarray,
    grid_points: np.ndarray,
    jump_distance: float,
    interaction_spread: float,
    silent: bool = True,
):
    """
    Return a list of index arrays containing candidate neighbor indices
    on `grid_points` for each point in `search_points`.

    Geometry
    --------
    - Standard Euclidean 2D space.
    - For each search point, a circle of radius `jump_distance` is sampled.
    - KD-tree queries are performed around those circle samples.
    - Returned indices are a superset suitable for pruning.
    """

    search_points = np.asarray(search_points, dtype=float)
    grid_points = np.asarray(grid_points, dtype=float)

    if search_points.ndim != 2 or search_points.shape[1] != 2:
        raise ValueError("search_points must have shape (N,2)")
    if grid_points.ndim != 2 or grid_points.shape[1] != 2:
        raise ValueError("grid_points must have shape (M,2)")
    if grid_points.shape[0] == 0:
        raise ValueError("grid_points must be non-empty")

    if not np.all(np.isfinite(search_points)) or not np.all(np.isfinite(grid_points)):
        raise ValueError("inputs must contain only finite values")

    grid_tree = sci.spatial.cKDTree(grid_points)

    num_grid_points = grid_points.shape[0]

    # Estimate average spacing assuming uniform distribution
    cell_length = np.sqrt(1.0 / num_grid_points)

    radius = jump_distance

    # Choose number of circle samples
    approx_points = int(np.ceil((2.0 * np.pi * radius) / max(cell_length, 1e-15)))
    num_circle_points = max(8, approx_points)

    # Distance from midpoint between adjacent sample points
    midpoint_offset = radius * np.sin(np.pi / (2.0 * num_circle_points))

    search_radius = np.sqrt(
        (interaction_spread * cell_length) ** 2 + midpoint_offset**2
    )

    buffer = 1.05

    theta = np.linspace(0.0, 2.0 * np.pi, num_circle_points, endpoint=False)

    neighbor_indices = []

    for center in tqdm(search_points, disable=silent):

        circle_points = np.column_stack(
            (
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta),
            )
        )

        raw_indices = grid_tree.query_ball_point(circle_points, search_radius * buffer)

        nonempty = [np.asarray(x, dtype=np.int64) for x in raw_indices if len(x) > 0]

        if nonempty:
            flattened = np.unique(np.concatenate(nonempty))
        else:
            flattened = np.empty(0, dtype=np.int64)

        neighbor_indices.append(flattened)

    return neighbor_indices


def build_euclid_neighborhood(
    grid_points: np.ndarray,
    jump_distance: float,
    cell_length: float,
    delta: str = "two",
    silent: bool = True,
    no_prune: bool = False,
):
    discrete_functions = {
        "two": delta_two,
        "one_star": delta_one_star,
        "two_star": delta_two_star,
    }

    interaction_spreads = {
        "two": 2.0,
        "one_star": 1.5,
        "two_star": 2.5,
    }

    delta_key = (
        delta if isinstance(delta, str) and delta in discrete_functions else "two"
    )
    discrete_function = discrete_functions[delta_key]
    interaction_spread = interaction_spreads[delta_key]

    neighborhood = []
    pruned_neighborhood = neighbor_pruning_2d(
        grid_points, grid_points, jump_distance, interaction_spread, silent=silent
    )
    for i in tqdm(range(grid_points.shape[0]), disable=silent):
        pruned_spins = pruned_neighborhood[i]
        if no_prune:
            pruned_spins = np.arange(grid_points.shape[0])
        distances = euclid_distances(grid_points[pruned_spins], grid_points[i])
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
