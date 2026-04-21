from __future__ import annotations

import numpy as np
from typing import Any, Literal

from .core import Neighborhood, NonAntipodalNeighborhood

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import cupy as cp
except ImportError:
    cp = None


def require_cupy() -> Any:
    if cp is None:
        raise ImportError(
            "GPU neighborhood building requires CuPy. Install a CUDA-matched build, "
            "for example `pip install cupy-cuda12x`."
        )
    return cp

GPUDiscreteDeltaName = Literal["two", "one_star", "two_star"]


def calculate_prob_prefactor_gpu(num_points: int, theta: float) -> float:
    return 4.0 / (np.sin(theta) * num_points ** (3 / 2) * np.sqrt(np.pi))


# ============================================================
# GPU delta kernels
# ============================================================


def delta_two_gpu(
    spin_distance_gpu: cp.ndarray,
    cell_length: float,
) -> tuple[cp.ndarray, cp.ndarray]:
    r_gpu = spin_distance_gpu / cell_length
    valid_mask_gpu = r_gpu < 2.0
    contributions_gpu = cp.where(
        valid_mask_gpu,
        (1.0 + cp.cos(cp.pi * r_gpu / 2.0)) / 4.0,
        0.0,
    )
    return contributions_gpu, valid_mask_gpu


def delta_one_star_gpu(
    spin_distance_gpu: cp.ndarray,
    cell_length: float,
) -> tuple[cp.ndarray, cp.ndarray]:
    r_gpu = spin_distance_gpu / cell_length
    valid_mask_gpu = r_gpu <= 1.5

    inner_gpu = (3.0 / 4.0) - r_gpu**2
    outer_gpu = (9.0 / 8.0) - 1.5 * r_gpu + 0.5 * r_gpu**2

    contributions_gpu = cp.where(
        valid_mask_gpu,
        cp.where(r_gpu <= 0.5, inner_gpu, outer_gpu),
        0.0,
    )
    return contributions_gpu, valid_mask_gpu


def delta_two_star_gpu(
    spin_distance_gpu: cp.ndarray,
    cell_length: float,
) -> tuple[cp.ndarray, cp.ndarray]:
    r_gpu = spin_distance_gpu / cell_length
    valid_mask_gpu = r_gpu <= 2.5

    inner_gpu = (1.0 / (4.0 * cp.pi)) * (
        cp.pi
        + 2.0 * cp.sin((cp.pi / 4.0) * (2.0 * r_gpu + 1.0))
        - 2.0 * cp.sin((cp.pi / 4.0) * (2.0 * r_gpu - 1.0))
    )

    outer_gpu = -(1.0 / (8.0 * cp.pi)) * (
        -5.0 * cp.pi
        + 2.0 * cp.pi * r_gpu
        + 4.0 * cp.sin((cp.pi / 4.0) * (2.0 * r_gpu - 1.0))
    )

    contributions_gpu = cp.where(
        valid_mask_gpu,
        cp.where(r_gpu <= 1.5, inner_gpu, outer_gpu),
        0.0,
    )
    return contributions_gpu, valid_mask_gpu


def get_delta_kernel_gpu(
    delta_name: GPUDiscreteDeltaName,
):
    if delta_name == "two":
        return delta_two_gpu, 2.0
    if delta_name == "one_star":
        return delta_one_star_gpu, 1.5
    if delta_name == "two_star":
        return delta_two_star_gpu, 2.5
    raise ValueError(f"Unknown GPU delta kernel: {delta_name!r}")


# ============================================================
# GPU helpers
# ============================================================


def validate_grid_points_gpu(grid_points: np.ndarray) -> np.ndarray:
    grid_points_cpu = np.asarray(grid_points, dtype=np.float32)
    if grid_points_cpu.ndim != 2 or grid_points_cpu.shape[1] != 3:
        raise ValueError("grid_points must have shape (N, 3)")
    return np.ascontiguousarray(grid_points_cpu)


def get_euclid_distances_gpu(
    grid_points_gpu: cp.ndarray,
    spin_gpu: cp.ndarray,
) -> cp.ndarray:
    return cp.linalg.norm(grid_points_gpu - spin_gpu, axis=1)


def get_angular_distances_gpu(
    grid_points_gpu: cp.ndarray,
    spin_gpu: cp.ndarray,
) -> cp.ndarray:
    dot_products_gpu = cp.clip(grid_points_gpu @ spin_gpu, -1.0, 1.0)
    return cp.arccos(dot_products_gpu)


def compute_batched_angles_gpu(
    centers_gpu: cp.ndarray,
    candidate_points_gpu: cp.ndarray,
) -> cp.ndarray:
    dot_products_gpu = cp.clip(centers_gpu @ candidate_points_gpu.T, -1.0, 1.0)
    return cp.arccos(dot_products_gpu)


def get_support_window_cosines_gpu(
    jump_distance: float,
    interaction_spread: float,
    cell_length: float,
) -> tuple[float, float]:
    angular_width = interaction_spread * cell_length
    min_angle = max(0.0, jump_distance - angular_width)
    max_angle = min(np.pi, jump_distance + angular_width)
    cos_lo = float(np.cos(max_angle))
    cos_hi = float(np.cos(min_angle))
    return cos_lo, cos_hi


def compute_gpu_contributions_from_angles(
    angles_gpu: cp.ndarray,
    jump_distance: float,
    cell_length: float,
    delta_func_gpu,
    cos_lo: float,
    cos_hi: float,
) -> tuple[cp.ndarray, cp.ndarray]:
    dot_products_gpu = cp.cos(angles_gpu)
    annulus_mask_gpu = (dot_products_gpu >= cos_lo) & (dot_products_gpu <= cos_hi)

    abs_distances_gpu = cp.abs(jump_distance - angles_gpu)
    contributions_gpu, kernel_mask_gpu = delta_func_gpu(abs_distances_gpu, cell_length)

    valid_mask_gpu = annulus_mask_gpu & kernel_mask_gpu
    contributions_gpu = cp.where(valid_mask_gpu, contributions_gpu, 0.0)
    return contributions_gpu, valid_mask_gpu


def extract_gpu_rows_to_cpu(
    valid_mask_gpu: cp.ndarray,
    contributions_gpu: cp.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    valid_mask_cpu = cp.asnumpy(valid_mask_gpu)
    contributions_cpu = cp.asnumpy(contributions_gpu)

    row_indices_cpu: list[np.ndarray] = []
    row_contributions_cpu: list[np.ndarray] = []
    row_sums_cpu = np.empty(valid_mask_cpu.shape[0], dtype=np.float64)

    for row_idx in range(valid_mask_cpu.shape[0]):
        neighbor_indices_cpu = np.flatnonzero(valid_mask_cpu[row_idx]).astype(
            np.int64, copy=False
        )
        neighbor_contributions_cpu = contributions_cpu[
            row_idx, neighbor_indices_cpu
        ].astype(np.float64, copy=False)
        row_indices_cpu.append(neighbor_indices_cpu)
        row_contributions_cpu.append(neighbor_contributions_cpu)
        row_sums_cpu[row_idx] = neighbor_contributions_cpu.sum(dtype=np.float64)

    return row_indices_cpu, row_contributions_cpu, row_sums_cpu


# ============================================================
# Public GPU API
# ============================================================


def build_neighborhood_gpu(
    grid_points: np.ndarray,
    jump_distance: float,
    antipodal: bool = True,
    delta: GPUDiscreteDeltaName = "two",
    silent: bool = True,
    batch_size: int = 128,
) -> list[Neighborhood] | list[NonAntipodalNeighborhood]:
    require_cupy()
    if antipodal:
        return build_antipodal_neighborhood_gpu(
            grid_points=grid_points,
            jump_distance=jump_distance,
            delta=delta,
            silent=silent,
            batch_size=batch_size,
        )
    return build_nonantipodal_neighborhood_gpu(
        grid_points=grid_points,
        jump_distance=jump_distance,
        delta=delta,
        silent=silent,
        batch_size=batch_size,
    )


def build_antipodal_neighborhood_gpu(
    grid_points: np.ndarray,
    jump_distance: float,
    delta: GPUDiscreteDeltaName = "two",
    silent: bool = True,
    batch_size: int = 128,
) -> list[Neighborhood]:
    require_cupy()
    grid_points_cpu = validate_grid_points_gpu(grid_points)
    num_points = grid_points_cpu.shape[0]

    if num_points % 2 != 0:
        raise ValueError("Number of points must be even for antipodal neighborhood.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    num_spins = num_points // 2
    jump_distance = float(abs(jump_distance) % np.pi)
    cell_length = float(np.sqrt((4.0 * np.pi) / num_points))

    delta_func_gpu, interaction_spread = get_delta_kernel_gpu(delta)
    cos_lo, cos_hi = get_support_window_cosines_gpu(
        jump_distance=jump_distance,
        interaction_spread=interaction_spread,
        cell_length=cell_length,
    )

    grid_points_gpu = cp.asarray(grid_points_cpu, dtype=cp.float32)
    first_half_gpu = grid_points_gpu[:num_spins]
    second_half_gpu = grid_points_gpu[num_spins:]

    neighborhoods_cpu: list[Neighborhood] = []

    for batch_start in tqdm(range(0, num_spins, batch_size), disable=silent):
        batch_stop = min(batch_start + batch_size, num_spins)
        centers_gpu = first_half_gpu[batch_start:batch_stop]

        positive_angles_gpu = compute_batched_angles_gpu(centers_gpu, first_half_gpu)
        positive_contributions_gpu, positive_valid_mask_gpu = (
            compute_gpu_contributions_from_angles(
                angles_gpu=positive_angles_gpu,
                jump_distance=jump_distance,
                cell_length=cell_length,
                delta_func_gpu=delta_func_gpu,
                cos_lo=cos_lo,
                cos_hi=cos_hi,
            )
        )

        negative_angles_gpu = compute_batched_angles_gpu(centers_gpu, second_half_gpu)
        negative_contributions_gpu, negative_valid_mask_gpu = (
            compute_gpu_contributions_from_angles(
                angles_gpu=negative_angles_gpu,
                jump_distance=jump_distance,
                cell_length=cell_length,
                delta_func_gpu=delta_func_gpu,
                cos_lo=cos_lo,
                cos_hi=cos_hi,
            )
        )

        positive_indices_cpu, positive_values_cpu, positive_sums_cpu = (
            extract_gpu_rows_to_cpu(
                positive_valid_mask_gpu,
                positive_contributions_gpu,
            )
        )
        negative_indices_cpu, negative_values_cpu, negative_sums_cpu = (
            extract_gpu_rows_to_cpu(
                negative_valid_mask_gpu,
                negative_contributions_gpu,
            )
        )

        for local_idx in range(batch_stop - batch_start):
            neighborhoods_cpu.append(
                Neighborhood(
                    positive_spins=positive_indices_cpu[local_idx],
                    positive_contributions=positive_values_cpu[local_idx],
                    negative_spins=negative_indices_cpu[local_idx],
                    negative_contributions=negative_values_cpu[local_idx],
                    spin_potential=float(
                        positive_sums_cpu[local_idx] + negative_sums_cpu[local_idx]
                    ),
                )
            )

    return neighborhoods_cpu


def build_nonantipodal_neighborhood_gpu(
    grid_points: np.ndarray,
    jump_distance: float,
    delta: GPUDiscreteDeltaName = "two",
    silent: bool = True,
    batch_size: int = 128,
) -> list[NonAntipodalNeighborhood]:
    require_cupy()
    grid_points_cpu = validate_grid_points_gpu(grid_points)
    num_spins = grid_points_cpu.shape[0]

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    jump_distance = float(abs(jump_distance) % np.pi)
    cell_length = float(np.sqrt((4.0 * np.pi) / num_spins))

    delta_func_gpu, interaction_spread = get_delta_kernel_gpu(delta)
    cos_lo, cos_hi = get_support_window_cosines_gpu(
        jump_distance=jump_distance,
        interaction_spread=interaction_spread,
        cell_length=cell_length,
    )

    grid_points_gpu = cp.asarray(grid_points_cpu, dtype=cp.float32)

    neighborhoods_cpu: list[NonAntipodalNeighborhood] = []

    for batch_start in tqdm(range(0, num_spins, batch_size), disable=silent):
        batch_stop = min(batch_start + batch_size, num_spins)
        centers_gpu = grid_points_gpu[batch_start:batch_stop]

        angles_gpu = compute_batched_angles_gpu(centers_gpu, grid_points_gpu)
        contributions_gpu, valid_mask_gpu = compute_gpu_contributions_from_angles(
            angles_gpu=angles_gpu,
            jump_distance=jump_distance,
            cell_length=cell_length,
            delta_func_gpu=delta_func_gpu,
            cos_lo=cos_lo,
            cos_hi=cos_hi,
        )

        neighbor_indices_cpu, contribution_values_cpu, spin_sums_cpu = (
            extract_gpu_rows_to_cpu(
                valid_mask_gpu,
                contributions_gpu,
            )
        )

        for local_idx in range(batch_stop - batch_start):
            neighborhoods_cpu.append(
                NonAntipodalNeighborhood(
                    neighbor_spins=neighbor_indices_cpu[local_idx],
                    contributions=contribution_values_cpu[local_idx],
                    spin_potential=float(spin_sums_cpu[local_idx]),
                )
            )

    return neighborhoods_cpu
