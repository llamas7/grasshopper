from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .core import KernelSpec


EXACT_DELTA_ATOL = 100.0 * np.finfo(float).eps
ZERO_CONTRIBUTION_ATOL = 100.0 * np.finfo(float).eps


def calculate_prob_prefactor(n_points: int, theta: float) -> float:
    return 4 / (np.sin(theta) * n_points ** (3 / 2) * np.sqrt(np.pi))


def cosine_kernel(
    spin_distance: np.ndarray, cell_length: float
) -> tuple[np.ndarray, np.ndarray]:
    r = spin_distance / cell_length
    r_bool = r < 2
    return (1 + np.cos(np.pi * r[r_bool] / 2)) / 4.0, r_bool


def cell_averaged_hat_kernel(
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


def cell_averaged_cosine_kernel(
    spin_distance: np.ndarray, cell_length: float
) -> tuple[np.ndarray, np.ndarray]:
    def calculate_contributions(r: np.ndarray) -> np.ndarray:
        contributions = np.zeros(len(r))
        bool_mask = r <= 1.5
        contributions[bool_mask] = (1 / (4 * np.pi)) * (
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


def delta_exact(
    spin_distance: np.ndarray, cell_length: float
) -> tuple[np.ndarray, np.ndarray]:
    """Unit contribution only when distance error is roundoff-level zero."""
    tolerance = EXACT_DELTA_ATOL * max(1.0, float(abs(cell_length)))
    interacting = np.isclose(spin_distance, 0.0, rtol=0.0, atol=tolerance)
    return np.ones(np.count_nonzero(interacting), dtype=np.float64), interacting


def gaussian_kernel(
    spin_distance: np.ndarray,
    cell_length: float,
    *,
    sigma: float,
    cutoff: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``h G_sigma`` weights and their truncation mask."""
    if cutoff is None:
        interacting = np.ones(spin_distance.shape, dtype=bool)
    else:
        interacting = spin_distance <= cutoff * sigma
    errors = spin_distance[interacting]
    normalization = cell_length / (np.sqrt(2.0 * np.pi) * sigma)
    return normalization * np.exp(-0.5 * (errors / sigma) ** 2), interacting


@dataclass(frozen=True)
class ResolvedKernel:
    function: Callable
    support_width: float | None
    exact_scan_required: bool
    spec: KernelSpec | None


# Compatibility aliases for the original discrete-delta terminology.
delta_two = cosine_kernel
delta_one_star = cell_averaged_hat_kernel
delta_two_star = cell_averaged_cosine_kernel


def resolve_kernel(
    kernel: KernelSpec | str | Callable,
    *,
    cell_length: float,
) -> ResolvedKernel:
    """Resolve a kernel and its physical support width."""
    if callable(kernel) and not isinstance(kernel, KernelSpec):
        return ResolvedKernel(kernel, 2.0 * cell_length, False, None)

    spec = KernelSpec.from_value(kernel)
    discrete_functions = {
        "cosine": cosine_kernel,
        "cell_averaged_hat": cell_averaged_hat_kernel,
        "cell_averaged_cosine": cell_averaged_cosine_kernel,
        "exact": delta_exact,
    }
    interaction_spreads = {
        "cosine": 2.0,
        "cell_averaged_hat": 1.5,
        "cell_averaged_cosine": 2.5,
        "exact": 0.0,
    }
    if spec.kind == "gaussian":
        function = lambda distances, h: gaussian_kernel(
            distances,
            h,
            sigma=float(spec.sigma),
            cutoff=spec.cutoff,
        )
        support_width = (
            None if spec.cutoff is None else float(spec.cutoff * spec.sigma)
        )
        return ResolvedKernel(
            function=function,
            support_width=support_width,
            exact_scan_required=spec.cutoff is None,
            spec=spec,
        )
    return ResolvedKernel(
        function=discrete_functions[spec.kind],
        support_width=interaction_spreads[spec.kind] * cell_length,
        exact_scan_required=spec.kind == "exact",
        spec=spec,
    )


def resolve_delta_kernel(
    delta: KernelSpec | str | Callable,
) -> tuple[Callable, float, bool]:
    """Deprecated resolver returning support in units of a unit cell."""
    resolved = resolve_kernel(delta, cell_length=1.0)
    spread = np.inf if resolved.support_width is None else resolved.support_width
    return resolved.function, float(spread), resolved.exact_scan_required


def calculate_contributions(
    jump_distance: float,
    cell_length: float,
    distances: np.ndarray,
    delta_func: Callable,
) -> tuple[np.ndarray, np.ndarray, float]:
    abs_distances = np.abs(jump_distance - distances)
    contributions, interacting_bool = delta_func(abs_distances, cell_length)
    nonzero_bool = np.abs(contributions) > ZERO_CONTRIBUTION_ATOL
    if not np.all(nonzero_bool):
        interacting_indices = np.flatnonzero(interacting_bool)
        interacting_bool = interacting_bool.copy()
        interacting_bool[interacting_indices[~nonzero_bool]] = False
        contributions = contributions[nonzero_bool]
    return contributions, interacting_bool, float(np.sum(contributions))
