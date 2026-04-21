from __future__ import annotations

from collections.abc import Callable

import numpy as np


EXACT_DELTA_ATOL = 100.0 * np.finfo(float).eps


def calculate_prob_prefactor(n_points: int, theta: float) -> float:
    return 4 / (np.sin(theta) * n_points ** (3 / 2) * np.sqrt(np.pi))


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


def delta_two_star(
    spin_distance: np.ndarray, cell_length: float
) -> tuple[np.ndarray, np.ndarray]:
    def calculate_contributions(r: np.ndarray) -> np.ndarray:
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


def delta_exact(
    spin_distance: np.ndarray, cell_length: float
) -> tuple[np.ndarray, np.ndarray]:
    """Unit contribution only when distance error is roundoff-level zero."""
    tolerance = EXACT_DELTA_ATOL * max(1.0, float(abs(cell_length)))
    interacting = np.isclose(spin_distance, 0.0, rtol=0.0, atol=tolerance)
    return np.ones(np.count_nonzero(interacting), dtype=np.float64), interacting


def resolve_delta_kernel(delta: str | Callable) -> tuple[Callable, float, bool]:
    """Return (kernel function, interaction spread, exact_scan_required)."""
    if callable(delta):
        return delta, 2.0, False

    raw_delta = getattr(delta, "value", delta)
    delta_key = str(raw_delta).lower().replace("-", "_")
    aliases = {
        "two": "two",
        "one*": "one_star",
        "one_star": "one_star",
        "two*": "two_star",
        "two_star": "two_star",
        "delta": "exact",
        "dirac": "exact",
        "exact": "exact",
        "exact_delta": "exact",
        "real_delta": "exact",
    }
    canonical = aliases.get(delta_key, "two")
    discrete_functions = {
        "two": delta_two,
        "one_star": delta_one_star,
        "two_star": delta_two_star,
        "exact": delta_exact,
    }
    interaction_spreads = {
        "two": 2.0,
        "one_star": 1.5,
        "two_star": 2.5,
        "exact": 0.0,
    }
    return (
        discrete_functions[canonical],
        interaction_spreads[canonical],
        canonical == "exact",
    )


def calculate_contributions(
    jump_distance: float,
    cell_length: float,
    distances: np.ndarray,
    delta_func: Callable,
) -> tuple[np.ndarray, np.ndarray, float]:
    abs_distances = np.abs(jump_distance - distances)
    contributions, interacting_bool = delta_func(abs_distances, cell_length)
    return contributions, interacting_bool, float(np.sum(contributions))
