from __future__ import annotations

from collections.abc import Callable
from math import gamma
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

from .core import (
    ConstraintModel,
    Geometry,
    KernelSpec,
    Neighborhood,
    NonAntipodalNeighborhoodCSR,
)
from .engines import get_engine
from .metrics import resolve_distance_metric

if TYPE_CHECKING:
    from .system import System


ProbabilityNormalization = str


def euclidean_shell_measure(radius: float, dimension: int) -> float:
    """Surface measure of a radius-``radius`` sphere in ``dimension`` dimensions."""
    radius = float(radius)
    if radius < 0:
        return 0.0
    return float(
        2.0
        * np.pi ** (dimension / 2.0)
        * radius ** (dimension - 1)
        / gamma(dimension / 2.0)
    )


def spherical_shell_measure(angle: float) -> float:
    """Circumference of a geodesic circle on the unit two-sphere."""
    angle = float(angle)
    if angle < 0 or angle > np.pi:
        return 0.0
    return float(2.0 * np.pi * np.sin(angle))


def _validate_spins(
    system: "System",
    spins: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    normalized = np.asarray(spins)
    if normalized.dtype != np.bool_:
        raise TypeError(f"spins must be boolean, got {normalized.dtype!r}.")
    expected_shape = (
        (int(system.system_shape),)
        if isinstance(system.system_shape, int)
        else tuple(system.system_shape)
    )
    if normalized.shape != expected_shape:
        raise ValueError(
            f"spins has shape {normalized.shape}, expected {expected_shape}."
        )
    if (
        system.geometry == Geometry.EUCLIDEAN
        and int(np.count_nonzero(normalized)) != system.num_spins
    ):
        raise ValueError(
            "Euclidean probability requires the configured fixed density of "
            f"{system.num_spins} occupied sites."
        )
    return np.ascontiguousarray(normalized, dtype=np.bool_)


def _cell_measure(system: "System") -> float:
    if system.geometry == Geometry.SPHERICAL:
        return float(4.0 * np.pi / len(system.grid))
    return float(1.0 / system.num_spins)


def _cell_length(system: "System") -> float:
    if system.geometry == Geometry.SPHERICAL:
        return float(np.sqrt(_cell_measure(system)))
    return float(_cell_measure(system) ** (1.0 / system.dimension))


def _starting_measure(
    system: "System",
    spins: NDArray[np.bool_] | None,
) -> float:
    cell_measure = _cell_measure(system)
    if system.constraint in (
        ConstraintModel.ANTIPODAL,
        ConstraintModel.INDEPENDENT,
    ):
        return float(system.num_spins * cell_measure)
    if spins is None:
        occupied = system.num_spins
    else:
        occupied = int(np.count_nonzero(spins))
    starting_measure = float(occupied * cell_measure)
    if starting_measure <= 0:
        raise ZeroDivisionError(
            "Cannot calculate probability for a state with zero occupied measure."
        )
    return starting_measure


def _default_shell_geometry(
    system: "System",
) -> tuple[Callable[[float], float], tuple[float, float]]:
    metric = system.distance_metric
    if metric is None:
        metric_name = (
            "spherical"
            if system.geometry == Geometry.SPHERICAL
            else "euclidean"
        )
    else:
        metric_name = resolve_distance_metric(metric).name

    if system.geometry == Geometry.SPHERICAL and metric_name == "spherical":
        return spherical_shell_measure, (0.0, float(np.pi))
    if system.geometry == Geometry.EUCLIDEAN and metric_name == "euclidean":
        return (
            lambda radius: euclidean_shell_measure(radius, system.dimension),
            (0.0, float("inf")),
        )
    raise ValueError(
        "Continuum probability normalization is not defined for this geometry "
        "and distance metric. Pass shell_measure=... and optionally "
        "radial_bounds=... to compute_neighborhood(), or use "
        "normalization='graph'."
    )


def _shell_geometry(
    system: "System",
) -> tuple[Callable[[float], float], tuple[float, float]]:
    if system.shell_measure is None:
        return _default_shell_geometry(system)
    bounds = system.radial_bounds
    if bounds is None:
        bounds = (
            (0.0, float(np.pi))
            if system.geometry == Geometry.SPHERICAL
            else (0.0, float("inf"))
        )
    return system.shell_measure, bounds


def _gaussian_radial_normalization(
    *,
    jump_distance: float,
    kernel: KernelSpec,
    shell_measure: Callable[[float], float],
    radial_bounds: tuple[float, float],
) -> float:
    sigma = float(kernel.sigma)
    lower, upper = radial_bounds
    if kernel.cutoff is not None:
        width = float(kernel.cutoff * sigma)
        lower = max(lower, jump_distance - width)
        upper = min(upper, jump_distance + width)
    if upper <= lower:
        raise ZeroDivisionError("The Gaussian radial normalization is zero.")

    gaussian_scale = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)

    def integrand(radius: float) -> float:
        shell = float(shell_measure(radius))
        if not np.isfinite(shell) or shell < 0:
            raise ValueError(
                "shell_measure must return finite, nonnegative values."
            )
        error = (radius - jump_distance) / sigma
        return shell * gaussian_scale * np.exp(-0.5 * error * error)

    value, _ = quad(integrand, lower, upper, epsabs=1e-12, epsrel=1e-10)
    if not np.isfinite(value) or value <= 0:
        raise ZeroDivisionError("The Gaussian radial normalization is zero.")
    return float(value)


def continuum_probability_prefactor(
    system: "System",
    spins: NDArray[np.bool_] | None = None,
) -> float:
    """Return the multiplier that converts stored pair energy to probability."""
    if system.jump_distance is None:
        raise RuntimeError("Neighborhood construction metadata is unavailable.")
    kernel = system.kernel
    if not isinstance(kernel, KernelSpec):
        raise ValueError(
            "Continuum probability is unavailable for callable kernels; use "
            "normalization='graph'."
        )
    if kernel.kind == "exact":
        raise ValueError(
            "Continuum probability is unavailable for the exact diagnostic "
            "kernel; use normalization='graph'."
        )

    jump_distance = float(system.jump_distance)
    shell_measure, radial_bounds = _shell_geometry(system)
    if kernel.kind == "gaussian":
        radial_normalization = _gaussian_radial_normalization(
            jump_distance=jump_distance,
            kernel=kernel,
            shell_measure=shell_measure,
            radial_bounds=radial_bounds,
        )
    else:
        if jump_distance == 0 or (
            system.geometry == Geometry.SPHERICAL
            and jump_distance in (0.0, float(np.pi))
        ):
            raise ValueError(
                "Sharp continuum probability is undefined at a degenerate "
                "distance shell; use a Gaussian kernel or graph normalization."
            )
        radial_normalization = float(shell_measure(jump_distance))
        if not np.isfinite(radial_normalization) or radial_normalization <= 0:
            raise ZeroDivisionError("The radial shell normalization is zero.")

    cell_measure = _cell_measure(system)
    starting_measure = _starting_measure(system, spins)
    cell_length = _cell_length(system)
    return float(
        2.0
        * cell_measure**2
        / (starting_measure * cell_length * radial_normalization)
    )


def _iter_contribution_arrays(system: "System"):
    neighborhood = system.neighborhood
    if isinstance(neighborhood, NonAntipodalNeighborhoodCSR):
        yield neighborhood.contributions
        return
    if isinstance(neighborhood, list):
        for row in neighborhood:
            if isinstance(row, Neighborhood):
                yield row.positive_contributions
                yield row.negative_contributions
            else:
                yield row.contributions


def graph_probability(
    system: "System",
    spins: NDArray[np.bool_],
    *,
    energy: float,
) -> float:
    """Normalize successful weight by available outgoing graph weight."""
    for contributions in _iter_contribution_arrays(system):
        if np.any(contributions < -100.0 * np.finfo(float).eps):
            raise ValueError(
                "Graph probability requires nonnegative kernel contributions."
            )

    neighborhood = system.neighborhood
    if isinstance(neighborhood, NonAntipodalNeighborhoodCSR):
        available_weight = float(np.sum(neighborhood.spin_potential[spins]))
    elif isinstance(neighborhood, list):
        available_weight = float(sum(row.spin_potential for row in neighborhood))
    else:
        raise RuntimeError("Neighborhood has not been built.")
    if available_weight <= 0:
        raise ZeroDivisionError(
            "Cannot calculate graph probability with zero available weight."
        )
    return float(2.0 * energy / available_weight)


def calculate_probability(
    system: "System",
    spins: NDArray[np.bool_],
    *,
    normalization: ProbabilityNormalization,
    energy: float | None = None,
) -> float:
    """Calculate continuum or graph-normalized probability for one state."""
    normalized_spins = _validate_spins(system, spins)
    if system.neighborhood is None:
        raise RuntimeError(
            "Neighborhood has not been built. Call compute_neighborhood() first."
        )
    if energy is None:
        energy = float(
            get_engine(system.config).energy(normalized_spins, system.neighborhood)
        )

    mode = str(normalization).lower().replace("_", "-")
    if mode == "continuum":
        return float(
            energy * continuum_probability_prefactor(system, normalized_spins)
        )
    if mode == "graph":
        return graph_probability(system, normalized_spins, energy=energy)
    raise ValueError("normalization must be either 'continuum' or 'graph'.")
