from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DistanceMetric = str | Callable[[np.ndarray, np.ndarray], np.ndarray | float]


@dataclass(frozen=True)
class ResolvedMetric:
    distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    kd_tree_p: float | None
    name: str | None


def spherical_distances(grid_points: np.ndarray, spin: np.ndarray) -> np.ndarray:
    chord_distances = np.linalg.norm(grid_points - spin, axis=1)
    return 2 * np.arcsin(np.clip(chord_distances / 2, 0, 1))


def _builtin_metric(name: str) -> ResolvedMetric:
    metric_name = name.lower().replace("-", "_")
    if metric_name in {"spherical", "angular", "geodesic"}:
        return ResolvedMetric(spherical_distances, None, "spherical")
    if metric_name in {"euclidean", "l2"}:
        return ResolvedMetric(
            lambda points, spin: np.linalg.norm(points - spin, axis=1),
            2.0,
            "euclidean",
        )
    if metric_name in {"manhattan", "cityblock", "taxicab", "l1"}:
        return ResolvedMetric(
            lambda points, spin: np.sum(np.abs(points - spin), axis=1),
            1.0,
            "manhattan",
        )
    if metric_name in {"chebyshev", "linf", "l_inf", "infinity"}:
        return ResolvedMetric(
            lambda points, spin: np.max(np.abs(points - spin), axis=1),
            np.inf,
            "chebyshev",
        )
    raise ValueError(
        "Unknown distance metric "
        f"{name!r}. Use 'spherical', 'euclidean', 'manhattan', "
        "'chebyshev', or a callable."
    )


def _call_custom_metric(
        metric: Callable[[np.ndarray, np.ndarray], np.ndarray | float],
        points: np.ndarray,
        spin: np.ndarray,
) -> np.ndarray:
    if len(points) == 0:
        return np.array([], dtype=np.float64)

    try:
        result = np.asarray(metric(points, spin), dtype=np.float64)
        if result.shape == (len(points),):
            return result
        if result.ndim == 0 and len(points) == 1:
            return np.array([float(result)], dtype=np.float64)
    except Exception:
        pass

    distances = np.asarray(
        [metric(point, spin) for point in points], dtype=np.float64
    )
    if distances.shape != (len(points),):
        raise ValueError(
            "Custom distance metric must return one scalar per point. "
            f"Expected shape {(len(points),)}, got {distances.shape}."
        )
    return distances


def resolve_distance_metric(distance_metric: DistanceMetric) -> ResolvedMetric:
    if isinstance(distance_metric, str):
        return _builtin_metric(distance_metric)
    if callable(distance_metric):
        return ResolvedMetric(
            lambda points, spin: _call_custom_metric(distance_metric, points, spin),
            None,
            None,
        )
    raise TypeError("distance_metric must be a string or callable")


def validate_distances(distances: np.ndarray) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError("distance metric must return a 1D distance array")
    if np.any(~np.isfinite(distances)):
        raise ValueError("distance metric returned non-finite values")
    if np.any(distances < 0):
        raise ValueError("distance metric returned negative values")
    return distances
