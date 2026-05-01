"""
Regular 2D lattice coordinate builders.

These functions return raw candidate-site coordinates. The matching
`System.*_lattice()` constructors apply fixed-density Euclidean normalization by
occupied-site count.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


Point2D = tuple[float, float]
Spacing2D = float | tuple[float, float] | Sequence[float] | NDArray[np.float64] | None


def _validate_lattice_size(width: int, height: int) -> tuple[int, int]:
    if isinstance(width, bool) or isinstance(height, bool):
        raise TypeError("width and height must be integers.")
    if not isinstance(width, (int, np.integer)) or not isinstance(
        height,
        (int, np.integer),
    ):
        raise TypeError("width and height must be integers.")
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must both be positive.")
    return width, height


def _validate_spacing(spacing: float) -> float:
    if isinstance(spacing, bool):
        raise TypeError("spacing must be a positive finite number.")
    spacing = float(spacing)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be a positive finite number.")
    return spacing


def _default_square_spacing(width: int, height: int) -> float:
    return 1.0 / np.sqrt(float(width) * float(height))


def _default_triangular_spacing(width: int, height: int) -> float:
    return np.sqrt(2.0 / (np.sqrt(3.0) * float(width) * float(height)))


def _default_hexagonal_spacing(width: int, height: int) -> float:
    return np.sqrt(2.0 / (3.0 * np.sqrt(3.0) * float(width) * float(height)))


def _default_rectangular_spacing(width: int, height: int) -> tuple[float, float]:
    return 1.0 / int(width), 1.0 / int(height)


def _resolve_scalar_spacing(spacing: Spacing2D, default_spacing: float) -> float:
    if spacing is None:
        return float(default_spacing)
    return _validate_scalar_spacing(spacing)


def _resolve_spacing_2d(spacing: Spacing2D, width: int, height: int) -> tuple[float, float]:
    if spacing is None:
        return _default_rectangular_spacing(width, height)
    return _validate_spacing_2d(spacing)


def _validate_scalar_spacing(spacing: Spacing2D) -> float:
    if np.isscalar(spacing):
        return _validate_spacing(float(spacing))
    values = np.asarray(spacing, dtype=float)
    if values.shape == (1,):
        return _validate_spacing(float(values[0]))
    raise ValueError("spacing must be scalar for this lattice.")


def _validate_spacing_2d(spacing: Spacing2D) -> tuple[float, float]:
    if np.isscalar(spacing):
        value = _validate_spacing(float(spacing))
        return value, value

    values = np.asarray(spacing, dtype=float)
    if values.shape != (2,):
        raise ValueError("2D spacing must be a scalar or a length-2 sequence.")
    x_spacing = _validate_spacing(float(values[0]))
    y_spacing = _validate_spacing(float(values[1]))
    return x_spacing, y_spacing


def _validate_origin(origin: Point2D | Sequence[float]) -> NDArray[np.float64]:
    values = np.asarray(origin, dtype=float)
    if values.shape != (2,):
        raise ValueError("origin must be a length-2 coordinate.")
    if not np.all(np.isfinite(values)):
        raise ValueError("origin must contain finite values.")
    return values


def _finish_points(
    points: NDArray[np.float64],
    *,
    origin: Point2D | Sequence[float],
    center: bool,
) -> NDArray[np.float64]:
    points = np.asarray(points, dtype=np.float64)
    if center:
        bounds_center = (np.min(points, axis=0) + np.max(points, axis=0)) / 2.0
        points = points - bounds_center
    points = points + _validate_origin(origin)
    return np.ascontiguousarray(points, dtype=np.float64)


def rectangular_lattice(
    width: int,
    height: int,
    *,
    spacing: Spacing2D = None,
    origin: Point2D | Sequence[float] = (0.0, 0.0),
    center: bool = False,
) -> NDArray[np.float64]:
    """
    Return row-major 2D coordinates for a rectangular lattice.

    This low-level grid builder works in coordinate units only. When `spacing`
    is omitted, the default spans approximately unit area across all candidate
    sites. The `System.rectangular_lattice()` constructor may rescale these
    coordinates so occupied-site area, not candidate-site count, sets the
    physical length scale.
    """
    width, height = _validate_lattice_size(width, height)
    x_spacing, y_spacing = _resolve_spacing_2d(spacing, width, height)

    x_coords = np.arange(width, dtype=np.float64) * x_spacing
    y_coords = np.arange(height, dtype=np.float64) * y_spacing
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    return _finish_points(points, origin=origin, center=center)


def square_lattice(
    width: int,
    height: int,
    *,
    spacing: float | None = None,
    origin: Point2D | Sequence[float] = (0.0, 0.0),
    center: bool = False,
) -> NDArray[np.float64]:
    """
    Return row-major 2D coordinates for a square lattice.

    Omitting `spacing` gives candidate-site spacing `1 / sqrt(width * height)`.
    Use `System.square_lattice()` when the coordinates should be normalized by
    the number of occupied sites in a fixed-density Euclidean system.
    """
    width, height = _validate_lattice_size(width, height)
    spacing = _resolve_scalar_spacing(spacing, _default_square_spacing(width, height))
    return rectangular_lattice(
        width,
        height,
        spacing=(spacing, spacing),
        origin=origin,
        center=center,
    )


def triangular_lattice(
    width: int,
    height: int,
    *,
    spacing: float | None = None,
    origin: Point2D | Sequence[float] = (0.0, 0.0),
    center: bool = False,
) -> NDArray[np.float64]:
    """
    Return row-major 2D coordinates for a triangular Bravais lattice.

    The direct grid builder returns coordinates in its own spacing units.
    `System.triangular_lattice()` applies the fixed-density Euclidean
    normalization based on occupied sites by default.
    """
    width, height = _validate_lattice_size(width, height)
    spacing = _resolve_scalar_spacing(
        spacing,
        _default_triangular_spacing(width, height),
    )

    rows = np.arange(height, dtype=np.float64)
    cols = np.arange(width, dtype=np.float64)
    xx = cols[None, :] * spacing + rows[:, None] * spacing / 2.0
    yy = rows[:, None] * spacing * np.sqrt(3.0) / 2.0
    yy = np.broadcast_to(yy, xx.shape)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    return _finish_points(points, origin=origin, center=center)


def hexagonal_lattice(
    width: int,
    height: int,
    *,
    spacing: float | None = None,
    origin: Point2D | Sequence[float] = (0.0, 0.0),
    center: bool = False,
) -> NDArray[np.float64]:
    """Return 2D coordinates for a honeycomb/hexagonal tiling lattice.

    Each unit cell contributes two sites. `spacing` is the nearest-neighbor
    distance between honeycomb sites. `System.hexagonal_lattice()` applies the
    fixed-density Euclidean normalization based on occupied sites by default.
    """
    width, height = _validate_lattice_size(width, height)
    spacing = _resolve_scalar_spacing(
        spacing,
        _default_hexagonal_spacing(width, height),
    )

    a1 = np.array([np.sqrt(3.0) * spacing, 0.0], dtype=np.float64)
    a2 = np.array([np.sqrt(3.0) * spacing / 2.0, 1.5 * spacing], dtype=np.float64)
    basis = np.array([[0.0, 0.0], [0.0, spacing]], dtype=np.float64)

    points = []
    for row in range(height):
        for col in range(width):
            base = col * a1 + row * a2
            points.extend(base + basis_site for basis_site in basis)

    return _finish_points(np.array(points), origin=origin, center=center)


def honeycomb_lattice(
    width: int,
    height: int,
    *,
    spacing: float | None = None,
    origin: Point2D | Sequence[float] = (0.0, 0.0),
    center: bool = False,
) -> NDArray[np.float64]:
    """Alias for `hexagonal_lattice`."""
    return hexagonal_lattice(
        width,
        height,
        spacing=spacing,
        origin=origin,
        center=center,
    )


def regular_lattice(
    lattice: str,
    width: int,
    height: int,
    *,
    spacing: Spacing2D = None,
    origin: Point2D | Sequence[float] = (0.0, 0.0),
    center: bool = False,
) -> NDArray[np.float64]:
    """
    Dispatch to a built-in 2D regular lattice constructor.

    This returns raw coordinates from the selected grid builder. Use
    `System.regular_lattice()` to construct a fixed-density Euclidean system
    whose physical scale is normalized by occupied-site count.
    """
    if not isinstance(lattice, str):
        raise TypeError("lattice must be a string.")
    lattice_key = lattice.lower().replace("-", "_")
    if lattice_key in {"square", "sq"}:
        return square_lattice(
            width,
            height,
            spacing=_resolve_scalar_spacing(
                spacing,
                _default_square_spacing(width, height),
            ),
            origin=origin,
            center=center,
        )
    if lattice_key in {"rectangular", "rectangle", "rect"}:
        return rectangular_lattice(
            width,
            height,
            spacing=spacing,
            origin=origin,
            center=center,
        )
    if lattice_key in {"triangular", "triangle", "tri"}:
        return triangular_lattice(
            width,
            height,
            spacing=_resolve_scalar_spacing(
                spacing,
                _default_triangular_spacing(width, height),
            ),
            origin=origin,
            center=center,
        )
    if lattice_key in {"hexagonal", "hex", "honeycomb"}:
        return hexagonal_lattice(
            width,
            height,
            spacing=_resolve_scalar_spacing(
                spacing,
                _default_hexagonal_spacing(width, height),
            ),
            origin=origin,
            center=center,
        )

    valid = "square, rectangular, triangular, hexagonal, honeycomb"
    raise ValueError(f"Unknown lattice {lattice!r}. Expected one of: {valid}.")
