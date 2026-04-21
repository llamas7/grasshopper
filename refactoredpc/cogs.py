from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from .plot_system import convert_ss
from functools import partial

# ---------- Types ----------
SphericalPoint = tuple[float, float]  # (colatitude p, azimuth t) in radians


# ---------- Minimal geometry ----------
def to_cartesian(colatitude: float, azimuth: float) -> np.ndarray:
    """Convert spherical (colatitude p, azimuth t) -> Cartesian (x, y, z) under the given convention."""
    sp = np.sin(colatitude)
    return np.array(
        [sp * np.sin(azimuth), -sp * np.cos(azimuth), np.cos(colatitude)], dtype=float
    )


def normalized_lerp_path(
        start_sph: SphericalPoint,
        end_sph: SphericalPoint,
        num_samples: int = 1000,
) -> np.ndarray:
    """Linear blend in Cartesian space followed by renormalization (matches ParametricPlot3D + n1 behavior)."""
    v0 = to_cartesian(*start_sph)
    v1 = to_cartesian(*end_sph)
    u = np.linspace(0.0, 1.0, num_samples)[:, None]  # (num_samples, 1)
    pts = (1.0 - u) * v0 + u * v1
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


@dataclass(frozen=True)
class CogPattern:
    num_teeth: int
    vertex_offset: float  # eps

    @property
    def sector_angle(self) -> float:
        return np.pi / self.num_teeth

    @property
    def q(self) -> float:
        return np.pi / 2.0 - self.vertex_offset

    def cog_vertex(self, i: int) -> SphericalPoint:
        return (np.pi - self.q, i * self.sector_angle)

    def anticog_vertex(self, i: int) -> SphericalPoint:
        return (self.q, i * self.sector_angle)

    def left_edge(self, i: int, samples: int = 200) -> np.ndarray:
        return normalized_lerp_path(
            self.anticog_vertex(2 * i - 1), self.cog_vertex(2 * i), samples
        )

    def right_edge(self, i: int, samples: int = 200) -> np.ndarray:
        return normalized_lerp_path(
            self.cog_vertex(2 * i), self.anticog_vertex(2 * i + 1), samples
        )

    def all_edges(self, samples: int = 200) -> np.ndarray:
        """Concatenate all left and right edges for i = 1..k into a single (N, 3) array."""
        left = [self.left_edge(i, samples) for i in range(1, self.num_teeth + 1)]
        right = [self.right_edge(i, samples) for i in range(1, self.num_teeth + 1)]
        return np.concatenate(left + right, axis=0)


def cogwheel_interpd(num_cogs: float, cog_height: float):
    if num_cogs % 2 == 0:
        endpoint_y = float(cog_height)
    else:
        endpoint_y = 0.0

    pattern = CogPattern(num_teeth=num_cogs, vertex_offset=cog_height)
    edges = pattern.all_edges(samples=200)

    # Convert cog edges to (s, s) coords expected downstream
    surface_points = convert_ss(edges)
    # Clamp endpoints
    surface_points = np.vstack(
        (
            surface_points,
            [[-np.pi, np.pi / 2 + endpoint_y], [np.pi, np.pi / 2 + endpoint_y]],
        )
    )
    surface_points = surface_points[surface_points[:, 0].argsort()]

    interped = interp1d(surface_points[:, 0], surface_points[:, 1])
    return interped


def init_from_interp1d(grid, interped, antipodal=True):
    grid_ss = convert_ss(grid)
    if antipodal:
        system_size = len(grid_ss) // 2
    else:
        system_size = len(grid_ss)

    system = (grid_ss[:, 1] > interped(grid_ss[:, 0]))[:system_size]

    return system
