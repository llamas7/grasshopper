from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from refactored.core import (
    AnnealData,
    DeltaType,
    NeighborhoodLike,
    Parameters,
    SystemType,
)
import xarray as xr
from numpy.typing import NDArray
import scipy.spatial as sci
from scipy.spatial import cKDTree
from .build_neighbors import build_euclid_neighborhood
from .annealing import simulated_annealing, thermalize_system

from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box


def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = 2 * np.ptp(vor.points, axis=0).max()

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v1 >= 0 and v2 >= 0:
                continue

            v = v1 if v1 >= 0 else v2
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def estimate_spacing_voronoi(
    points: np.ndarray, domain: Polygon | None = None
) -> float:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    if domain is None:
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        pad_x = 0.05 * max(xmax - xmin, 1e-12)
        pad_y = 0.05 * max(ymax - ymin, 1e-12)
        domain = box(xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y)

    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    areas = []
    for region in regions:
        poly = Polygon(vertices[region]).intersection(domain)
        if poly.is_empty:
            continue
        areas.append(poly.area)

    areas = np.asarray(areas, dtype=float)
    if len(areas) != len(points):
        raise ValueError("failed to compute one Voronoi area per point")

    return float(np.median(np.sqrt(areas)))


def estimate_spacing_knn(points: np.ndarray, k: int = 6) -> float:
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)  # first column is self-distance = 0
    local_scale = np.mean(dists[:, 1:], axis=1)
    return float(np.median(local_scale))


@dataclass
class EuclidSystem:
    grid: NDArray[np.float64]
    jump_distance: float
    num_spins: int
    delta: DeltaType = DeltaType.TWO
    parameters: Parameters = field(init=True, default_factory=Parameters)

    initialized: bool = field(init=False, default=False)
    system_type: SystemType = field(
        init=False, default=SystemType.NON_ANTIPODAL
    )  # Use the non-antipodal engine for Annealing
    neighborhood: list[NeighborhoodLike] | None = field(init=False, default=None)
    prob_prefactor: float = field(init=False, default=0.0)
    system_shape: tuple[int] | None = field(init=False, default=None)
    current_system: NDArray[np.float64] | None = field(init=False, default=None)
    current_energy: float = field(init=False, default=0.0)
    best_system: NDArray[np.float64] | None = field(init=False, default=None)
    best_energy: float = field(init=False, default=0.0)
    anneal_data: AnnealData | None = field(init=False, default=None)
    parameters_set: bool | None = field(init=False, default=False)
    up_spins: list[int] | None = field(init=False, default=None)
    down_spins: list[int] | None = field(init=False, default=None)

    def __post_init__(self):
        if isinstance(self.delta, str):
            try:
                self.delta = DeltaType(self.delta.lower())
            except ValueError:
                self.delta = DeltaType.TWO
        if self.grid.ndim < 2:
            raise ValueError(
                "Grid must be a ND array where each row is a point in space."
            )
        if not np.issubdtype(self.grid.dtype, np.number):
            raise TypeError("grid must contain numeric values")
        self.dim = self.grid.shape[1]

        # Convex Hull-based spacing estimation
        # self.spin_volume = 1 / self.num_spins
        # self.spin_length = self.spin_volume ** (1 / self.dim)
        # self.grid_volume = sci.ConvexHull(self.grid).volume
        # self.grid_length = (self.grid_volume / self.grid.shape[0]) ** (1 / self.dim)
        # self.lattice_scale = self.spin_length / self.grid_length
        # self.grid = self.grid * self.lattice_scale

        # KNN-based spacing estimation (alternative to convex hull volume)
        # self.grid_length = estimate_spacing_knn(self.grid, k=6)
        # self.lattice_scale = self.spin_length / self.grid_length
        # self.grid = self.grid * self.lattice_scale

        # Voronoi-based spacing estimation (alternative to convex hull volume)
        self.spin_volume = 1 / self.num_spins
        self.spin_length = self.spin_volume ** (1 / self.dim)
        self.grid_length = estimate_spacing_voronoi(self.grid)
        self.lattice_scale = self.spin_length / self.grid_length
        self.grid = self.grid * self.lattice_scale

        self.prob_prefactor = 10
        self.anneal_data = AnnealData(
            steps=self.parameters.starting_steps,
            temperature=self.parameters.starting_temp,
        )

    def compute_neighborhood(self, silent: bool = True) -> None:
        self.neighborhood = build_euclid_neighborhood(
            grid_points=self.grid,
            jump_distance=self.jump_distance,
            cell_length=self.spin_length,
            delta=self.delta,
            silent=silent,
            no_prune=True,
        )

    def _require_neighborhood(self) -> None:
        if self.neighborhood is None:
            raise RuntimeError(
                "Neighborhood has not been built. Call compute_neighborhood() first."
            )

    def _require_init(self) -> None:
        if not self.initialized:
            raise RuntimeError(
                "System needs to be initialized. Call random_init(), hemisphere_init(), tri_cog_init(), cos_cog_init(), or from_array_init() first."
            )

    def set_parameters(self, parameters: Parameters) -> None:
        self.parameters = parameters
        self.parameters_set = True

    # --- initializers ---
    def random_init(self) -> None:
        all_sites = np.arange(self.grid.shape[0])
        current = np.random.choice(all_sites, size=self.num_spins, replace=False)
        self.up_spins = current.tolist()
        self.down_spins = all_sites[~np.isin(all_sites, current)].tolist()
        self.current_system = np.zeros(self.grid.shape[0], dtype=bool)
        self.current_system[self.up_spins] = True
        self.best_system = self.current_system.copy()
        self.initialized = True

    def disk_init(self, axis: NDArray[np.float64]) -> None:
        pass

    def array_init(self, system_array: NDArray[np.float64]) -> None:
        pass

    # --- Annealing ---
    def anneal(self, save: bool = True, verbose: bool = False) -> None:
        self._require_neighborhood()
        self._require_init()
        simulated_annealing(self, save=save, verbose=verbose)

    def thermalize(
        self,
        target_acceptance: float = 0.6,
        tolerance: float = 0.05,
        steps_per_temp: int | None = None,
        max_adjustments: int = 20,
        k_p: float = 0.5,
        verbose: bool = False,
    ) -> None:
        """Adjust the annealing temperature to reach a target Metropolis acceptance.

        Parameters
        ----------
        target_acceptance : float, optional
            Desired acceptance probability for SA start, by default 0.6.
        tolerance : float, optional
            Allowed deviation |acc - target|, by default 0.05.
        steps_per_temp : int | None, optional
            MC steps per temperature probe. If None, defaults to 10 * num_spins.
        max_adjustments : int, optional
            Maximum number of temperature adjustments, by default 20.
        k_p : float, optional
            Proportional gain in log(T) update used by `thermalize_system`,
            by default 0.5.
        verbose : bool, optional
            If True, print thermalization diagnostics, by default False.
        """
        if steps_per_temp is None:
            steps_per_temp = self.num_spins * 10

        thermalize_system(
            system=self,
            target_acceptance=target_acceptance,
            tolerance=tolerance,
            steps_per_temp=steps_per_temp,
            max_adjustments=max_adjustments,
            verbose=verbose,
            k_p=k_p,
        )
