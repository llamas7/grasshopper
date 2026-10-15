from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .annealing import simulated_annealing, thermalize_system
from .build_neighbors import build_neighborhood
from .core import (
    AnnealData,
    AnnealSchedule,
    GreedyHistory,
    KernelSpec,
    NeighborhoodData,
    FixedDensityNeighborhoodCSR,
    OccupancyState,
    ParallelTemperingHistory,
    SystemState,
    SystemKind,
)
from .euclidean import (
    DistanceMetric,
    build_euclidean_neighborhood,
    normalize_euclidean_grid,
)
from .lattices import fibonacci_lattice, regular_lattice
from .rng import Seed
from .initialization import (
    init_random,
    init_random_hemisphere,
    init_from_array,
    init_cosine_cogwheel,
    init_triangle_cogwheel,
)
from .greedy import GreedyResult, greedy_optimization
from .engines import get_engine
from .parallel_tempering import ParallelTemperingResult, parallel_tempering
from .plot_system import plot_spin_system, graph_boundary
from .probability import (
    _cell_length,
    calculate_probability,
    continuum_probability_prefactor,
)


class System:
    """A spin system on a spherical or Euclidean grid.

    Choose a named constructor, build interactions with
    ``compute_neighborhood()``, initialize spins, then run ``anneal()``,
    ``greedy()``, or ``parallel_temper()``. ``thermalize()`` can first tune the
    temperature. Setup methods return the system so they can be chained.

    ``current_spins`` and ``best_spins`` hold the spin arrays. Energies are
    unnormalized; use ``current_probability`` and ``best_probability`` for
    normalized Grasshopper values. Optimization histories are available as
    ``anneal_data``, ``greedy_history``, and ``parallel_tempering_history``.

    For Euclidean systems, ``len(grid)`` counts candidate sites and
    ``num_spins`` counts occupied sites. Occupied volume is fixed at 1, giving
    cell length ``num_spins ** (-1 / dimension)``. Spherical systems derive
    their spin count from the grid; antipodal grids store antipodes in the
    second half of their rows.

    ``save()`` writes states, histories, and interaction metadata to NetCDF.
    Grid coordinates and neighborhood rows are not saved. Construct a system
    on the same grid before calling ``load()``; rebuild its neighborhood to
    resume optimization. Stored continuum probabilities remain available
    without rebuilding. ``clone()`` copies spins and histories while sharing
    the grid and neighborhood.
    """

    _SAVE_FORMAT_VERSION = "7.0"
    _SUPPORTED_SAVE_FORMAT_VERSIONS = {"4.0", "5.0", "6.0", "7.0"}

    # --- Construction ---

    def __init__(
        self,
        grid: NDArray[np.float64],
        kind: SystemKind | str,
        *,
        num_spins: int | None = None,
        _prescaled: bool = False,
    ) -> None:
        """
        Construct a system of the given ``kind`` from grid coordinates.

        For Euclidean configs, `num_spins` is the occupied-site count. The
        conserved occupied volume is fixed to 1.0, so the candidate grid is
        scaled to estimated volume `len(grid) / num_spins`.
        """
        self._initialize_grid(grid, kind, num_spins, _prescaled)
        self._kernel = KernelSpec.cosine()
        self._kernel_is_custom = False
        self._jump_distance = None
        self._jump_weights = None
        self._sampling_probabilities = None
        self._distance_metric = None
        self._distance_metric_is_custom = False
        self._shell_measure = None
        self._shell_measure_is_custom = False
        self._radial_bounds = None
        self._neighborhood = None
        self._initialized = False
        self._current_spins = None
        self._current_energy = 0.0
        self._best_spins = None
        self._best_energy = 0.0
        self._occupancy = None
        self._temperature = AnnealSchedule().starting_temp
        self._anneal_data = AnnealData()
        self._greedy_history = GreedyHistory()
        self._parallel_tempering_history = ParallelTemperingHistory()
        self._stored_probabilities: dict[str, float] = {}
        self._stored_angle_probabilities = {}

    @classmethod
    def complementary(cls, grid: NDArray[np.float64]) -> "System":
        """Construct a spherical antipodal complementary system."""
        return cls(grid, SystemKind.COMPLEMENTARY)

    @classmethod
    def independent(cls, grid: NDArray[np.float64]) -> "System":
        """Construct a spherical independent-spin system."""
        return cls(grid, SystemKind.INDEPENDENT)

    @classmethod
    def fixed_density(cls, grid: NDArray[np.float64]) -> "System":
        """Construct a spherical fixed-density system (half the sites occupied)."""
        return cls(grid, SystemKind.FIXED_DENSITY)

    @classmethod
    def euclidean(
        cls,
        grid: NDArray[np.float64],
        *,
        num_spins: int,
    ) -> "System":
        """
        Construct a fixed-density Euclidean system from candidate grid points.

        `num_spins` is the number of occupied sites, not necessarily the number
        of rows in `grid`. The candidate grid is always scaled to estimated
        volume `len(grid) / num_spins`, which makes each occupied spin represent
        area/volume `1 / num_spins`.
        """
        return cls(
            grid,
            SystemKind.EUCLIDEAN,
            num_spins=num_spins,
        )

    @classmethod
    def square_lattice(
        cls,
        width: int,
        height: int,
        *,
        num_spins: int | None = None,
        density: float | None = None,
        origin: tuple[float, float] = (0.0, 0.0),
        center: bool = False,
    ) -> "System":
        """
        Construct a 2D fixed-density Euclidean system on a square lattice.

        `num_spins` or `density` determines the occupied-site count. By default,
        coordinates are normalized so the physical length scale is set by that
        occupied-site count (`1 / sqrt(num_spins)` in 2D), not by
        `width * height`.
        """
        return cls.regular_lattice(
            "square", width, height,
            num_spins=num_spins, density=density, origin=origin, center=center,
        )

    @classmethod
    def rectangular_lattice(
        cls,
        width: int,
        height: int,
        *,
        num_spins: int | None = None,
        density: float | None = None,
        origin: tuple[float, float] = (0.0, 0.0),
        center: bool = False,
    ) -> "System":
        """
        Construct a 2D fixed-density Euclidean system on a rectangular lattice.

        The candidate lattice has `width * height` sites, but the Euclidean
        interaction scale is normalized from the occupied-site count resolved
        from `num_spins` or `density`.
        """
        return cls.regular_lattice(
            "rectangular", width, height,
            num_spins=num_spins, density=density, origin=origin, center=center,
        )

    @classmethod
    def triangular_lattice(
        cls,
        width: int,
        height: int,
        *,
        num_spins: int | None = None,
        density: float | None = None,
        origin: tuple[float, float] = (0.0, 0.0),
        center: bool = False,
    ) -> "System":
        """
        Construct a compact triangular patch with ``width * height`` sites.

        The lattice coordinates are normalized by occupied-site count by
        default, so refining the candidate lattice does not change the physical
        jump-distance scale when `num_spins` is unchanged.
        """
        return cls.regular_lattice(
            "triangular", width, height,
            num_spins=num_spins, density=density, origin=origin, center=center,
        )

    @classmethod
    def hexagonal_lattice(
        cls,
        width: int,
        height: int,
        *,
        num_spins: int | None = None,
        density: float | None = None,
        origin: tuple[float, float] = (0.0, 0.0),
        center: bool = False,
    ) -> "System":
        """
        Construct a compact honeycomb patch with ``2 * width * height`` sites.

        Each hexagonal unit cell contributes two candidate sites. The default
        Euclidean normalization uses the occupied-site count from `num_spins` or
        `density`, not the total candidate-site count, to set cell length.
        """
        return cls.regular_lattice(
            "hexagonal", width, height,
            num_spins=num_spins, density=density, origin=origin, center=center,
        )

    honeycomb_lattice = hexagonal_lattice

    @classmethod
    def fibonacci_lattice(
        cls,
        num_sites: int,
        *,
        num_spins: int | None = None,
        density: float | None = None,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> "System":
        """Construct a fixed-density system on a planar Fibonacci disk.

        ``num_sites`` is the candidate-site budget. Choose ``num_spins`` or
        ``density`` (default: half filling). The disk area is
        ``num_sites / num_spins``, so each occupied site represents area 1/k
        for k occupied sites. ``origin`` is the center in final physical units.
        """
        grid = fibonacci_lattice(num_sites, origin=origin)
        occupied = cls._resolve_regular_lattice_num_spins(
            num_sites, num_spins=num_spins, density=density,
        )
        # Use the known disk area rather than its enclosing bounding box.
        grid = (grid - origin) * np.sqrt(num_sites / occupied) + origin
        return cls(
            grid, SystemKind.EUCLIDEAN, num_spins=occupied, _prescaled=True
        )

    @classmethod
    def regular_lattice(
        cls,
        lattice: str,
        width: int,
        height: int,
        *,
        num_spins: int | None = None,
        density: float | None = None,
        origin: tuple[float, float] = (0.0, 0.0),
        center: bool = False,
    ) -> "System":
        """
        Construct a 2D fixed-density Euclidean system on a named regular lattice.

        Supported lattice names include square, rectangular, triangular,
        hexagonal, and honeycomb aliases. Default normalization uses the
        occupied-site count resolved from `num_spins` or `density`.
        """
        grid = regular_lattice(
            lattice,
            width,
            height,
            origin=origin,
            center=center,
        )
        occupied = cls._resolve_regular_lattice_num_spins(
            len(grid), num_spins=num_spins, density=density,
        )
        # Built-in grids have total lattice-cell area 1. Use that known area;
        # a bounding box includes empty corners around a circular patch.
        return cls(
            grid * np.sqrt(len(grid) / occupied),
            SystemKind.EUCLIDEAN,
            num_spins=occupied,
            _prescaled=True,
        )

    # --- Configuration and observables ---

    def __repr__(self) -> str:
        return (
            "System("
            f"kind={self.kind.value!r}, "
            f"grid_shape={tuple(int(dim) for dim in self.grid.shape)!r}, "
            f"jump_distance={self.jump_distance!r}, "
            f"kernel={self._kernel_display(self.kernel)!r}, "
            f"distance_metric={self.distance_metric!r}, "
            f"num_spins={self.num_spins}, "
            f"initialized={self.initialized}"
            ")"
        )

    @property
    def kind(self) -> SystemKind:
        return self._kind

    @property
    def grid(self) -> NDArray[np.float64]:
        return self._grid

    @property
    def jump_distance(self) -> float | tuple[float, ...] | None:
        return self._jump_distance

    @property
    def jump_weights(self) -> tuple[float, ...] | None:
        return self._jump_weights

    @property
    def kernel(self) -> KernelSpec | Callable | None:
        return self._kernel

    @property
    def distance_metric(self) -> DistanceMetric | None:
        return self._distance_metric

    @property
    def num_spins(self) -> int:
        return self._num_spins

    @property
    def dimension(self) -> int:
        return int(self.grid.shape[1])

    @property
    def antipodal(self) -> bool:
        return self.kind.antipodal

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def neighborhood(self) -> NeighborhoodData | None:
        return self._neighborhood

    @property
    def shell_measure(self) -> Callable[[float], float] | None:
        return self._shell_measure

    @property
    def radial_bounds(self) -> tuple[float, float] | None:
        return self._radial_bounds

    @property
    def system_shape(self) -> int | tuple[int, ...] | None:
        if self.kind is SystemKind.COMPLEMENTARY:
            return self.num_spins
        if self.kind is SystemKind.INDEPENDENT:
            return (self.num_spins, 2)
        return len(self.grid)

    @property
    def current_spins(self) -> NDArray[np.bool_] | None:
        return self._current_spins

    @current_spins.setter
    def current_spins(self, value: NDArray[np.bool_] | None) -> None:
        self._current_spins = value
        self._forget_stored_probabilities("current")

    @property
    def current_energy(self) -> float:
        return self._current_energy

    @current_energy.setter
    def current_energy(self, value: float) -> None:
        self._current_energy = float(value)
        self._forget_stored_probabilities("current")

    @property
    def best_spins(self) -> NDArray[np.bool_] | None:
        return self._best_spins

    @best_spins.setter
    def best_spins(self, value: NDArray[np.bool_] | None) -> None:
        self._best_spins = value
        self._forget_stored_probabilities("best")

    @property
    def best_energy(self) -> float:
        return self._best_energy

    @best_energy.setter
    def best_energy(self, value: float) -> None:
        self._best_energy = float(value)
        self._forget_stored_probabilities("best")

    def probability(
        self,
        spins: NDArray[np.bool_] | str = "current",
        *,
        normalization: str = "continuum",
    ) -> float:
        """Return the Grasshopper probability of a spin configuration.

        ``spins`` is ``"current"``, ``"best"``, or an explicit spin array.
        ``normalization`` is ``"continuum"`` (default) or ``"graph"``.
        """
        if isinstance(spins, str):
            if spins not in ("current", "best"):
                raise ValueError("spins must be 'current', 'best', or a spin array.")
            return self._state_probability(spins, normalization)
        return calculate_probability(self, spins, normalization=normalization)

    @property
    def current_probability(self) -> float:
        """Continuum probability of ``current_spins``."""
        return self._state_probability("current", "continuum")

    @property
    def best_probability(self) -> float:
        """Continuum probability of ``best_spins``."""
        return self._state_probability("best", "continuum")

    @property
    def anneal_data(self) -> AnnealData:
        return self._anneal_data

    @property
    def greedy_history(self) -> GreedyHistory:
        return self._greedy_history

    @property
    def parallel_tempering_history(self) -> ParallelTemperingHistory:
        return self._parallel_tempering_history

    @property
    def occupancy(self) -> OccupancyState | None:
        return self._occupancy

    def state_summary(self) -> dict[str, object]:
        """Return a compact snapshot of system configuration and readiness."""
        neighborhood_built = self.neighborhood is not None
        has_current_spins = self.current_spins is not None
        has_best_spins = self.best_spins is not None
        kernel = self.kernel
        metric = self.distance_metric
        kernel_summary = (
            "custom"
            if kernel is None and self._kernel_is_custom
            else self._kernel_display(kernel)
        )
        metric_summary = (
            "custom"
            if metric is None and self._distance_metric_is_custom
            else metric if isinstance(metric, str) or metric is None else repr(metric)
        )
        return {
            "kind": self.kind.value,
            "site_count": int(self.grid.shape[0]),
            "num_spins": int(self.num_spins),
            "dimension": int(self.dimension),
            "system_shape": self.system_shape,
            "neighborhood_built": neighborhood_built,
            "initialized": bool(self.initialized),
            "has_current_spins": has_current_spins,
            "has_best_spins": has_best_spins,
            "ready": neighborhood_built and bool(self.initialized),
            "jump_distance": self.jump_distance,
            "jump_weights": self.jump_weights,
            "sampling_probabilities": self.sampling_probabilities,
            "objective_type": "anticorrelation" if self.sampling_probabilities is not None else None,
            "kernel": kernel_summary,
            "distance_metric": metric_summary,
            "current_energy": float(self.current_energy),
            "best_energy": float(self.best_energy),
            "temperature": float(self.temperature),
        }

    @property
    def temperature(self) -> float:
        """Current runtime annealing temperature."""
        return float(self._temperature)

    @temperature.setter
    def temperature(self, value: float) -> None:
        temperature = float(value)
        if not np.isfinite(temperature):
            raise ValueError(f"temperature must be finite, got {value!r}.")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {value!r}.")
        self._temperature = temperature

    # --- Neighborhood and initial state ---

    def compute_neighborhood(
        self,
        jump_distance: float | Sequence[float] | None = None,
        *,
        jump_weights: Sequence[float] | None = None,
        kernel: KernelSpec | str | Callable | None = None,
        silent: bool = True,
        distance_metric: DistanceMetric | None = None,
        shell_measure: Callable[[float], float] | None = None,
        radial_bounds: tuple[float, float] | None = None,
    ) -> "System":
        """
        Build and store a neighborhood plus probability-normalization metadata.

        For Euclidean systems, kernel cell length is based on `self.num_spins`
        (occupied sites), not `len(self.grid)`. Thus refining the candidate grid
        leaves the physical jump-distance scale unchanged as long as the
        occupied-site count is unchanged.
        """
        self._sampling_probabilities = None
        self._stored_angle_probabilities.clear()
        using_saved_distance = jump_distance is None
        coerced_jump_distance = self._coerce_jump_distance(
            self.jump_distance if using_saved_distance else jump_distance
        )
        distance_count = (
            len(coerced_jump_distance)
            if isinstance(coerced_jump_distance, tuple)
            else 1
        )
        if jump_weights is None and using_saved_distance:
            jump_weights = self.jump_weights
        coerced_jump_weights = self._coerce_jump_weights(jump_weights, distance_count)
        self._jump_distance = coerced_jump_distance
        self._jump_weights = coerced_jump_weights
        resolved_kernel = self._resolve_kernel_for_compute(kernel)
        self._kernel = resolved_kernel
        self._kernel_is_custom = not isinstance(resolved_kernel, KernelSpec)
        metric = self._resolve_distance_metric_for_compute(distance_metric)
        self._distance_metric = metric
        self._distance_metric_is_custom = not isinstance(metric, str)

        if shell_measure is not None:
            if not callable(shell_measure):
                raise TypeError("shell_measure must be callable.")
            self._shell_measure = shell_measure
            self._shell_measure_is_custom = True
        elif (
            self._shell_measure_is_custom
            and self._shell_measure is None
        ):
            raise ValueError(
                "The saved normalization metadata used a custom shell_measure. "
                "Pass shell_measure=... to compute_neighborhood() to rebuild it."
            )
        if radial_bounds is not None:
            self._radial_bounds = self._coerce_radial_bounds(
                radial_bounds
            )

        if self.kind is SystemKind.EUCLIDEAN:
            rows = build_euclidean_neighborhood(
                self.grid,
                self.jump_distance,
                kernel=resolved_kernel,
                silent=silent,
                distance_metric=metric,
                occupied_sites=self.num_spins,
            )
            self._neighborhood = (
                FixedDensityNeighborhoodCSR.from_rows(rows)
            )
            self._refresh_energies_after_neighborhood_build()
            return self

        rows = build_neighborhood(
            self.grid,
            self.jump_distance,
            antipodal=self.antipodal,
            kernel=resolved_kernel,
            silent=silent,
            distance_metric=metric,
            jump_weights=self.jump_weights,
        )
        self._neighborhood = (
            rows if self.antipodal else FixedDensityNeighborhoodCSR.from_rows(rows)
        )
        self._refresh_energies_after_neighborhood_build()
        return self

    @property
    def sampling_probabilities(self) -> tuple[float, ...] | None:
        """Experimental angle probabilities, or None for a raw neighborhood."""
        return self._sampling_probabilities

    def compute_anticorrelation_neighborhood(
        self, angles: Sequence[float], *, probabilities: Sequence[float] | None = None,
        kernel: KernelSpec | str | None = None,
    ) -> "System":
        """Build a fixed mixture for independent lawns, including exact theta=0.

        Stored energy is num_spins/2 times the continuum mixture score.
        Nonzero angles use the selected regularization; zero is always exact.
        """
        from .anticorrelation import build_objective, validate_objective
        angles, probabilities, kernel = validate_objective(
            self, angles, probabilities, kernel
        )
        rows, weights = build_objective(self, angles, probabilities, kernel)
        self._sampling_probabilities = probabilities
        self._jump_distance = angles
        self._jump_weights = weights
        self._kernel = kernel
        self._kernel_is_custom = False
        self._distance_metric = "spherical"
        self._distance_metric_is_custom = False
        self._shell_measure = None
        self._shell_measure_is_custom = False
        self._radial_bounds = None
        self._neighborhood = rows
        self._stored_probabilities.clear()
        self._stored_angle_probabilities.clear()
        self._refresh_energies_after_neighborhood_build()
        if self.current_spins is not None and (
            self.best_spins is None or self.current_energy > self.best_energy
        ):
            self.best_spins = self.current_spins.copy()
            self.best_energy = self.current_energy
        return self

    def angle_probabilities(self, spins: NDArray[np.bool_] | None = None) -> np.ndarray:
        """Evaluate each configured angle on the same state, in input order."""
        from .anticorrelation import component_probabilities
        if self.sampling_probabilities is None:
            raise RuntimeError("Build an anticorrelation objective first.")
        state = "current" if spins is None or spins is self.current_spins else (
            "best" if spins is self.best_spins else None
        )
        if self.neighborhood is None and state in self._stored_angle_probabilities:
            return np.array(self._stored_angle_probabilities[state], copy=True)
        return component_probabilities(self, self.current_spins if spins is None else spins)

    def clone(self) -> "System":
        """Copy spins and histories while sharing the grid and neighborhood."""
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__ = self.__dict__.copy()
        clone._stored_probabilities = self._stored_probabilities.copy()
        clone._stored_angle_probabilities = {k: tuple(v) for k, v in self._stored_angle_probabilities.items()}
        state = self.snapshot_state()
        clone._current_spins = state.current_spins
        clone._best_spins = state.best_spins
        clone._occupancy = state.occupancy
        clone._anneal_data = self.anneal_data.copy()
        clone._greedy_history = self.greedy_history.copy()
        clone._parallel_tempering_history = self.parallel_tempering_history.copy()
        return clone

    def __copy__(self) -> "System":
        return self.clone()

    def set_occupancy(self, occupancy: OccupancyState | None) -> None:
        """Attach fixed-density occupancy lists."""
        self._occupancy = None if occupancy is None else occupancy.copy()

    def sync_occupancy_from_spins(self) -> None:
        """Rebuild occupied/unoccupied lists from current_spins."""
        if self.current_spins is None:
            self.set_occupancy(None)
            return
        occupancy = OccupancyState.from_spin_array(self.current_spins)
        if (
            self.kind is SystemKind.EUCLIDEAN
            and occupancy.occupied.size != self.num_spins
        ):
            raise ValueError(
                f"Expected {self.num_spins} occupied sites, got {occupancy.occupied.size}."
            )
        self._occupancy = occupancy

    def snapshot_state(self) -> SystemState:
        """Return a detached copy of mutable spin state."""
        return SystemState(
            current_spins=self.current_spins,
            current_energy=self.current_energy,
            best_spins=self.best_spins,
            best_energy=self.best_energy,
            occupancy=self.occupancy,
        ).copy()

    def restore_state(self, state: SystemState) -> None:
        """Restore mutable spin state from a snapshot."""
        state = state.copy()
        self.current_spins = state.current_spins
        self.current_energy = state.current_energy
        self.best_spins = state.best_spins
        self.best_energy = state.best_energy
        self._occupancy = state.occupancy

    def random_init(self, seed: Seed = None) -> "System":
        """Initialize a random spin state (fixed density keeps ``num_spins`` occupied)."""
        self._require_neighborhood()
        init_random(self, seed=seed)
        self._initialized = True
        return self

    def ball_init(self, radius: float) -> "System":
        """Occupy a Euclidean ball of ``radius`` cells and initialize its state.

        One cell length is ``h = num_spins ** (-1 / dimension)``. The center
        is the site nearest the grid's bounding-box midpoint (first site on
        ties). All sites at distance <= ``radius * h`` are occupied.

        The selected count becomes ``num_spins``. If it changes, the grid is
        rescaled to keep occupied volume 1, the neighborhood is rebuilt with
        its existing physical interaction parameters, and histories are reset.
        Build the neighborhood before calling this method. Radius zero selects
        the center site; a ball covering every site is not a valid swap system.
        """
        if self.kind is not SystemKind.EUCLIDEAN:
            raise ValueError("ball_init() requires a Euclidean grid.")
        self._require_neighborhood()
        radius = float(radius)
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative.")

        midpoint = (self.grid.min(axis=0) + self.grid.max(axis=0)) / 2.0
        center_distances = np.linalg.norm(self.grid - midpoint, axis=1)
        center_index = np.flatnonzero(np.isclose(
            center_distances, center_distances.min(), rtol=1e-12, atol=0,
        ))[0]
        h = _cell_length(self)
        distances = np.linalg.norm(self.grid - self.grid[center_index], axis=1) / h
        spins = (distances <= radius) | np.isclose(distances, radius, rtol=1e-12, atol=0)
        occupied = int(np.count_nonzero(spins))
        if occupied == len(self.grid):
            raise ValueError("Ball covers every site; leave at least one unoccupied site.")
        if occupied == self.num_spins:
            return self.array_init(spins)

        # Build the new density on a copy so a failed rebuild leaves this run intact.
        trial = self.clone()
        trial._grid = np.ascontiguousarray(
            self.grid * (self.num_spins / occupied) ** (1.0 / self.dimension)
        )
        trial._num_spins = occupied
        trial.current_spins = None
        trial.best_spins = None
        trial._anneal_data = AnnealData()
        trial._greedy_history = GreedyHistory()
        trial._parallel_tempering_history = ParallelTemperingHistory()
        trial.compute_neighborhood(
            kernel=self.kernel,
            distance_metric=self.distance_metric,
            shell_measure=self.shell_measure,
        )
        trial.array_init(spins)
        self.__dict__.update(trial.__dict__)
        return self

    def hemisphere_init(self, seed: Seed = None) -> "System":
        """Initialize a spherical system with one randomly oriented hemisphere up."""
        self._require_neighborhood()
        if self.kind is SystemKind.EUCLIDEAN:
            raise ValueError("hemisphere_init() requires a spherical grid.")
        init_random_hemisphere(self, seed=seed)
        self._initialized = True
        return self

    def tri_cog_init(self, num_cogs: int, cog_height: float) -> "System":
        """Initialize a spherical system to a triangle-wave cogwheel boundary."""
        self._require_neighborhood()
        init_triangle_cogwheel(self, num_cogs, cog_height)
        self._initialized = True
        return self

    def cos_cog_init(self, num_cogs: int, cog_height: float) -> "System":
        """Initialize a spherical system to a cosine-wave cogwheel boundary."""
        self._require_neighborhood()
        init_cosine_cogwheel(self, num_cogs, cog_height)
        self._initialized = True
        return self

    def array_init(self, spin_array: NDArray[np.bool_]) -> "System":
        """Initialize from an explicit spin array shaped like ``system_shape``."""
        self._require_neighborhood()
        init_from_array(self, spin_array)
        self._initialized = True
        return self

    # --- Optimization ---

    def anneal(
        self,
        *,
        schedule: AnnealSchedule | dict | None = None,
        starting_temp: float | None = None,
        final_temp: float | None = None,
        cooling_rate: float | None = None,
        min_acceptance: float | None = None,
        max_runtime_hours: float | None = None,
        max_iterations: int | None = None,
        steps: int | None = None,
        save_path: str | Path | None = None,
        save_interval: int = 1,
        verbose: bool = False,
        reinit: bool = False,
        seed: Seed = None,
    ) -> AnnealData:
        """Run simulated annealing and return the updated ``anneal_data``.

        ``schedule`` sets the base schedule; any non-None keyword overrides
        the matching field (``steps`` sets ``starting_steps``).
        """
        self._require_neighborhood()
        self._require_init()
        resolved_starting_temp = (
            self.temperature if starting_temp is None else starting_temp
        )
        run_schedule = self._make_anneal_schedule(
            schedule,
            starting_temp=resolved_starting_temp,
            final_temp=final_temp,
            cooling_rate=cooling_rate,
            min_acceptance=min_acceptance,
            max_runtime_hours=max_runtime_hours,
            max_iterations=max_iterations,
            steps=steps,
        )
        simulated_annealing(
            self,
            schedule=run_schedule,
            save_path=save_path,
            save_interval=save_interval,
            verbose=verbose,
            reinit=reinit,
            seed=seed,
        )
        return self.anneal_data

    def greedy(
        self,
        *,
        max_steps: int | None = None,
        min_delta: float = 0.0,
        track_best: bool = True,
        reinit: bool = False,
        verbose: bool = False,
    ) -> GreedyResult:
        """Apply the best single update while its energy gain exceeds ``min_delta``."""
        self._require_neighborhood()
        self._require_init()
        return greedy_optimization(
            self,
            max_steps=max_steps,
            min_delta=min_delta,
            track_best=track_best,
            reinit=reinit,
            verbose=verbose,
        )

    def parallel_temper(
        self,
        *,
        temperatures: (
            list[float] | tuple[float, ...] | NDArray[np.float64] | None
        ) = None,
        num_replicas: int = 8,
        iterations: int = 100,
        sweep_steps: int = 100_000,
        exchange_interval: int = 1,
        initial_states: (
            list[NDArray[np.bool_]] | tuple[NDArray[np.bool_], ...] | None
        ) = None,
        cold_temp: float = 0.05,
        hot_temp: float | None = None,
        reinit: bool = False,
        track_best: bool = True,
        save_path: str | Path | None = None,
        seed: Seed = None,
        n_workers: int = 1,
        verbose: bool = False,
    ) -> ParallelTemperingResult:
        """Run replica-exchange Monte Carlo and return the run summary.

        Pass ``temperatures`` directly or let ``num_replicas`` geometric
        temperatures span ``cold_temp`` to ``hot_temp``.
        """
        self._require_neighborhood()
        self._require_init()
        return parallel_tempering(
            self,
            temperatures=temperatures,
            num_replicas=num_replicas,
            iterations=iterations,
            sweep_steps=sweep_steps,
            exchange_interval=exchange_interval,
            initial_states=initial_states,
            cold_temp=cold_temp,
            hot_temp=hot_temp,
            reinit=reinit,
            track_best=track_best,
            save_path=save_path,
            seed=seed,
            n_workers=n_workers,
            verbose=verbose,
        )

    def thermalize(
        self,
        target_acceptance: float = 0.6,
        tolerance: float = 0.05,
        steps_per_temp: int | None = None,
        max_adjustments: int = 20,
        k_p: float = 0.5,
        verbose: bool = False,
        seed: Seed = None,
    ) -> float:
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
        seed : int | None, optional
            Seed for this thermalization call. The caller's NumPy RNG state is
            restored afterward.
        """
        if steps_per_temp is None:
            steps_per_temp = self.num_spins * 10

        return thermalize_system(
            system=self,
            target_acceptance=target_acceptance,
            tolerance=tolerance,
            steps_per_temp=steps_per_temp,
            max_adjustments=max_adjustments,
            verbose=verbose,
            k_p=k_p,
            seed=seed,
        )

    # --- Plotting ---

    def plot(
        self,
        plot_current: bool = False,
        auto_rotate: bool = False,
        show_edges: bool = False,
        edge_distance: float | None = None,
        edge_tolerance: float = 1e-6,
        voronoi: bool = False,
        plot_figure: bool = True,
    ):
        """Plot the best (or current) spin state; returns the figure if not shown."""
        system_to_plot = self.current_spins if plot_current else self.best_spins
        if system_to_plot is None and self.kind is not SystemKind.EUCLIDEAN:
            self._require_init()
        voronoi_padding_distance = None
        if voronoi:
            if self.kind is not SystemKind.EUCLIDEAN or self.dimension != 2:
                raise ValueError(
                    "voronoi=True is only available for 2D Euclidean systems."
                )
            if system_to_plot is None:
                raise RuntimeError(
                    "voronoi=True requires an initialized spin state. "
                    "Call random_init() or array_init() first."
                )
            voronoi_padding_distance = _cell_length(self) / 2.0
        return plot_spin_system(
            kind=self.kind,
            grid_points=self.grid,
            spin_states=system_to_plot,
            fig_title="Current System" if plot_current else "Best System",
            rotate=auto_rotate,
            show_edges=show_edges,
            edge_distance=edge_distance,
            edge_tolerance=edge_tolerance,
            voronoi_padding_distance=voronoi_padding_distance,
            plot_figure=plot_figure,
        )

    def plot_neighbors(
        self,
        *,
        selected_site: int,
        plot_figure: bool = True,
    ):
        """Plot one site's neighbors colored by contribution (2D Euclidean only)."""
        if self.kind is not SystemKind.EUCLIDEAN or self.dimension != 2:
            raise ValueError(
                "plot_neighbors() is only available for 2D Euclidean systems."
            )
        if isinstance(selected_site, bool) or not isinstance(
            selected_site, (int, np.integer)
        ):
            raise TypeError("selected_site must be an integer.")
        selected_site = int(selected_site)
        if selected_site < 0 or selected_site >= len(self.grid):
            raise ValueError(
                f"selected_site must be between 0 and {len(self.grid) - 1}, "
                f"got {selected_site}."
            )
        self._require_neighborhood()
        selected_row = self.neighborhood[selected_site]
        if not hasattr(selected_row, "neighbor_spins") or not hasattr(
            selected_row, "contributions"
        ):
            raise TypeError(
                "plot_neighbors(selected_site=...) requires fixed-density "
                "neighborhood rows with neighbor_spins and contributions."
            )
        return plot_spin_system(
            kind=self.kind,
            grid_points=self.grid,
            spin_states=None,
            fig_title="Neighbors",
            selected_site=selected_site,
            selected_neighbor_spins=selected_row.neighbor_spins,
            selected_contributions=selected_row.contributions,
            voronoi_padding_distance=_cell_length(self) / 2.0,
            plot_figure=plot_figure,
        )

    def plot_boundary(self, plot_current: bool = False, plot_figure: bool = True):
        """Plot the up/down boundary in (phi, theta); returns ``(fig, ax)`` if not shown."""
        if self.kind is SystemKind.EUCLIDEAN:
            raise ValueError(
                "Boundary plotting is only implemented for spherical grids."
            )
        system_to_plot = self.current_spins if plot_current else self.best_spins
        independent = self.kind is SystemKind.INDEPENDENT
        return graph_boundary(
            grid=self.grid,
            spin_states=system_to_plot,
            antipodal=self.antipodal,
            independent=independent,
            plot_figure=plot_figure,
        )

    # --- Saving and loading ---

    def to_dataset(self) -> xr.Dataset:
        """
        Return an xarray Dataset containing serializable system state.

        Euclidean metadata includes both `site_count` and `num_spins` because
        candidate sites and occupied sites have different roles. The conserved
        occupied volume is fixed to 1.0, so it is not stored as a user variable.
        """
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        coords: dict[str, np.ndarray] = {}

        spin_dims, spin_coords = self._spin_dims_and_coords()
        coords.update(spin_coords)

        if self.current_spins is not None:
            data_vars["current_spins"] = (spin_dims, self.current_spins)
        if self.best_spins is not None:
            data_vars["best_spins"] = (spin_dims, self.best_spins)

        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Scalar metadata needed to validate and rebuild runtime state.
        attrs = {
            "format_version": self._SAVE_FORMAT_VERSION,
            "kind": self.kind.value,
            "initialized": bool(self.initialized),
            "num_spins": int(self.num_spins),
            "site_count": int(self.grid.shape[0]),
            "dimension": int(self.dimension),
            "grid_shape_json": json.dumps([int(dim) for dim in self.grid.shape]),
            "grid_dtype": str(self.grid.dtype),
            "current_energy": float(self.current_energy),
            "best_energy": float(self.best_energy),
            "temperature": float(self.temperature),
            "anneal_data_json": self._history_to_json(self.anneal_data),
            "greedy_history_json": self._history_to_json(self.greedy_history),
            "parallel_tempering_history_json": self._history_to_json(
                self.parallel_tempering_history
            ),
        }
        if self.jump_distance is not None:
            if isinstance(self.jump_distance, tuple):
                attrs["jump_distances_json"] = json.dumps(
                    [float(distance) for distance in self.jump_distance]
                )
                attrs["jump_weights_json"] = json.dumps(
                    [float(weight) for weight in (self.jump_weights or ())]
                )
            else:
                attrs["jump_distance"] = float(self.jump_distance)
                if self.jump_weights not in (None, (1.0,)):
                    attrs["jump_weights_json"] = json.dumps(
                        [float(weight) for weight in self.jump_weights]
                    )
        if self.sampling_probabilities is not None:
            attrs["objective_type"] = "anticorrelation"
            attrs["objective_normalization"] = "energy=num_spins*score/2"
            attrs["sampling_probabilities_json"] = json.dumps(self.sampling_probabilities)
            for state in ("current", "best"):
                spins = getattr(self, f"{state}_spins")
                if spins is not None:
                    attrs[f"{state}_angle_probabilities_json"] = json.dumps(
                        self.angle_probabilities(spins).tolist()
                    )
        attrs.update(self._kernel_metadata())
        attrs.update(self._normalization_metadata())
        attrs.update(self._distance_metric_metadata())
        attrs.update(self._neighborhood_storage_metadata())
        attrs.update(self._saved_probability_metadata())
        ds.attrs.update(attrs)
        return ds

    def save(self, save_path: str | Path) -> Path:
        """Write the NetCDF state file and return its path."""
        netcdf_path = self._resolve_save_path(save_path)
        self.to_dataset().to_netcdf(netcdf_path)
        return netcdf_path

    def load(
        self,
        load_path: str | Path,
        *,
        rebuild_neighborhood: bool = False,
        silent: bool = True,
    ) -> "System":
        """
        Load state from NetCDF saved by `save()` without touching `self.grid`.
        Returns the system so it can be chained after a constructor.
        Validates that the saved state matches this instance's grid shape and configuration.
        Optionally rebuild the neighborhood from saved built-in metadata.

        The grid itself is supplied by the live `System` instance and must
        match the saved candidate-site metadata.
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"No such file: {load_path}")

        ds = xr.load_dataset(load_path)
        format_version = str(self._require_dataset_attr(ds, "format_version"))
        if format_version not in self._SUPPORTED_SAVE_FORMAT_VERSIONS:
            raise ValueError(
                "Unsupported saved-state format_version "
                f"{format_version!r}. Expected one of "
                f"{sorted(self._SUPPORTED_SAVE_FORMAT_VERSIONS)!r}."
            )

        saved_kind = self._kind_from_dataset_attrs(ds)
        saved_kernel_custom = bool(ds.attrs.get("kernel_custom", False))
        kernel_kind = str(self._require_dataset_attr(ds, "kernel_kind"))
        if saved_kernel_custom:
            saved_kernel = None
        elif kernel_kind == "gaussian":
            cutoff = (
                None
                if bool(ds.attrs.get("gaussian_cutoff_unbounded", False))
                else float(self._require_dataset_attr(ds, "gaussian_cutoff"))
            )
            saved_kernel = KernelSpec.gaussian(
                float(self._require_dataset_attr(ds, "gaussian_sigma")),
                cutoff=cutoff,
            )
        else:
            saved_kernel = KernelSpec.from_value(kernel_kind)
        saved_shell_measure_custom = bool(
            ds.attrs.get("shell_measure_custom", False)
        )
        saved_radial_bounds = self._coerce_radial_bounds(
            self._json_attr(ds, "radial_bounds_json")
        )

        # Check that the checkpoint belongs to this grid and model.
        site_count_saved = int(self._require_dataset_attr(ds, "site_count"))
        if site_count_saved != self.grid.shape[0]:
            raise ValueError(
                f"Saved state expects site_count={site_count_saved}, "
                f"but current grid has {self.grid.shape[0]} sites."
            )
        if "dimension" in ds.attrs and int(ds.attrs["dimension"]) != self.grid.shape[1]:
            raise ValueError(
                f"Saved state expects dimension={int(ds.attrs['dimension'])}, "
                f"but current grid has dimension={self.grid.shape[1]}."
            )
        if "grid_shape_json" in ds.attrs:
            saved_grid_shape = tuple(self._json_attr(ds, "grid_shape_json"))
            if saved_grid_shape != tuple(int(dim) for dim in self.grid.shape):
                raise ValueError(
                    f"Saved state expects grid_shape={saved_grid_shape}, "
                    f"but current grid has shape={tuple(self.grid.shape)}."
                )

        # Enums and core attrs
        if saved_kind is not self.kind:
            raise ValueError(
                "Saved state kind does not match this instance. "
                f"saved={saved_kind.value!r} current={self.kind.value!r}"
            )
        loaded_num_spins = int(self._require_dataset_attr(ds, "num_spins"))
        if loaded_num_spins != self.num_spins:
            raise ValueError(
                f"Saved state expects num_spins={loaded_num_spins}, "
                f"but this system has num_spins={self.num_spins}. "
                "Construct a matching system before loading."
            )
        self._kernel = saved_kernel
        self._kernel_is_custom = saved_kernel_custom
        jump_distance = self._json_attr(
            ds, "jump_distances_json", ds.attrs.get("jump_distance")
        )
        self._jump_distance = (
            None if jump_distance is None else self._coerce_jump_distance(jump_distance)
        )
        self._jump_weights = None
        if self.jump_distance is not None:
            count = len(self.jump_distance) if isinstance(self.jump_distance, tuple) else 1
            self._jump_weights = self._coerce_jump_weights(
                self._json_attr(ds, "jump_weights_json"), count
            )
        self._shell_measure = None
        self._shell_measure_is_custom = saved_shell_measure_custom
        self._radial_bounds = saved_radial_bounds
        saved_metric = ds.attrs.get("distance_metric")
        saved_metric_custom = bool(ds.attrs.get("distance_metric_custom", False))
        self._distance_metric_is_custom = saved_metric_custom
        self._distance_metric = (
            None if saved_metric_custom or saved_metric is None else str(saved_metric)
        )
        self._initialized = bool(self._require_dataset_attr(ds, "initialized"))
        self.current_energy = float(self._require_dataset_attr(ds, "current_energy"))
        self.best_energy = float(self._require_dataset_attr(ds, "best_energy"))
        self._anneal_data = self._history_from_json(
            self._require_dataset_attr(ds, "anneal_data_json"),
            AnnealData,
            attr_name="anneal_data_json",
        )
        self._greedy_history = self._history_from_json(
            self._require_dataset_attr(ds, "greedy_history_json"),
            GreedyHistory,
            attr_name="greedy_history_json",
        )
        self._parallel_tempering_history = self._history_from_json(
            self._require_dataset_attr(ds, "parallel_tempering_history_json"),
            ParallelTemperingHistory,
            attr_name="parallel_tempering_history_json",
        )
        self.temperature = float(self._require_dataset_attr(ds, "temperature"))

        expected_shape = (
            (self.system_shape,) if isinstance(self.system_shape, int) else self.system_shape
        )
        for state in ("current", "best"):
            name = f"{state}_spins"
            # Format 6.0 and earlier named these variables "<state>_system".
            stored_name = name if name in ds else f"{state}_system"
            spins = None
            if stored_name in ds:
                spins = np.ascontiguousarray(ds[stored_name].values, dtype=np.bool_)
                if spins.shape != expected_shape:
                    raise ValueError(
                        f"Saved {name} has shape {spins.shape}, expected {expected_shape}."
                    )
                if (
                    self.kind is SystemKind.EUCLIDEAN
                    and np.count_nonzero(spins) != self.num_spins
                ):
                    raise ValueError(
                        f"Saved Euclidean {name} occupied count does not match "
                        f"num_spins={self.num_spins}."
                    )
            setattr(self, name, spins)
            for suffix in ("probability", "graph_probability"):
                key = f"{state}_{suffix}"
                value = self._load_saved_probability(ds, key, present=spins is not None)
                if value is not None:
                    self._stored_probabilities[key] = value

        if not self.antipodal and self.current_spins is not None:
            self.sync_occupancy_from_spins()
        else:
            self.set_occupancy(None)

        # Neighborhood rows are not stored in the dataset.
        self._neighborhood = None

        # If no spin state is present, mark uninitialized.
        if self.current_spins is None and self.best_spins is None:
            self._initialized = False

        self._sampling_probabilities = None
        self._stored_angle_probabilities.clear()
        objective = ds.attrs.get("objective_type")
        if objective is not None:
            if objective != "anticorrelation" or format_version not in ("6.0", "7.0"):
                raise ValueError("Unsupported saved objective type or format.")
            if ds.attrs.get("objective_normalization") != "energy=num_spins*score/2":
                raise ValueError("Unsupported objective normalization.")
            from .anticorrelation import validate_objective
            self._require_dataset_attr(ds, "sampling_probabilities_json")
            angles, probabilities, _ = validate_objective(
                self, self.jump_distance,
                self._json_attr(ds, "sampling_probabilities_json"), self.kernel
            )
            self._jump_distance = angles
            self._sampling_probabilities = probabilities
            for state in ("current", "best"):
                values = self._json_attr(ds, f"{state}_angle_probabilities_json")
                if values is not None:
                    values = np.asarray(values, dtype=float)
                    if values.shape != (len(angles),) or not np.all(np.isfinite(values)):
                        raise ValueError("Invalid saved angle probabilities.")
                    self._stored_angle_probabilities[state] = tuple(values)
        if rebuild_neighborhood:
            if self.sampling_probabilities is not None:
                self.compute_anticorrelation_neighborhood(
                    self.jump_distance, probabilities=self.sampling_probabilities,
                    kernel=self.kernel,
                )
            else:
                self.compute_neighborhood(silent=silent)
        return self

    # --- Implementation helpers ---

    @classmethod
    def _coerce_kernel(
        cls,
        value: KernelSpec | str | Callable,
    ) -> KernelSpec | Callable:
        if callable(value) and not isinstance(value, KernelSpec):
            return value
        return KernelSpec.from_value(value)

    @staticmethod
    def _kernel_display(kernel: KernelSpec | Callable | None) -> str:
        if isinstance(kernel, KernelSpec):
            if kernel.kind == "gaussian":
                return (
                    f"gaussian(sigma={kernel.sigma!r}, "
                    f"cutoff={kernel.cutoff!r})"
                )
            return kernel.kind
        return repr(kernel)

    def _resolve_kernel_for_compute(
        self,
        kernel: KernelSpec | str | Callable | None,
    ) -> KernelSpec | Callable:
        if kernel is not None:
            return self._coerce_kernel(kernel)
        if self._kernel_is_custom:
            raise ValueError(
                "The saved neighborhood metadata used a custom kernel. "
                "Pass kernel=... to compute_neighborhood() to rebuild it."
            )
        if self.kernel is not None:
            return self._coerce_kernel(self.kernel)
        return KernelSpec.cosine()


    def _resolve_distance_metric_for_compute(
        self,
        distance_metric: DistanceMetric | None,
    ) -> DistanceMetric:
        if distance_metric is not None:
            return distance_metric
        if self._distance_metric is not None:
            return self._distance_metric
        if self._distance_metric_is_custom:
            raise ValueError(
                "The saved neighborhood metadata used a custom distance_metric. "
                "Pass distance_metric=... to compute_neighborhood() to rebuild it."
            )
        return "spherical" if self.kind.spherical else "euclidean"

    @staticmethod
    def _is_jump_distance_sequence(value: object) -> bool:
        if isinstance(value, (str, bytes)):
            return False
        if isinstance(value, np.ndarray):
            return value.ndim > 0
        return isinstance(value, Sequence)

    def _coerce_jump_distance(
        self,
        value: float | Sequence[float] | None,
    ) -> float | tuple[float, ...]:
        if value is None:
            raise ValueError(
                "jump_distance is required. Pass it to compute_neighborhood("
                "jump_distance=...)."
            )
        if self._is_jump_distance_sequence(value):
            distances = np.asarray(value, dtype=np.float64)
            if distances.ndim != 1 or distances.size == 0:
                raise ValueError(
                    "jump_distance must be a finite scalar or non-empty 1D sequence."
                )
            if self.kind is SystemKind.EUCLIDEAN:
                raise ValueError(
                    "Multiple jump distances are currently supported only for "
                    "spherical systems."
                )
            if not np.all(np.isfinite(distances)):
                raise ValueError("jump_distance values must be finite.")
            if np.any(distances < 0):
                raise ValueError("jump_distance values must be nonnegative.")
            if np.any(distances > np.pi):
                raise ValueError("Spherical jump_distance values must lie in [0, pi].")
            return tuple(float(distance) for distance in distances)

        jump_distance = float(value)
        if not np.isfinite(jump_distance):
            raise ValueError(f"jump_distance must be finite, got {value!r}.")
        if jump_distance < 0:
            raise ValueError("jump_distance must be nonnegative.")
        if self.kind.spherical and jump_distance > np.pi:
            raise ValueError("Spherical jump_distance must lie in [0, pi].")
        return jump_distance

    @staticmethod
    def _coerce_jump_weights(
        value: Sequence[float] | None,
        distance_count: int,
    ) -> tuple[float, ...]:
        if value is None:
            return tuple(1.0 for _ in range(distance_count))
        if isinstance(value, (str, bytes)):
            raise TypeError("jump_weights must be a 1D sequence of finite numbers.")
        weights = np.asarray(value, dtype=np.float64)
        if weights.ndim != 1 or weights.size != distance_count:
            raise ValueError(
                "jump_weights must have one finite value per jump distance."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("jump_weights values must be finite.")
        return tuple(float(weight) for weight in weights)

    @staticmethod
    def _coerce_radial_bounds(
        value: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError("radial_bounds must contain exactly two values.")
        lower, upper = float(value[0]), float(value[1])
        if not np.isfinite(lower) or lower < 0:
            raise ValueError("The lower radial bound must be finite and nonnegative.")
        if np.isnan(upper) or upper <= lower:
            raise ValueError("The upper radial bound must be greater than the lower.")
        return lower, upper

    @classmethod
    def _make_anneal_schedule(
        cls,
        schedule: AnnealSchedule | dict | None = None,
        *,
        starting_temp: float | None = None,
        final_temp: float | None = None,
        cooling_rate: float | None = None,
        min_acceptance: float | None = None,
        max_runtime_hours: float | None = None,
        max_iterations: int | None = None,
        steps: int | None = None,
    ) -> AnnealSchedule:
        if schedule is None:
            data = asdict(AnnealSchedule())
        elif isinstance(schedule, AnnealSchedule):
            data = asdict(schedule)
        elif isinstance(schedule, dict):
            data = asdict(AnnealSchedule.from_mapping(schedule))
        else:
            raise TypeError("schedule must be an AnnealSchedule, mapping, or None.")

        updates = {
            "starting_temp": starting_temp,
            "final_temp": final_temp,
            "cooling_rate": cooling_rate,
            "min_acceptance": min_acceptance,
            "max_runtime_hours": max_runtime_hours,
            "max_iterations": max_iterations,
            "starting_steps": steps,
        }
        for key, value in updates.items():
            if value is not None:
                data[key] = value

        return AnnealSchedule.from_mapping(data)

    @staticmethod
    def _resolve_regular_lattice_num_spins(
        n_sites: int,
        *,
        num_spins: int | None,
        density: float | None,
    ) -> int:
        if num_spins is not None and density is not None:
            raise ValueError("Pass either num_spins or density, not both.")
        if density is not None:
            if isinstance(density, bool):
                raise TypeError("density must be a number between 0 and 1.")
            density = float(density)
            if not np.isfinite(density) or not 0 < density < 1:
                raise ValueError("density must be between 0 and 1.")
            num_spins = int(round(n_sites * density))
        elif num_spins is None:
            num_spins = n_sites // 2
        return System._validate_occupied_count(num_spins, n_sites)

    @staticmethod
    def _validate_occupied_count(num_spins: int, n_sites: int) -> int:
        if isinstance(num_spins, bool) or not isinstance(num_spins, (int, np.integer)):
            raise TypeError("num_spins must be an integer.")
        resolved = int(num_spins)
        if not 0 < resolved < n_sites:
            raise ValueError(
                f"num_spins must be between 1 and N-1 for N={n_sites}, got {resolved}."
            )
        return resolved

    def _initialize_grid(
        self,
        grid: NDArray[np.float64],
        kind: SystemKind | str,
        num_spins: int | None,
        prescaled: bool,
    ) -> None:
        """Validate the grid and set its physical scale and particle count.

        ``prescaled`` Euclidean grids already have occupied volume 1 and are
        not rescaled by their bounding box.
        """
        kind = SystemKind(kind)
        if not isinstance(grid, np.ndarray):
            raise TypeError(f"grid must be a numpy.ndarray, got {type(grid)!r}")

        if grid.ndim != 2:
            raise ValueError(
                f"grid must be a 2D array of Cartesian coordinates with shape (N, d), "
                f"but got shape {grid.shape}"
            )
        if kind is SystemKind.EUCLIDEAN:
            if grid.shape[1] < 1:
                raise ValueError(
                    f"Euclidean grid must have at least one coordinate dimension, "
                    f"but got shape {grid.shape}"
                )
        elif grid.shape[1] != 3:
            raise ValueError(
                f"Spherical grid must be 3D Cartesian coordinates with shape (N, 3), "
                f"but got shape {grid.shape}"
            )
        if not np.issubdtype(grid.dtype, np.number):
            raise TypeError("grid must contain numeric values")
        grid = np.ascontiguousarray(grid, dtype=np.float64)

        n_sites = int(grid.shape[0])

        # Enforce even number of sites only for antipodal systems
        if kind.antipodal and n_sites % 2 != 0:
            raise ValueError(
                "Complementary and independent systems require an even number "
                "of lattice sites. "
                "The second half of lattice sites are assumed to be antipodes of the first half."
            )

        if num_spins is not None and kind is not SystemKind.EUCLIDEAN:
            raise ValueError(
                "num_spins can only be set explicitly for Euclidean systems. "
                "Spherical systems derive num_spins from the grid size."
            )
        if kind is SystemKind.EUCLIDEAN:
            resolved_num_spins = self._validate_occupied_count(
                n_sites // 2 if num_spins is None else num_spins, n_sites
            )
            if not prescaled:
                grid = normalize_euclidean_grid(
                    grid, target_volume=n_sites / resolved_num_spins
                )
        else:
            resolved_num_spins = n_sites // 2

        self._grid = grid
        self._kind = kind
        self._num_spins = resolved_num_spins

    def _forget_stored_probabilities(self, state: str) -> None:
        """Drop probabilities loaded from disk once a state's spins or energy change."""
        self._stored_probabilities.pop(f"{state}_probability", None)
        self._stored_probabilities.pop(f"{state}_graph_probability", None)
        self._stored_angle_probabilities.pop(state, None)

    def _refresh_energies_after_neighborhood_build(self) -> None:
        if self.neighborhood is None:
            return
        energy = get_engine(self.kind).energy
        if self.current_spins is not None:
            self.current_energy = energy(self.current_spins, self.neighborhood)
        if self.best_spins is not None:
            self.best_energy = energy(self.best_spins, self.neighborhood)

    def _require_neighborhood(self) -> None:
        if self.neighborhood is None:
            raise RuntimeError(
                "Neighborhood has not been built. Call compute_neighborhood() first."
            )

    def _require_init(self) -> None:
        if not self.initialized:
            raise RuntimeError(
                "System needs to be initialized. Call random_init(), ball_init(), hemisphere_init(), tri_cog_init(), cos_cog_init(), or array_init() first."
            )

    def _spin_dims_and_coords(self) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
        """Canonical dims/coords for storing spin arrays based on configuration."""
        n_sites = int(self.grid.shape[0])
        if self.kind is SystemKind.COMPLEMENTARY:
            dims = ("spin",)
            coords = {"spin": np.arange(self.num_spins)}
        elif self.kind is SystemKind.INDEPENDENT:
            dims = ("spin", "comp")
            coords = {"spin": np.arange(self.num_spins), "comp": np.array([0, 1])}
        else:  # fixed-density spherical or Euclidean
            dims = ("site",)
            coords = {"site": np.arange(n_sites)}
        return dims, coords

    @staticmethod
    def _history_to_json(history) -> str:
        payload = {} if history is None else asdict(history)
        return json.dumps(payload)

    @staticmethod
    def _history_from_json(payload: str, history_type, *, attr_name: str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Saved {attr_name} is not valid JSON.") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Saved {attr_name} must encode an object.")
        return history_type.from_mapping(data)

    @staticmethod
    def _require_dataset_attr(ds: xr.Dataset, key: str):
        if key not in ds.attrs:
            raise ValueError(f"Saved state is missing required attribute {key!r}.")
        return ds.attrs[key]

    @staticmethod
    def _load_saved_probability(
        ds: xr.Dataset, key: str, *, present: bool
    ) -> float | None:
        if key not in ds.attrs:
            return None
        if not present:
            raise ValueError(
                f"Saved state contains {key!r} but has no corresponding spin state."
            )
        value = float(ds.attrs[key])
        if not np.isfinite(value):
            raise ValueError(f"Saved {key!r} must be finite, got {value!r}.")
        return value

    @classmethod
    def _kind_from_dataset_attrs(cls, ds: xr.Dataset) -> SystemKind:
        if "kind" in ds.attrs:
            return SystemKind(str(ds.attrs["kind"]))
        # Format 6.0 and earlier stored four config fields; the constraint and
        # geometry fields alone determine the kind.
        geometry = str(cls._require_dataset_attr(ds, "geometry")).lower()
        constraint = str(cls._require_dataset_attr(ds, "constraint")).lower().replace("_", "-")
        if constraint == "fixed-density":
            return SystemKind.EUCLIDEAN if geometry == "euclidean" else SystemKind.FIXED_DENSITY
        if constraint == "antipodal":
            return SystemKind.COMPLEMENTARY
        if constraint == "independent":
            return SystemKind.INDEPENDENT
        raise ValueError(f"Unsupported saved constraint {constraint!r}.")

    @staticmethod
    def _resolve_save_path(save_path: str | Path) -> Path:
        """Treat suffix-less paths and directories as folders for system_state.nc."""
        save_path = Path(save_path)
        if save_path.suffix == "" or save_path.is_dir():
            save_path = save_path / "system_state.nc"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        return save_path

    def _neighborhood_storage_metadata(self) -> dict[str, int | str | bool]:
        neighborhood = self.neighborhood
        if neighborhood is None:
            return {
                "neighborhood_built": False,
                "neighborhood_storage": "none",
                "neighborhood_rows": 0,
                "neighborhood_edges": 0,
            }
        if isinstance(neighborhood, FixedDensityNeighborhoodCSR):
            return {
                "neighborhood_built": True,
                "neighborhood_storage": "csr",
                "neighborhood_rows": len(neighborhood),
                "neighborhood_edges": int(neighborhood.neighbor_spins.size),
            }
        if isinstance(neighborhood, list):
            edges = 0
            for row in neighborhood:
                if hasattr(row, "neighbor_spins"):
                    edges += int(row.neighbor_spins.size)
                else:
                    edges += int(row.positive_spins.size)
                    edges += int(row.negative_spins.size)
            return {
                "neighborhood_built": True,
                "neighborhood_storage": "rows",
                "neighborhood_rows": len(neighborhood),
                "neighborhood_edges": edges,
            }
        return {
            "neighborhood_built": True,
            "neighborhood_storage": type(neighborhood).__name__,
            "neighborhood_rows": 0,
            "neighborhood_edges": 0,
        }

    def _distance_metric_metadata(self) -> dict[str, str | bool]:
        metric = self.distance_metric
        custom = bool(self._distance_metric_is_custom)
        attrs: dict[str, str | bool] = {"distance_metric_custom": custom}
        if metric is None:
            return attrs
        if custom:
            attrs["distance_metric"] = "custom"
            attrs["distance_metric_repr"] = repr(metric)
        else:
            attrs["distance_metric"] = str(metric)
        return attrs

    def _kernel_metadata(self) -> dict[str, str | bool | float]:
        kernel = self.kernel
        if self._kernel_is_custom or not isinstance(kernel, KernelSpec):
            return {"kernel_custom": True, "kernel_kind": "custom", "kernel_repr": repr(kernel)}
        attrs: dict[str, str | bool | float] = {"kernel_custom": False}
        attrs["kernel_kind"] = kernel.kind
        if kernel.kind == "gaussian":
            attrs["gaussian_sigma"] = float(kernel.sigma)
            attrs["gaussian_cutoff_unbounded"] = kernel.cutoff is None
            if kernel.cutoff is not None:
                attrs["gaussian_cutoff"] = float(kernel.cutoff)
        return attrs

    def _normalization_metadata(self) -> dict[str, str | bool]:
        custom = bool(self._shell_measure_is_custom)
        attrs: dict[str, str | bool] = {"shell_measure_custom": custom}
        if custom:
            attrs["shell_measure_repr"] = repr(self.shell_measure)
        if self.radial_bounds is not None:
            attrs["radial_bounds_json"] = json.dumps(
                [float(bound) for bound in self.radial_bounds]
            )
        return attrs

    def _saved_probability_metadata(self) -> dict[str, float | str]:
        """Save both normalizations, or explain why a value is undefined."""
        attrs: dict[str, float | str] = {}
        for state in ("current", "best"):
            spins = getattr(self, f"{state}_spins")
            if spins is None:
                continue
            for mode, suffix in (("continuum", "probability"), ("graph", "graph_probability")):
                key = f"{state}_{suffix}"
                try:
                    if self.neighborhood is None and key in self._stored_probabilities:
                        value = self._stored_probabilities[key]
                    elif mode == "continuum":
                        if self.neighborhood is None and self._shell_measure_is_custom:
                            raise ValueError("Custom shell measure must be supplied again.")
                        value = (
                            getattr(self, f"{state}_energy")
                            * continuum_probability_prefactor(self, spins)
                        )
                    else:
                        value = self._state_probability(state, mode)
                    if not np.isfinite(value):
                        raise ValueError("Probability is not finite.")
                    attrs[key] = float(value)
                except (RuntimeError, ValueError, ZeroDivisionError) as exc:
                    # Exact/custom kernels or signed weights may not define
                    # both normalizations. Preserve the state and the reason.
                    attrs[f"{key}_unavailable"] = str(exc)
        return attrs

    def _state_probability(self, state: str, normalization: str) -> float:
        mode = str(normalization).lower().replace("_", "-")
        if mode not in ("continuum", "graph"):
            raise ValueError("normalization must be either 'continuum' or 'graph'.")
        spins = getattr(self, f"{state}_spins")
        if spins is None:
            raise RuntimeError(f"System has no {state} spin configuration.")
        suffix = "probability" if mode == "continuum" else "graph_probability"
        key = f"{state}_{suffix}"
        if self.neighborhood is None and key in self._stored_probabilities:
            return self._stored_probabilities[key]
        return calculate_probability(
            self, spins, normalization=mode, energy=getattr(self, f"{state}_energy")
        )

    @staticmethod
    def _json_attr(ds: xr.Dataset, key: str, default=None):
        if key not in ds.attrs:
            return default
        try:
            return json.loads(ds.attrs[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Saved {key} is not valid JSON.") from exc
