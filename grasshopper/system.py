from __future__ import annotations

import json
from collections.abc import Callable
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
    ConstraintModel,
    DeltaType,
    Geometry,
    GreedyHistory,
    InteractionModel,
    NeighborhoodModel,
    NeighborhoodData,
    NonAntipodalNeighborhoodCSR,
    OptimizationRuntime,
    OccupancyState,
    ParallelTemperingHistory,
    SystemConfig,
    SystemDefinition,
    SystemState,
    UpdateRule,
)
from .euclidean import (
    DistanceMetric,
    build_euclidean_neighborhood,
    calculate_prob_prefactor_euclidean,
    normalize_euclidean_grid,
)
from .kernels import calculate_prob_prefactor
from .lattices import (
    hexagonal_lattice,
    rectangular_lattice,
    regular_lattice,
    square_lattice,
    triangular_lattice,
)
from .rng import Seed
from .initialization import (
    init_random,
    init_random_hemisphere,
    init_from_array,
    init_cosine_cogwheel,
    init_triangle_cogwheel,
)
from .greedy import GreedyResult, greedy_optimization
from .parallel_tempering import ParallelTemperingResult, parallel_tempering
from .plot_system import plot_spin_system, graph_boundary


class System:
    """
    Representation of a spin system defined on a spherical or Euclidean lattice.

    This class encapsulates all state, metadata, and operations required to run
    Monte-Carlo-based optimization (thermalization + simulated annealing) for the
    grasshopper spin system. A `System` instance owns:

    - the grid coordinates,
    - system geometry, constraints, interaction model, and update rule,
    - neighborhood parameters once built,
    - computed neighborhood structure,
    - current and best spin configurations,
    - optimization history,
    - save/load metadata for reproducibility.

    Euclidean systems distinguish candidate lattice sites from occupied sites.
    `grid.shape[0]` is the number of candidate sites, while `num_spins` is the
    number of occupied sites conserved by fixed-density updates. The physical
    cell length is derived from occupied volume, not candidate-site count:
    `(1.0 / num_spins) ** (1 / dimension)`. In 2D, the Euclidean length scale is
    `1 / sqrt(num_spins)`.

    The lifecycle of a `System` object is:

    1. **Construction**
       Provide a grid and a `SystemConfig`, or use one of the named
       constructors. The constructor normalizes enums, validates grid shape,
       infers the expected spin layout, computes the number of spins, and
       prepares optimization history containers.

    2. **Neighborhood construction**
       Call `compute_neighborhood(jump_distance=..., delta=...)`. This builds
       the adjacency list used by all Metropolis updates and also sets the
       probability prefactor needed for computing Boltzmann factors.

    3. **Initialization**
       Choose one of:
           - `random_init(seed=...)`
           - `hemisphere_init(seed=...)`
           - `tri_cog_init()`
           - `cos_cog_init()`
           - `array_init(spin_array)`
       These create `current_system`, compute its energy, and record the initial
       best configuration.

    4. **Thermalization (optional, but recommended)**
       `thermalize()` adaptively adjusts the temperature to achieve a target
       Metropolis acceptance rate before annealing begins. The adjustment uses
       a proportional controller in log-temperature space for stability and
       scale-invariance.

    5. **Simulated Annealing**
       Calling `anneal(schedule=...)` executes a call-time annealing schedule,
       updates spin states in place, and records best energies and acceptance
       statistics.

    6. **Saving and Loading**
       `save(path)` writes the spin states and anneal statistics to NetCDF.
       Note that **grid and neighborhood rows are not saved**; reconstructing a
       system after loading requires rebuilding the neighborhood manually.
       Neighborhood construction metadata is saved for built-in distance
       metrics.
       `load(path)` restores spin arrays, energies, and optimization records
       while validating consistency with the current grid.

    Attributes
    ----------
    grid : ndarray of shape (N, d)
        Point coordinates for the active geometry. For Euclidean systems these
        are candidate lattice sites.
    jump_distance : float | None
        Last distance used to construct the current neighborhood. Prefer
        passing this to `compute_neighborhood()`.
    config : SystemConfig
        Canonical orthogonal description of the system geometry,
        constraint model, interaction model, and update rule.
    delta : DeltaType
        Last interaction kernel used for neighborhood construction. Prefer
        passing this to `compute_neighborhood()`.
    num_spins : int | None
        For Euclidean systems, the number of occupied sites to initialize and
        preserve during fixed-density updates, and the count that determines
        physical cell length. If omitted in the low-level constructor, defaults
        to N/2. Spherical systems derive this from the grid size.

    internal attributes (set automatically)
    ---------------------------------------
    antipodal : bool
        Derived from the active constraint model.
    initialized : bool
        Tracks whether spin configuration has been initialized.
    neighborhood : NeighborhoodData
        Precomputed adjacency list for every lattice site.
    prob_prefactor : float
        Multiplicative factor used in Boltzmann probability calculations.
    num_spins : int
        Number of logical spins. For Euclidean systems this is the occupied-site
        count; for antipodal systems it is N/2.
    system_shape : tuple or int
        Expected shape of spin arrays based on `config`.
    current_system : ndarray or None
        Current spin configuration.
    best_system : ndarray or None
        Best configuration encountered so far.
    current_energy : float
        Energy of current_system.
    best_energy : float
        Energy of best_system.
    anneal_data : AnnealData
        Optimization history and acceptance statistics.
    Notes
    -----
    - You must call `compute_neighborhood(jump_distance=...)` before initializing spins.
    - You must initialize spins before calling `anneal()` or `thermalize()`.
    - Saving includes spin configurations and annealing metadata, but does NOT
      include the grid itself; the same grid must exist on the object before `load()`.
    """

    _SAVE_FORMAT_VERSION = "3.0"

    def __init__(
        self,
        grid: NDArray[np.float64],
        config: SystemConfig,
        *,
        num_spins: int | None = None,
    ) -> None:
        """
        Construct a system from grid coordinates and an explicit configuration.

        For Euclidean configs, `num_spins` is the occupied-site count. The
        conserved occupied volume is fixed to 1.0, so the candidate grid is
        scaled to estimated volume `len(grid) / num_spins`.
        """
        self._definition = self._build_definition(
            grid=grid,
            config=self._coerce_config(config),
            num_spins=num_spins,
        )
        self._neighborhood_model = NeighborhoodModel(
            delta=DeltaType.TWO,
            delta_is_custom=False,
            jump_distance=None,
            distance_metric=None,
            distance_metric_is_custom=False,
        )
        self._runtime = OptimizationRuntime()

    @classmethod
    def inverse(cls, grid: NDArray[np.float64]) -> "System":
        """Construct a spherical antipodal inverse system."""
        return cls(grid, config=SystemConfig.inverse())

    @classmethod
    def independent(cls, grid: NDArray[np.float64]) -> "System":
        """Construct a spherical independent-spin system."""
        return cls(grid, config=SystemConfig.independent())

    @classmethod
    def non_antipodal(cls, grid: NDArray[np.float64]) -> "System":
        """Construct a spherical fixed-density non-antipodal system."""
        return cls(grid, config=SystemConfig.non_antipodal())

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
            config=SystemConfig.euclidean(),
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
        grid = square_lattice(
            width,
            height,
            origin=origin,
            center=center,
        )
        return cls.euclidean(
            grid,
            num_spins=cls._resolve_regular_lattice_num_spins(
                len(grid),
                num_spins=num_spins,
                density=density,
            ),
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
        grid = rectangular_lattice(
            width,
            height,
            origin=origin,
            center=center,
        )
        return cls.euclidean(
            grid,
            num_spins=cls._resolve_regular_lattice_num_spins(
                len(grid),
                num_spins=num_spins,
                density=density,
            ),
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
        Construct a 2D fixed-density Euclidean system on a triangular lattice.

        The lattice coordinates are normalized by occupied-site count by
        default, so refining the candidate lattice does not change the physical
        jump-distance scale when `num_spins` is unchanged.
        """
        grid = triangular_lattice(
            width,
            height,
            origin=origin,
            center=center,
        )
        return cls.euclidean(
            grid,
            num_spins=cls._resolve_regular_lattice_num_spins(
                len(grid),
                num_spins=num_spins,
                density=density,
            ),
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
        Construct a 2D fixed-density Euclidean system on a honeycomb lattice.

        Each hexagonal unit cell contributes two candidate sites. The default
        Euclidean normalization uses the occupied-site count from `num_spins` or
        `density`, not the total candidate-site count, to set cell length.
        """
        grid = hexagonal_lattice(
            width,
            height,
            origin=origin,
            center=center,
        )
        return cls.euclidean(
            grid,
            num_spins=cls._resolve_regular_lattice_num_spins(
                len(grid),
                num_spins=num_spins,
                density=density,
            ),
        )

    honeycomb_lattice = hexagonal_lattice

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
        return cls.euclidean(
            grid,
            num_spins=cls._resolve_regular_lattice_num_spins(
                len(grid),
                num_spins=num_spins,
                density=density,
            ),
        )

    def __repr__(self) -> str:
        return (
            "System("
            f"geometry={self.geometry.value!r}, "
            f"constraint={self.constraint.value!r}, "
            f"interaction_model={self.interaction_model.value!r}, "
            f"update_rule={self.update_rule.value!r}, "
            f"grid_shape={tuple(int(dim) for dim in self.grid.shape)!r}, "
            f"jump_distance={self.jump_distance!r}, "
            f"delta={self._delta_display(self.delta)!r}, "
            f"distance_metric={self.distance_metric!r}, "
            f"num_spins={self.num_spins}, "
            f"initialized={self.initialized}"
            ")"
        )

    @property
    def config(self) -> SystemConfig:
        return self._definition.config

    @property
    def grid(self) -> NDArray[np.float64]:
        return self._definition.grid

    @property
    def jump_distance(self) -> float | None:
        return self._neighborhood_model.jump_distance

    @property
    def geometry(self) -> Geometry:
        return self._definition.geometry

    @property
    def constraint(self) -> ConstraintModel:
        return self._definition.constraint

    @property
    def interaction_model(self) -> InteractionModel:
        return self._definition.interaction_model

    @property
    def update_rule(self) -> UpdateRule:
        return self._definition.update_rule

    @property
    def delta(self) -> DeltaType | Callable:
        return self._neighborhood_model.delta

    @property
    def distance_metric(self) -> DistanceMetric | None:
        return self._neighborhood_model.distance_metric

    @property
    def num_spins(self) -> int:
        return self._definition.num_spins

    @property
    def dimension(self) -> int:
        return self._definition.dimension

    @property
    def antipodal(self) -> bool:
        return self._definition.antipodal

    @property
    def initialized(self) -> bool:
        return self._runtime.initialized

    @property
    def neighborhood(self) -> NeighborhoodData | None:
        return self._neighborhood_model.neighborhood

    @property
    def prob_prefactor(self) -> float:
        return self._neighborhood_model.prob_prefactor

    @property
    def system_shape(self) -> int | tuple[int, ...] | None:
        return self._definition.system_shape

    @property
    def current_system(self) -> NDArray[np.bool_] | None:
        return self._runtime.current_system

    @current_system.setter
    def current_system(self, value: NDArray[np.bool_] | None) -> None:
        self._runtime.current_system = value

    @property
    def current_energy(self) -> float:
        return self._runtime.current_energy

    @current_energy.setter
    def current_energy(self, value: float) -> None:
        self._runtime.current_energy = float(value)

    @property
    def best_system(self) -> NDArray[np.bool_] | None:
        return self._runtime.best_system

    @best_system.setter
    def best_system(self, value: NDArray[np.bool_] | None) -> None:
        self._runtime.best_system = value

    @property
    def best_energy(self) -> float:
        return self._runtime.best_energy

    @best_energy.setter
    def best_energy(self, value: float) -> None:
        self._runtime.best_energy = float(value)

    @property
    def anneal_data(self) -> AnnealData:
        return self._runtime.anneal_data

    @property
    def greedy_history(self) -> GreedyHistory:
        return self._runtime.greedy_history

    @property
    def parallel_tempering_history(self) -> ParallelTemperingHistory:
        return self._runtime.parallel_tempering_history

    @property
    def occupancy(self) -> OccupancyState | None:
        return self._runtime.occupancy

    @classmethod
    def _coerce_geometry(cls, value: Geometry | str) -> Geometry:
        if isinstance(value, Geometry):
            return value
        if isinstance(value, str):
            try:
                return Geometry(value.lower().replace("_", "-"))
            except ValueError as exc:
                valid = ", ".join(member.value for member in Geometry)
                raise ValueError(
                    f"Unknown geometry {value!r}. Expected one of: {valid}."
                ) from exc
        raise TypeError("geometry must be a Geometry or string.")

    @classmethod
    def _coerce_constraint(cls, value: ConstraintModel | str) -> ConstraintModel:
        if isinstance(value, ConstraintModel):
            return value
        if isinstance(value, str):
            normalized = value.lower().replace("_", "-")
            aliases = {
                "antipodal": ConstraintModel.ANTIPODAL,
                "independent": ConstraintModel.INDEPENDENT,
                "fixed-density": ConstraintModel.FIXED_DENSITY,
                "fixed_density": ConstraintModel.FIXED_DENSITY,
            }
            try:
                return aliases[normalized]
            except KeyError as exc:
                valid = ", ".join(member.value for member in ConstraintModel)
                raise ValueError(
                    f"Unknown constraint {value!r}. Expected one of: {valid}."
                ) from exc
        raise TypeError("constraint must be a ConstraintModel or string.")

    @classmethod
    def _coerce_interaction_model(
        cls,
        value: InteractionModel | str,
    ) -> InteractionModel:
        if isinstance(value, InteractionModel):
            return value
        if isinstance(value, str):
            normalized = value.lower().replace("_", "-")
            aliases = {
                "inverse": InteractionModel.INVERSE,
                "independent": InteractionModel.INDEPENDENT,
                "occupancy": InteractionModel.OCCUPANCY,
                "pairwise-occupancy": InteractionModel.OCCUPANCY,
                "pairwise_occupancy": InteractionModel.OCCUPANCY,
            }
            try:
                return aliases[normalized]
            except KeyError as exc:
                valid = ", ".join(member.value for member in InteractionModel)
                raise ValueError(
                    "Unknown interaction_model "
                    f"{value!r}. Expected one of: {valid}."
                ) from exc
        raise TypeError("interaction_model must be an InteractionModel or string.")

    @classmethod
    def _coerce_update_rule(cls, value: UpdateRule | str) -> UpdateRule:
        if isinstance(value, UpdateRule):
            return value
        if isinstance(value, str):
            normalized = value.lower().replace("_", "-")
            aliases = {
                "flip": UpdateRule.FLIP,
                "swap": UpdateRule.SWAP,
            }
            try:
                return aliases[normalized]
            except KeyError as exc:
                valid = ", ".join(member.value for member in UpdateRule)
                raise ValueError(
                    f"Unknown update_rule {value!r}. Expected one of: {valid}."
                ) from exc
        raise TypeError("update_rule must be an UpdateRule or string.")

    @classmethod
    def _coerce_config(
        cls,
        value: SystemConfig,
    ) -> SystemConfig:
        if isinstance(value, SystemConfig):
            return value
        raise TypeError("config must be a SystemConfig.")

    @staticmethod
    def _derive_system_shape(
        *,
        config: SystemConfig,
        n_sites: int,
        num_spins: int,
    ) -> int | tuple[int, ...]:
        if config.constraint == ConstraintModel.ANTIPODAL:
            return num_spins
        if config.constraint == ConstraintModel.INDEPENDENT:
            return (num_spins, 2)
        return n_sites

    @classmethod
    def _coerce_delta(cls, value: DeltaType | str | Callable) -> DeltaType | Callable:
        if callable(value):
            return value
        if isinstance(value, DeltaType):
            return value
        if isinstance(value, str):
            delta_key = value.lower().replace("-", "_")
            for delta in DeltaType:
                if delta_key == delta.value.lower().replace("-", "_"):
                    return delta
            valid = ", ".join(delta.value for delta in DeltaType)
            raise ValueError(f"Unknown delta {value!r}. Expected one of: {valid}.")
        raise TypeError("delta must be a DeltaType, string, or callable.")

    @staticmethod
    def _delta_argument(delta: DeltaType | Callable) -> str | Callable:
        return delta.value if isinstance(delta, DeltaType) else delta

    @staticmethod
    def _delta_display(delta: DeltaType | Callable) -> str:
        return delta.value if isinstance(delta, DeltaType) else repr(delta)

    def _resolve_delta_for_compute(
        self,
        delta: DeltaType | str | Callable | None,
    ) -> DeltaType | Callable:
        if delta is not None:
            return self._coerce_delta(delta)
        if self._neighborhood_model.delta_is_custom:
            raise ValueError(
                "The saved neighborhood metadata used a custom delta kernel. "
                "Pass delta=... to compute_neighborhood() to rebuild it."
            )
        if self._neighborhood_model.delta is not None:
            return self._coerce_delta(self.delta)
        return DeltaType.TWO

    @staticmethod
    def _is_custom_delta(delta: DeltaType | Callable) -> bool:
        return not isinstance(delta, DeltaType)

    @staticmethod
    def _default_distance_metric(geometry: Geometry) -> str:
        return "euclidean" if geometry == Geometry.EUCLIDEAN else "spherical"

    def _resolve_distance_metric_for_compute(
        self,
        distance_metric: DistanceMetric | None,
    ) -> DistanceMetric:
        if distance_metric is not None:
            return distance_metric
        if self._neighborhood_model.distance_metric is not None:
            return self._neighborhood_model.distance_metric
        if self._neighborhood_model.distance_metric_is_custom:
            raise ValueError(
                "The saved neighborhood metadata used a custom distance_metric. "
                "Pass distance_metric=... to compute_neighborhood() to rebuild it."
            )
        return self._default_distance_metric(self.geometry)

    @staticmethod
    def _is_custom_distance_metric(distance_metric: DistanceMetric) -> bool:
        return not isinstance(distance_metric, str)

    @staticmethod
    def _coerce_jump_distance(value: float | None) -> float:
        if value is None:
            raise ValueError(
                "jump_distance is required. Pass it to compute_neighborhood("
                "jump_distance=...)."
            )
        jump_distance = float(value)
        if not np.isfinite(jump_distance):
            raise ValueError(
                f"jump_distance must be finite, got {value!r}."
            )
        return jump_distance

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

        if isinstance(num_spins, bool) or not isinstance(num_spins, (int, np.integer)):
            raise TypeError("num_spins must be an integer.")
        resolved = int(num_spins)
        if not 0 < resolved < n_sites:
            raise ValueError(
                f"num_spins must be between 1 and N-1 for N={n_sites}, got {resolved}."
            )
        return resolved

    @classmethod
    def _build_definition(
        cls,
        *,
        grid: NDArray[np.float64],
        config: SystemConfig,
        num_spins: int | None,
    ) -> SystemDefinition:
        """
        Validate constructor inputs and derive static system metadata.

        For Euclidean systems, normalization targets candidate-grid volume
        `n_sites / num_spins`. This preserves total occupied volume 1.0 while
        allowing the candidate lattice to have more sites than are occupied.
        """
        if not isinstance(grid, np.ndarray):
            raise TypeError(f"grid must be a numpy.ndarray, got {type(grid)!r}")

        if grid.ndim != 2:
            raise ValueError(
                f"grid must be a 2D array of Cartesian coordinates with shape (N, d), "
                f"but got shape {grid.shape}"
            )
        if config.geometry == Geometry.EUCLIDEAN:
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
        dimension = int(grid.shape[1])
        requested_num_spins = num_spins

        # Enforce even number of sites only for antipodal systems
        if config.antipodal:
            if n_sites % 2 != 0:
                raise ValueError(
                    "INVERSE and INDEPENDENT systems require an even number of lattice sites. "
                    "The second half of lattice sites are assumed to be antipodes of the first half."
                )

        if requested_num_spins is not None:
            if config.geometry != Geometry.EUCLIDEAN:
                raise ValueError(
                    "num_spins can only be set explicitly for Euclidean systems. "
                    "Spherical systems derive num_spins from the grid size."
                )
            if isinstance(requested_num_spins, bool) or not isinstance(
                requested_num_spins, (int, np.integer)
            ):
                raise TypeError("num_spins must be an integer.")
            resolved_num_spins = int(requested_num_spins)
        else:
            resolved_num_spins = n_sites // 2

        if (
            config.geometry == Geometry.EUCLIDEAN
            and not 0 < resolved_num_spins < n_sites
        ):
            raise ValueError(
                f"Euclidean num_spins must be between 1 and N-1 "
                f"for N={n_sites} grid sites, got {resolved_num_spins}."
            )

        if config.geometry == Geometry.EUCLIDEAN:
            target_volume = n_sites / resolved_num_spins
            grid = normalize_euclidean_grid(grid, target_volume=target_volume)

        return SystemDefinition(
            grid=grid,
            config=config,
            num_spins=resolved_num_spins,
            dimension=dimension,
            system_shape=cls._derive_system_shape(
                config=config,
                n_sites=n_sites,
                num_spins=resolved_num_spins,
            ),
        )

    def compute_neighborhood(
        self,
        jump_distance: float | None = None,
        *,
        delta: DeltaType | str | Callable | None = None,
        silent: bool = True,
        distance_metric: DistanceMetric | None = None,
    ) -> None:
        """
        Build and store the neighborhood and probability prefactor.

        For Euclidean systems, kernel cell length is based on `self.num_spins`
        (occupied sites), not `len(self.grid)`. Thus refining the candidate grid
        leaves the physical jump-distance scale unchanged as long as the
        occupied-site count is unchanged.
        """
        self._neighborhood_model.jump_distance = self._coerce_jump_distance(
            self.jump_distance if jump_distance is None else jump_distance
        )
        self._neighborhood_model.delta = self._resolve_delta_for_compute(delta)
        self._neighborhood_model.delta_is_custom = self._is_custom_delta(self.delta)
        metric = self._resolve_distance_metric_for_compute(distance_metric)
        self._neighborhood_model.distance_metric = metric
        self._neighborhood_model.distance_metric_is_custom = (
            self._is_custom_distance_metric(metric)
        )

        if self.geometry == Geometry.EUCLIDEAN:
            self._neighborhood_model.prob_prefactor = calculate_prob_prefactor_euclidean(
                self.num_spins,
                self.jump_distance,
                self.dimension,
            )
        else:
            self._neighborhood_model.prob_prefactor = calculate_prob_prefactor(
                len(self.grid), self.jump_distance
            )

        if self.geometry == Geometry.EUCLIDEAN:
            rows = build_euclidean_neighborhood(
                self.grid,
                self.jump_distance,
                delta=self._delta_argument(self.delta),
                silent=silent,
                distance_metric=metric,
                occupied_sites=self.num_spins,
            )
            self._neighborhood_model.neighborhood = NonAntipodalNeighborhoodCSR.from_rows(rows)
            return

        rows = build_neighborhood(
            self.grid,
            self.jump_distance,
            antipodal=self.antipodal,
            delta=self._delta_argument(self.delta),
            silent=silent,
            distance_metric=metric,
        )
        self._neighborhood_model.neighborhood = (
            rows if self.antipodal else NonAntipodalNeighborhoodCSR.from_rows(rows)
        )

    def _require_neighborhood(self) -> None:
        if self.neighborhood is None:
            raise RuntimeError(
                "Neighborhood has not been built. Call compute_neighborhood() first."
            )

    def _require_init(self) -> None:
        if not self.initialized:
            raise RuntimeError(
                "System needs to be initialized. Call random_init(), hemisphere_init(), tri_cog_init(), cos_cog_init(), or array_init() first."
            )

    @property
    def temperature(self) -> float:
        """Current runtime annealing temperature."""
        return float(self._runtime.temperature)

    @temperature.setter
    def temperature(self, value: float) -> None:
        temperature = float(value)
        if not np.isfinite(temperature):
            raise ValueError(f"temperature must be finite, got {value!r}.")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {value!r}.")
        self._runtime.temperature = temperature

    def clone(self) -> "System":
        """Copy mutable runtime state while sharing validated static structures."""
        clone = self.__class__.__new__(self.__class__)
        clone._definition = SystemDefinition(
            grid=self.grid,
            config=self.config,
            num_spins=int(self.num_spins),
            dimension=int(self.dimension),
            system_shape=self.system_shape,
        )
        clone._neighborhood_model = NeighborhoodModel(
            delta=self.delta,
            delta_is_custom=self._neighborhood_model.delta_is_custom,
            jump_distance=self.jump_distance,
            distance_metric=self.distance_metric,
            distance_metric_is_custom=self._neighborhood_model.distance_metric_is_custom,
            neighborhood=self.neighborhood,
            prob_prefactor=float(self.prob_prefactor),
        )
        clone._runtime = self._runtime.copy()
        return clone

    def __copy__(self) -> "System":
        return self.clone()

    def set_occupancy(self, occupancy: OccupancyState | None) -> None:
        """Attach fixed-density occupancy lists."""
        self._runtime.occupancy = None if occupancy is None else occupancy.copy()

    def sync_occupancy_from_spins(self) -> None:
        """Rebuild occupied/unoccupied lists from current_system."""
        if self.current_system is None:
            self.set_occupancy(None)
            return
        occupancy = OccupancyState.from_spin_array(self.current_system)
        expected_occupied = (
            int(self.num_spins) if self.geometry == Geometry.EUCLIDEAN else None
        )
        occupancy.validate(
            n_sites=int(self.current_system.shape[0]),
            expected_occupied=expected_occupied,
            spin_array=self.current_system,
        )
        self.set_occupancy(occupancy)

    def snapshot_state(self) -> SystemState:
        """Return a detached copy of mutable spin state."""
        return self._runtime.snapshot_state()

    def restore_state(self, state: SystemState) -> None:
        """Restore mutable spin state from a snapshot."""
        self._runtime.restore_state(state)

    # --- initializers ---
    def random_init(self, seed: Seed = None) -> None:
        self._require_neighborhood()
        init_random(self, seed=seed)
        self._runtime.initialized = True

    def hemisphere_init(self, seed: Seed = None) -> None:
        self._require_neighborhood()
        if self.geometry == Geometry.EUCLIDEAN:
            raise ValueError("hemisphere_init() requires a spherical grid.")
        init_random_hemisphere(self, seed=seed)
        self._runtime.initialized = True

    def tri_cog_init(self, num_cogs: int, cog_height: float) -> None:
        self._require_neighborhood()
        init_triangle_cogwheel(self, num_cogs, cog_height)
        self._runtime.initialized = True

    def cos_cog_init(self, num_cogs: int, cog_height: float) -> None:
        self._require_neighborhood()
        init_cosine_cogwheel(self, num_cogs, cog_height)
        self._runtime.initialized = True

    def array_init(self, spin_array: NDArray[np.bool_]) -> None:
        self._require_neighborhood()
        init_from_array(self, spin_array)
        self._runtime.initialized = True

    # --- Annealing ---
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
    ) -> None:
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

    def greedy(
        self,
        *,
        max_steps: int | None = None,
        min_delta: float = 0.0,
        track_best: bool = True,
        reinit: bool = False,
        verbose: bool = False,
    ) -> GreedyResult:
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

    # --- visualization ---
    def plot(
        self,
        plot_current: bool = False,
        auto_rotate: bool = False,
        show_edges: bool = False,
        edge_distance: float | None = None,
        edge_tolerance: float = 1e-6,
        plot_figure: bool = True,
    ):
        system_to_plot = self.current_system if plot_current else self.best_system
        if system_to_plot is None and self.geometry != Geometry.EUCLIDEAN:
            self._require_init()
        return plot_spin_system(
            system_config=self.config,
            grid_points=self.grid,
            spin_states=system_to_plot,
            fig_title="Current System" if plot_current else "Best System",
            rotate=auto_rotate,
            show_edges=show_edges,
            edge_distance=edge_distance,
            edge_tolerance=edge_tolerance,
            plot_figure=plot_figure,
        )

    def plot_lattice(
        self,
        *,
        show_edges: bool = True,
        edge_distance: float | None = None,
        edge_tolerance: float = 1e-6,
        plot_figure: bool = True,
    ):
        if self.geometry != Geometry.EUCLIDEAN or self.dimension != 2:
            raise ValueError("plot_lattice() is only available for 2D Euclidean systems.")
        return plot_spin_system(
            system_config=self.config,
            grid_points=self.grid,
            spin_states=None,
            fig_title="Lattice",
            show_edges=show_edges,
            edge_distance=edge_distance,
            edge_tolerance=edge_tolerance,
            plot_figure=plot_figure,
        )

    def plot_boundary(
        self, plot_current: bool = False, plot_figure: bool = True
    ) -> None:
        if self.geometry == Geometry.EUCLIDEAN:
            raise ValueError(
                "Boundary plotting is only implemented for spherical grids."
            )
        system_to_plot = self.current_system if plot_current else self.best_system
        independent = self.constraint == ConstraintModel.INDEPENDENT
        graph_boundary(
            grid=self.grid,
            spin_states=system_to_plot,
            antipodal=self.antipodal,
            independent=independent,
            plot_figure=plot_figure,
        )

    # --- Save/Load helpers  ---
    def _spin_dims_and_coords(self) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
        """Canonical dims/coords for storing spin arrays based on configuration."""
        n_sites = int(self.grid.shape[0])
        if self.constraint == ConstraintModel.ANTIPODAL:
            dims = ("spin",)
            coords = {"spin": np.arange(self.num_spins)}
        elif self.constraint == ConstraintModel.INDEPENDENT:
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

    @classmethod
    def _config_from_dataset_attrs(cls, ds: xr.Dataset) -> SystemConfig:
        return SystemConfig(
            geometry=cls._coerce_geometry(cls._require_dataset_attr(ds, "geometry")),
            constraint=cls._coerce_constraint(cls._require_dataset_attr(ds, "constraint")),
            interaction_model=cls._coerce_interaction_model(
                cls._require_dataset_attr(ds, "interaction_model")
            ),
            update_rule=cls._coerce_update_rule(
                cls._require_dataset_attr(ds, "update_rule")
            ),
        )

    @staticmethod
    def _resolve_save_paths(save_path: str | Path) -> tuple[Path, Path]:
        save_path = Path(save_path)
        if save_path.suffix == "" or save_path.is_dir():
            save_dir = save_path
            save_dir.mkdir(parents=True, exist_ok=True)
            netcdf_path = save_dir / "system_state.nc"
        else:
            save_dir = save_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            netcdf_path = save_path
        return save_dir, netcdf_path

    def _neighborhood_storage_metadata(self) -> dict[str, int | str | bool]:
        neighborhood = self.neighborhood
        if neighborhood is None:
            return {
                "neighborhood_built": False,
                "neighborhood_storage": "none",
                "neighborhood_rows": 0,
                "neighborhood_edges": 0,
            }
        if isinstance(neighborhood, NonAntipodalNeighborhoodCSR):
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
        custom = bool(self._neighborhood_model.distance_metric_is_custom)
        attrs: dict[str, str | bool] = {"distance_metric_custom": custom}
        if metric is None:
            return attrs
        if custom:
            attrs["distance_metric"] = "custom"
            attrs["distance_metric_repr"] = repr(metric)
        else:
            attrs["distance_metric"] = str(metric)
        return attrs

    def _delta_metadata(self) -> dict[str, str | bool]:
        delta = self.delta
        custom = bool(self._neighborhood_model.delta_is_custom)
        attrs: dict[str, str | bool] = {"delta_custom": custom}
        if custom:
            attrs["delta"] = "custom"
            attrs["delta_repr"] = repr(delta)
        else:
            attrs["delta"] = self._delta_argument(delta)
        return attrs

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

        if self.current_system is not None:
            data_vars["current_system"] = (spin_dims, self.current_system)
        if self.best_system is not None:
            data_vars["best_system"] = (spin_dims, self.best_system)

        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Scalar metadata needed to validate and rebuild runtime state.
        attrs = {
            "format_version": self._SAVE_FORMAT_VERSION,
            "geometry": self.geometry.value,
            "constraint": self.constraint.value,
            "interaction_model": self.interaction_model.value,
            "update_rule": self.update_rule.value,
            "initialized": bool(self.initialized),
            "num_spins": int(self.num_spins),
            "site_count": int(self.grid.shape[0]),
            "dimension": int(self.dimension),
            "grid_shape_json": json.dumps([int(dim) for dim in self.grid.shape]),
            "grid_dtype": str(self.grid.dtype),
            "prob_prefactor": float(self.prob_prefactor),
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
            attrs["jump_distance"] = float(self.jump_distance)
        attrs.update(self._delta_metadata())
        attrs.update(self._distance_metric_metadata())
        attrs.update(self._neighborhood_storage_metadata())
        ds.attrs.update(attrs)
        return ds

    def save(self, save_path: str | Path) -> Path:
        """Write the NetCDF state file and return its path."""
        _, netcdf_path = self._resolve_save_paths(save_path)
        self.to_dataset().to_netcdf(netcdf_path)
        return netcdf_path

    def load(self, load_path: str | Path) -> None:
        """
        Load state from NetCDF saved by `save()` without touching `self.grid`.
        Validates that the saved state matches this instance's grid shape and configuration.
        Rebuild neighborhood yourself after loading if needed.

        For Euclidean systems, the saved occupied-site count (`num_spins`) is
        restored before the probability prefactor is recomputed. The grid itself
        is still supplied by the live `System` instance and must match the saved
        candidate-site metadata.
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"No such file: {load_path}")

        ds = xr.load_dataset(load_path)
        format_version = str(self._require_dataset_attr(ds, "format_version"))
        if format_version != self._SAVE_FORMAT_VERSION:
            raise ValueError(
                "Unsupported saved-state format_version "
                f"{format_version!r}. Expected {self._SAVE_FORMAT_VERSION!r}."
        )

        saved_config = self._config_from_dataset_attrs(ds)
        saved_delta_raw = self._require_dataset_attr(ds, "delta")
        saved_delta_custom = bool(ds.attrs.get("delta_custom", False))
        saved_delta = (
            DeltaType.TWO
            if saved_delta_custom
            else self._coerce_delta(saved_delta_raw)
        )

        # 1) Validate against current instance (grid must already exist)
        if not isinstance(self.grid, np.ndarray) or self.grid.ndim != 2:
            raise RuntimeError(
                "`self.grid` must be present on the instance before calling load()."
            )
        if saved_config.geometry == Geometry.EUCLIDEAN:
            if self.grid.shape[1] < 1:
                raise RuntimeError(
                    "`self.grid` must contain at least one coordinate dimension."
                )
        elif self.grid.shape[1] != 3:
            raise RuntimeError(
                "`self.grid` must have shape (N, 3) for spherical systems."
            )

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
            try:
                saved_grid_shape = tuple(
                    int(dim) for dim in json.loads(str(ds.attrs["grid_shape_json"]))
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("Saved grid_shape_json is not valid JSON.") from exc
            if saved_grid_shape != tuple(int(dim) for dim in self.grid.shape):
                raise ValueError(
                    f"Saved state expects grid_shape={saved_grid_shape}, "
                    f"but current grid has shape={tuple(self.grid.shape)}."
                )

        # Enums and core attrs
        if saved_config != self.config:
            raise ValueError(
                "Saved state config does not match this instance. "
                f"saved={saved_config!r} current={self.config!r}"
            )
        self._definition.config = saved_config
        self._definition.system_shape = self._derive_system_shape(
            config=saved_config,
            n_sites=int(self.grid.shape[0]),
            num_spins=int(self.num_spins),
        )
        self._neighborhood_model.delta = saved_delta
        self._neighborhood_model.delta_is_custom = saved_delta_custom
        self._neighborhood_model.jump_distance = (
            self._coerce_jump_distance(ds.attrs["jump_distance"])
            if "jump_distance" in ds.attrs
            else None
        )
        saved_metric = ds.attrs.get("distance_metric")
        saved_metric_custom = bool(ds.attrs.get("distance_metric_custom", False))
        if saved_metric_custom:
            self._neighborhood_model.distance_metric = None
            self._neighborhood_model.distance_metric_is_custom = True
        elif saved_metric is not None:
            self._neighborhood_model.distance_metric = str(saved_metric)
            self._neighborhood_model.distance_metric_is_custom = False
        else:
            self._neighborhood_model.distance_metric = None
            self._neighborhood_model.distance_metric_is_custom = False
        self._runtime.initialized = bool(self._require_dataset_attr(ds, "initialized"))
        loaded_num_spins = int(self._require_dataset_attr(ds, "num_spins"))
        if saved_config.geometry == Geometry.EUCLIDEAN and not (
            0 < loaded_num_spins < self.grid.shape[0]
        ):
            raise ValueError(
                f"Saved Euclidean num_spins must be between 1 and N-1 "
                f"for N={self.grid.shape[0]} grid sites, got {loaded_num_spins}."
            )
        self._definition.num_spins = loaded_num_spins
        self._definition.system_shape = self._derive_system_shape(
            config=self.config,
            n_sites=int(self.grid.shape[0]),
            num_spins=int(self._definition.num_spins),
        )
        self.current_energy = float(self._require_dataset_attr(ds, "current_energy"))
        self.best_energy = float(self._require_dataset_attr(ds, "best_energy"))
        self._runtime.anneal_data = self._history_from_json(
            self._require_dataset_attr(ds, "anneal_data_json"),
            AnnealData,
            attr_name="anneal_data_json",
        )
        self._runtime.greedy_history = self._history_from_json(
            self._require_dataset_attr(ds, "greedy_history_json"),
            GreedyHistory,
            attr_name="greedy_history_json",
        )
        self._runtime.parallel_tempering_history = self._history_from_json(
            self._require_dataset_attr(ds, "parallel_tempering_history_json"),
            ParallelTemperingHistory,
            attr_name="parallel_tempering_history_json",
        )
        self.temperature = float(self._require_dataset_attr(ds, "temperature"))

        # 2) Arrays
        self.current_system = None
        if "current_system" in ds:
            self.current_system = np.ascontiguousarray(
                ds["current_system"].values,
                dtype=np.bool_,
            )
            if (
                self.geometry == Geometry.EUCLIDEAN
                and int(np.count_nonzero(self.current_system)) != self.num_spins
            ):
                raise ValueError(
                    "Saved Euclidean current_system occupied count does not match "
                    f"num_spins={self.num_spins}."
                )

        self.best_system = None
        if "best_system" in ds:
            self.best_system = np.ascontiguousarray(
                ds["best_system"].values,
                dtype=np.bool_,
            )
            if (
                self.geometry == Geometry.EUCLIDEAN
                and int(np.count_nonzero(self.best_system)) != self.num_spins
            ):
                raise ValueError(
                    "Saved Euclidean best_system occupied count does not match "
                    f"num_spins={self.num_spins}."
                )

        if not self.antipodal and self.current_system is not None:
            self.sync_occupancy_from_spins()
        else:
            self.set_occupancy(None)

        # 3) Recompute prob_prefactor from live grid + jump_distance (grid wasn't saved)
        self._definition.dimension = int(self.grid.shape[1])
        if self.jump_distance is None:
            self._neighborhood_model.prob_prefactor = 0.0
        elif self.geometry == Geometry.EUCLIDEAN:
            self._neighborhood_model.prob_prefactor = calculate_prob_prefactor_euclidean(
                self.num_spins,
                self.jump_distance,
                self.dimension,
            )
        else:
            self._neighborhood_model.prob_prefactor = calculate_prob_prefactor(
                len(self.grid), self.jump_distance
            )

        # 4) Safety: neighborhood must be rebuilt
        self._neighborhood_model.neighborhood = None

        # 5) If no spin state present, mark uninitialized
        if self.current_system is None and self.best_system is None:
            self._runtime.initialized = False
