from __future__ import annotations

import json
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
    config_from_system_type,
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
    SystemType,
    system_type_from_config,
    UpdateRule,
)
from .euclidean import (
    DistanceMetric,
    build_euclidean_neighborhood,
    calculate_prob_prefactor_euclidean,
)
from .kernels import calculate_prob_prefactor
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
    spherical grasshopper system. A `System` instance owns:

    • the spherical grid (Nx3 Cartesian coordinates),
    • system type (INVERSE, INDEPENDENT, NON_ANTIPODAL),
    • neighborhood parameters (jump distance and interaction kernel) once built,
    • computed neighborhood structure,
    • current and best spin configurations,
    • optimization history (energy and acceptance records),
    • save/load routines for reproducibility.

    The lifecycle of a `System` object is:

    1. **Construction**
       Provide a grid and either an orthogonal config or legacy system type.
       The constructor normalizes enums, validates grid shape, infers the
       expected spin layout, computes the number of spins, and prepares an
       `AnnealData` history container.

    2. **Neighborhood construction**
       Call `compute_neighborhood(jump_distance=..., delta=...)`. This builds
       the adjacency list used by all Metropolis updates and also sets the
       probability prefactor needed for computing Boltzmann factors.

    3. **Initialization**
       Choose one of:
           • `random_init()`
           • `hemisphere_init()`
           • `tri_cog_init()`
           • `cos_cog_init()`
           • `array_init(spin_array)`
       These create `current_system`, compute its energy, and record the initial
       “best” configuration.

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
       Note that **grid and neighborhood are not saved**; reconstructing a system
       after loading requires rebuilding the neighborhood manually.
       `load(path)` restores spin arrays, energies, and optimization records
       while validating consistency with the current grid.

    Attributes
    ----------
    grid : ndarray of shape (N, d)
        Point coordinates for the active geometry.
    jump_distance : float | None
        Last distance used to construct the current neighborhood. Prefer
        passing this to `compute_neighborhood()`.
    config : SystemConfig
        Canonical orthogonal description of the system geometry,
        constraint model, interaction model, and update rule.
    system_type : SystemType
        Legacy compatibility view derived from `config`.
    delta : DeltaType
        Last interaction kernel used for neighborhood construction. Prefer
        passing this to `compute_neighborhood()`.
    num_spins : int | None
        For Euclidean systems, the number of occupied sites to initialize and
        preserve during fixed-density updates. If omitted, defaults to N/2.
        Spherical systems derive this from the grid size.

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
        Number of logical spins (N/2 for antipodal systems).
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
    • You must call `compute_neighborhood(jump_distance=...)` before initializing spins.
    • You must initialize spins before calling `anneal()` or `thermalize()`.
    • Saving includes spin configurations and annealing metadata, but does NOT
      include the grid itself; the same grid must exist on the object before `load()`.
    """

    _SAVE_FORMAT_VERSION = "3.0"

    def __init__(
        self,
        grid: NDArray[np.float64],
        jump_distance: float | None = None,
        system_type: SystemType | str | None = None,
        delta: DeltaType | str = DeltaType.TWO,
        num_spins: int | None = None,
        *,
        config: SystemConfig | dict[str, str] | None = None,
        geometry: Geometry | str | None = None,
        constraint: ConstraintModel | str | None = None,
        interaction_model: InteractionModel | str | None = None,
        update_rule: UpdateRule | str | None = None,
    ) -> None:
        resolved_config = self._resolve_config(
            system_type=system_type,
            config=config,
            geometry=geometry,
            constraint=constraint,
            interaction_model=interaction_model,
            update_rule=update_rule,
        )
        coerced_delta = self._coerce_delta(delta)
        coerced_jump_distance = (
            None
            if jump_distance is None
            else self._coerce_jump_distance(jump_distance)
        )

        self._definition = self._build_definition(
            grid=grid,
            config=resolved_config,
            num_spins=num_spins,
        )
        self._neighborhood_model = NeighborhoodModel(
            delta=coerced_delta,
            jump_distance=coerced_jump_distance,
        )
        self._runtime = OptimizationRuntime()

    def __repr__(self) -> str:
        return (
            "System("
            f"geometry={self.geometry.value!r}, "
            f"constraint={self.constraint_model.value!r}, "
            f"interaction_model={self.interaction_model.value!r}, "
            f"update_rule={self.update_rule.value!r}, "
            f"grid_shape={tuple(int(dim) for dim in self.grid.shape)!r}, "
            f"jump_distance={self.jump_distance!r}, "
            f"delta={self.delta.value!r}, "
            f"num_spins={self.num_spins}, "
            f"initialized={self.initialized}"
            ")"
        )

    @property
    def definition(self) -> SystemDefinition:
        return self._definition

    @property
    def neighborhood_model(self) -> NeighborhoodModel:
        return self._neighborhood_model

    @property
    def runtime(self) -> OptimizationRuntime:
        return self._runtime

    @property
    def config(self) -> SystemConfig:
        return self._definition.config

    @property
    def grid(self) -> NDArray[np.float64]:
        return self._definition.grid

    @property
    def jump_distance(self) -> float | None:
        return self._neighborhood_model.jump_distance

    @jump_distance.setter
    def jump_distance(self, value: float | None) -> None:
        self._neighborhood_model.jump_distance = value

    @property
    def system_type(self) -> SystemType:
        return system_type_from_config(self.config)

    @property
    def geometry(self) -> Geometry:
        return self._definition.geometry

    @property
    def constraint(self) -> ConstraintModel:
        return self._definition.constraint

    @property
    def constraint_model(self) -> ConstraintModel:
        return self.constraint

    @property
    def interaction_model(self) -> InteractionModel:
        return self._definition.interaction_model

    @property
    def update_rule(self) -> UpdateRule:
        return self._definition.update_rule

    def _set_config(self, config: SystemConfig) -> None:
        self._definition.config = config
        self._definition.system_shape = self._derive_system_shape(
            config=config,
            n_sites=int(self.grid.shape[0]),
            num_spins=int(self.num_spins),
        )

    @config.setter
    def config(self, value: SystemConfig | dict[str, str]) -> None:
        self._set_config(self._coerce_config(value))

    @property
    def delta(self) -> DeltaType:
        return self._neighborhood_model.delta

    @delta.setter
    def delta(self, value: DeltaType | str) -> None:
        self._neighborhood_model.delta = self._coerce_delta(value)

    @property
    def num_spins(self) -> int:
        return self._definition.num_spins

    @num_spins.setter
    def num_spins(self, value: int) -> None:
        self._definition.num_spins = int(value)
        self._definition.system_shape = self._derive_system_shape(
            config=self.config,
            n_sites=int(self.grid.shape[0]),
            num_spins=int(self._definition.num_spins),
        )

    @property
    def dimension(self) -> int:
        return self._definition.dimension

    @dimension.setter
    def dimension(self, value: int) -> None:
        self._definition.dimension = int(value)

    @property
    def antipodal(self) -> bool:
        return self._definition.antipodal

    @antipodal.setter
    def antipodal(self, value: bool) -> None:
        if bool(value):
            if self.antipodal:
                return
            self._set_config(
                SystemConfig(
                    geometry=self.geometry,
                    constraint=ConstraintModel.ANTIPODAL,
                    interaction_model=InteractionModel.INVERSE,
                    update_rule=UpdateRule.FLIP,
                )
            )
            return

        if not self.antipodal:
            return
        self._set_config(
            SystemConfig(
                geometry=self.geometry,
                constraint=ConstraintModel.FIXED_DENSITY,
                interaction_model=InteractionModel.OCCUPANCY,
                update_rule=UpdateRule.SWAP,
            )
        )

    @property
    def initialized(self) -> bool:
        return self._runtime.initialized

    @initialized.setter
    def initialized(self, value: bool) -> None:
        self._runtime.initialized = bool(value)

    @property
    def neighborhood(self) -> NeighborhoodData | None:
        return self._neighborhood_model.neighborhood

    @neighborhood.setter
    def neighborhood(self, value: NeighborhoodData | None) -> None:
        self._neighborhood_model.neighborhood = value

    @property
    def prob_prefactor(self) -> float:
        return self._neighborhood_model.prob_prefactor

    @prob_prefactor.setter
    def prob_prefactor(self, value: float) -> None:
        self._neighborhood_model.prob_prefactor = float(value)

    @property
    def system_shape(self) -> int | tuple[int, ...] | None:
        return self._definition.system_shape

    @system_shape.setter
    def system_shape(self, value: int | tuple[int, ...] | None) -> None:
        self._definition.system_shape = value

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

    @anneal_data.setter
    def anneal_data(self, value: AnnealData) -> None:
        self._runtime.anneal_data = value

    @property
    def greedy_history(self) -> GreedyHistory:
        return self._runtime.greedy_history

    @greedy_history.setter
    def greedy_history(self, value: GreedyHistory) -> None:
        self._runtime.greedy_history = value

    @property
    def parallel_tempering_history(self) -> ParallelTemperingHistory:
        return self._runtime.parallel_tempering_history

    @parallel_tempering_history.setter
    def parallel_tempering_history(self, value: ParallelTemperingHistory) -> None:
        self._runtime.parallel_tempering_history = value

    @property
    def occupancy(self) -> OccupancyState | None:
        return self._runtime.occupancy

    @occupancy.setter
    def occupancy(self, value: OccupancyState | None) -> None:
        self._runtime.occupancy = value

    @classmethod
    def _coerce_system_type(cls, value: SystemType | str) -> SystemType:
        if isinstance(value, SystemType):
            return value
        if isinstance(value, str):
            try:
                return SystemType(value.lower().replace("_", "-"))
            except ValueError as exc:
                valid = ", ".join(system_type.value for system_type in SystemType)
                raise ValueError(
                    f"Unknown system_type {value!r}. Expected one of: {valid}."
                ) from exc
        raise TypeError("system_type must be a SystemType or string.")

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
        value: SystemConfig | dict[str, str],
    ) -> SystemConfig:
        if isinstance(value, SystemConfig):
            return value
        if isinstance(value, dict):
            missing = {
                "geometry",
                "constraint",
                "interaction_model",
                "update_rule",
            } - set(value)
            if missing:
                raise TypeError(
                    f"SystemConfig mapping is missing required field(s): {sorted(missing)}."
                )
            return SystemConfig(
                geometry=cls._coerce_geometry(value["geometry"]),
                constraint=cls._coerce_constraint(value["constraint"]),
                interaction_model=cls._coerce_interaction_model(
                    value["interaction_model"]
                ),
                update_rule=cls._coerce_update_rule(value["update_rule"]),
            )
        raise TypeError("config must be a SystemConfig or mapping.")

    @classmethod
    def _resolve_config(
        cls,
        *,
        system_type: SystemType | str | None,
        config: SystemConfig | dict[str, str] | None,
        geometry: Geometry | str | None,
        constraint: ConstraintModel | str | None,
        interaction_model: InteractionModel | str | None,
        update_rule: UpdateRule | str | None,
    ) -> SystemConfig:
        orthogonal_values = (
            geometry,
            constraint,
            interaction_model,
            update_rule,
        )
        if config is not None:
            if system_type is not None or any(value is not None for value in orthogonal_values):
                raise ValueError(
                    "Pass either config=... or the legacy/new individual config fields, not both."
                )
            return cls._coerce_config(config)

        if any(value is not None for value in orthogonal_values):
            if system_type is not None:
                raise ValueError(
                    "Pass either system_type=... or geometry/constraint/interaction_model/update_rule, not both."
                )
            missing = [
                name
                for name, value in (
                    ("geometry", geometry),
                    ("constraint", constraint),
                    ("interaction_model", interaction_model),
                    ("update_rule", update_rule),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "Orthogonal config requires all fields. Missing: "
                    + ", ".join(missing)
                    + "."
                )
            return SystemConfig(
                geometry=cls._coerce_geometry(geometry),
                constraint=cls._coerce_constraint(constraint),
                interaction_model=cls._coerce_interaction_model(interaction_model),
                update_rule=cls._coerce_update_rule(update_rule),
            )

        legacy_system_type = (
            SystemType.INVERSE if system_type is None else cls._coerce_system_type(system_type)
        )
        return config_from_system_type(legacy_system_type)

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
    def _coerce_delta(cls, value: DeltaType | str) -> DeltaType:
        if isinstance(value, DeltaType):
            return value
        if isinstance(value, str):
            delta_key = value.lower().replace("-", "_")
            for delta in DeltaType:
                if delta_key == delta.value.lower().replace("-", "_"):
                    return delta
            valid = ", ".join(delta.value for delta in DeltaType)
            raise ValueError(f"Unknown delta {value!r}. Expected one of: {valid}.")
        raise TypeError("delta must be a DeltaType or string.")

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

    @classmethod
    def _build_definition(
        cls,
        *,
        grid: NDArray[np.float64],
        config: SystemConfig,
        num_spins: int | None,
    ) -> SystemDefinition:
        """Validate constructor inputs and derive static system metadata."""
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
        delta: DeltaType | str | None = None,
        silent: bool = True,
        distance_metric: DistanceMetric | None = None,
    ) -> None:
        """
        Build and store the neighborhood; also compute probability prefactor.
        """
        self.jump_distance = self._coerce_jump_distance(
            self.jump_distance if jump_distance is None else jump_distance
        )
        if delta is not None:
            self.delta = self._coerce_delta(delta)
        else:
            self.delta = self._coerce_delta(self.delta)

        if self.geometry == Geometry.EUCLIDEAN:
            self.prob_prefactor = calculate_prob_prefactor_euclidean(
                len(self.grid), self.jump_distance, self.dimension
            )
        else:
            self.prob_prefactor = calculate_prob_prefactor(
                len(self.grid), self.jump_distance
            )

        if self.geometry == Geometry.EUCLIDEAN:
            metric = "euclidean" if distance_metric is None else distance_metric
            rows = build_euclidean_neighborhood(
                self.grid,
                self.jump_distance,
                delta=self.delta.value,
                silent=silent,
                distance_metric=metric,
            )
            self.neighborhood = NonAntipodalNeighborhoodCSR.from_rows(rows)
            return

        rows = build_neighborhood(
            self.grid,
            self.jump_distance,
            antipodal=self.antipodal,
            delta=self.delta.value,
            silent=silent,
            distance_metric="spherical" if distance_metric is None else distance_metric,
        )
        self.neighborhood = (
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
                "System needs to be initialized. Call random_init(), hemisphere_init(), tri_cog_init(), cos_cog_init(), or from_array_init() first."
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
            jump_distance=self.jump_distance,
            neighborhood=self.neighborhood,
            prob_prefactor=float(self.prob_prefactor),
        )
        clone._runtime = self.runtime.copy()
        return clone

    def __copy__(self) -> "System":
        return self.clone()

    def set_occupancy(self, occupancy: OccupancyState | None) -> None:
        """Attach fixed-density occupancy lists."""
        self.occupancy = None if occupancy is None else occupancy.copy()

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
        return self.runtime.snapshot_state()

    def restore_state(self, state: SystemState) -> None:
        """Restore mutable spin state from a snapshot."""
        self.runtime.restore_state(state)

    # --- initializers ---
    def random_init(self) -> None:
        self._require_neighborhood()
        init_random(self)
        self.initialized = True

    def hemisphere_init(self) -> None:
        self._require_neighborhood()
        if self.geometry == Geometry.EUCLIDEAN:
            raise ValueError("hemisphere_init() requires a spherical grid.")
        init_random_hemisphere(self)
        self.initialized = True

    def tri_cog_init(self, num_cogs: int, cog_height: float) -> None:
        self._require_neighborhood()
        init_triangle_cogwheel(self, num_cogs, cog_height)
        self.initialized = True

    def cos_cog_init(self, num_cogs: int, cog_height: float) -> None:
        self._require_neighborhood()
        init_cosine_cogwheel(self, num_cogs, cog_height)
        self.initialized = True

    def array_init(self, spin_array: NDArray[np.bool_]) -> None:
        self._require_neighborhood()
        init_from_array(self, spin_array)
        self.initialized = True

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
        )

    def greedy(
        self,
        *,
        max_steps: int | None = None,
        min_delta: float = 0.0,
        track_best: bool = True,
        reinit: bool = True,
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
        reinit: bool = True,
        track_best: bool = True,
        save_path: str | Path | None = None,
        seed: int | None = None,
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
        )

    # --- visualization ---
    def plot(
        self,
        plot_current: bool = False,
        auto_rotate: bool = False,
        plot_figure: bool = True,
    ) -> None:
        system_to_plot = self.current_system if plot_current else self.best_system
        plot_spin_system(
            system_config=self.config,
            grid_points=self.grid,
            spin_states=system_to_plot,
            fig_title="Current System" if plot_current else "Best System",
            rotate=auto_rotate,
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
        independent = self.constraint_model == ConstraintModel.INDEPENDENT
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
        if self.constraint_model == ConstraintModel.ANTIPODAL:
            dims = ("spin",)
            coords = {"spin": np.arange(self.num_spins)}
        elif self.constraint_model == ConstraintModel.INDEPENDENT:
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

    def to_dataset(self) -> xr.Dataset:
        """Return an xarray Dataset containing serializable system state."""
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        coords: dict[str, np.ndarray] = {}

        spin_dims, spin_coords = self._spin_dims_and_coords()
        coords.update(spin_coords)

        if self.current_system is not None:
            data_vars["current_system"] = (spin_dims, self.current_system)
        if self.best_system is not None:
            data_vars["best_system"] = (spin_dims, self.best_system)

        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Minimal attrs matching the current SystemConfig-based API.
        attrs = {
            "format_version": self._SAVE_FORMAT_VERSION,
            "geometry": self.geometry.value,
            "constraint": self.constraint_model.value,
            "interaction_model": self.interaction_model.value,
            "update_rule": self.update_rule.value,
            "delta": self.delta.value,
            "initialized": bool(self.initialized),
            "num_spins": int(self.num_spins),
            "site_count": int(self.grid.shape[0]),
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
        saved_delta = self._coerce_delta(self._require_dataset_attr(ds, "delta"))

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

        # Enums and core attrs
        if saved_config != self.config:
            raise ValueError(
                "Saved state config does not match this instance. "
                f"saved={saved_config!r} current={self.config!r}"
            )
        self.config = saved_config
        self.delta = saved_delta
        self.jump_distance = (
            self._coerce_jump_distance(ds.attrs["jump_distance"])
            if "jump_distance" in ds.attrs
            else None
        )
        self.initialized = bool(self._require_dataset_attr(ds, "initialized"))
        loaded_num_spins = int(self._require_dataset_attr(ds, "num_spins"))
        if saved_config.geometry == Geometry.EUCLIDEAN and not (
            0 < loaded_num_spins < self.grid.shape[0]
        ):
            raise ValueError(
                f"Saved Euclidean num_spins must be between 1 and N-1 "
                f"for N={self.grid.shape[0]} grid sites, got {loaded_num_spins}."
            )
        self.num_spins = loaded_num_spins
        self.current_energy = float(self._require_dataset_attr(ds, "current_energy"))
        self.best_energy = float(self._require_dataset_attr(ds, "best_energy"))
        self.anneal_data = self._history_from_json(
            self._require_dataset_attr(ds, "anneal_data_json"),
            AnnealData,
            attr_name="anneal_data_json",
        )
        self.greedy_history = self._history_from_json(
            self._require_dataset_attr(ds, "greedy_history_json"),
            GreedyHistory,
            attr_name="greedy_history_json",
        )
        self.parallel_tempering_history = self._history_from_json(
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
        self.dimension = int(self.grid.shape[1])
        if self.jump_distance is None:
            self.prob_prefactor = 0.0
        elif self.geometry == Geometry.EUCLIDEAN:
            self.prob_prefactor = calculate_prob_prefactor_euclidean(
                len(self.grid), self.jump_distance, self.dimension
            )
        else:
            self.prob_prefactor = calculate_prob_prefactor(
                len(self.grid), self.jump_distance
            )

        # 4) Safety: neighborhood must be rebuilt
        self.neighborhood = None

        # 5) If no spin state present, mark uninitialized
        if self.current_system is None and self.best_system is None:
            self.initialized = False
