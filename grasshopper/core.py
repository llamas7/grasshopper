from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray


def _normalize_sparse_row(
    indices,
    values,
    *,
    index_name: str,
    value_name: str,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    normalized_indices = np.asarray(indices, dtype=np.int64)
    normalized_values = np.asarray(values, dtype=np.float64)
    if normalized_indices.ndim != 1:
        raise ValueError(f"{index_name} must be one-dimensional, got {normalized_indices.shape}.")
    if normalized_values.ndim != 1:
        raise ValueError(f"{value_name} must be one-dimensional, got {normalized_values.shape}.")
    if normalized_indices.shape != normalized_values.shape:
        raise ValueError(
            f"{index_name} and {value_name} must have matching shapes, "
            f"got {normalized_indices.shape} and {normalized_values.shape}."
        )
    if normalized_indices.size > 1:
        order = np.argsort(normalized_indices, kind="stable")
        normalized_indices = normalized_indices[order]
        normalized_values = normalized_values[order]
    return (
        np.ascontiguousarray(normalized_indices, dtype=np.int64),
        np.ascontiguousarray(normalized_values, dtype=np.float64),
    )


@dataclass(frozen=True)
class Neighborhood:
    positive_spins: NDArray[np.int_]
    positive_contributions: NDArray[np.float64]
    negative_spins: NDArray[np.int_]
    negative_contributions: NDArray[np.float64]
    spin_potential: float

    def __post_init__(self) -> None:
        positive_spins, positive_contributions = _normalize_sparse_row(
            self.positive_spins,
            self.positive_contributions,
            index_name="positive_spins",
            value_name="positive_contributions",
        )
        negative_spins, negative_contributions = _normalize_sparse_row(
            self.negative_spins,
            self.negative_contributions,
            index_name="negative_spins",
            value_name="negative_contributions",
        )
        object.__setattr__(self, "positive_spins", positive_spins)
        object.__setattr__(self, "positive_contributions", positive_contributions)
        object.__setattr__(self, "negative_spins", negative_spins)
        object.__setattr__(self, "negative_contributions", negative_contributions)
        object.__setattr__(self, "spin_potential", float(self.spin_potential))


@dataclass(frozen=True)
class NonAntipodalNeighborhood:
    neighbor_spins: NDArray[np.int_]
    contributions: NDArray[np.float64]
    spin_potential: float

    def __post_init__(self) -> None:
        neighbor_spins, contributions = _normalize_sparse_row(
            self.neighbor_spins,
            self.contributions,
            index_name="neighbor_spins",
            value_name="contributions",
        )
        object.__setattr__(self, "neighbor_spins", neighbor_spins)
        object.__setattr__(self, "contributions", contributions)
        object.__setattr__(self, "spin_potential", float(self.spin_potential))

    def __len__(self) -> int:
        return int(self.neighbor_spins.size)


@dataclass(frozen=True)
class NonAntipodalNeighborhoodCSR:
    """CSR-style fixed-density neighborhood storage.

    The object keeps contiguous arrays for fast whole-system reductions while
    still supporting ``neighborhood[i]`` row access for existing update code.
    """

    indptr: NDArray[np.int64]
    neighbor_spins: NDArray[np.int64]
    contributions: NDArray[np.float64]
    spin_potential: NDArray[np.float64]

    @classmethod
    def from_rows(
        cls,
        rows: list[NonAntipodalNeighborhood],
    ) -> "NonAntipodalNeighborhoodCSR":
        neighbor_rows = [
            np.ascontiguousarray(row.neighbor_spins, dtype=np.int64) for row in rows
        ]
        contribution_rows = [
            np.ascontiguousarray(row.contributions, dtype=np.float64) for row in rows
        ]

        lengths = np.fromiter(
            (row.neighbor_spins.size for row in rows),
            count=len(rows),
            dtype=np.int64,
        )
        indptr = np.empty(len(rows) + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(lengths, out=indptr[1:])
        if indptr[-1] == 0:
            neighbor_spins = np.array([], dtype=np.int64)
            contributions = np.array([], dtype=np.float64)
        else:
            neighbor_spins = np.concatenate(neighbor_rows).astype(
                np.int64,
                copy=False,
            )
            contributions = np.concatenate(contribution_rows).astype(
                np.float64, copy=False
            )
        spin_potential = np.asarray(
            [row.spin_potential for row in rows],
            dtype=np.float64,
        )
        return cls(
            indptr=indptr,
            neighbor_spins=np.ascontiguousarray(neighbor_spins),
            contributions=np.ascontiguousarray(contributions),
            spin_potential=np.ascontiguousarray(spin_potential),
        )

    def __len__(self) -> int:
        return int(self.indptr.size - 1)

    def __getitem__(self, index: int) -> NonAntipodalNeighborhood:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("NonAntipodalNeighborhoodCSR indices must be integers.")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = int(self.indptr[index])
        stop = int(self.indptr[index + 1])
        return NonAntipodalNeighborhood(
            neighbor_spins=self.neighbor_spins[start:stop],
            contributions=self.contributions[start:stop],
            spin_potential=float(self.spin_potential[index]),
        )


NeighborhoodData = (
    list[Neighborhood] | list[NonAntipodalNeighborhood] | NonAntipodalNeighborhoodCSR
)


@dataclass(frozen=True)
class Engine:
    """Typed bundle of operations for one system update rule."""

    mode: str
    energy: Callable[..., float]
    grid_gen: Callable[..., NDArray[np.float64]]
    update: Callable[..., None]
    suggest: Callable[..., tuple[Any, float]]
    best_update: Callable[..., tuple[Any | None, float]]


@dataclass
class OccupancyState:
    """Cached occupied/unoccupied site arrays for fixed-density systems."""

    occupied: NDArray[np.intp] = field(
        default_factory=lambda: np.empty(0, dtype=np.intp)
    )
    unoccupied: NDArray[np.intp] = field(
        default_factory=lambda: np.empty(0, dtype=np.intp)
    )

    def __post_init__(self) -> None:
        self.occupied = np.ascontiguousarray(self.occupied, dtype=np.intp)
        self.unoccupied = np.ascontiguousarray(self.unoccupied, dtype=np.intp)

    @classmethod
    def from_spin_array(cls, spin_array: NDArray[np.bool_]) -> "OccupancyState":
        spins = np.asarray(spin_array, dtype=np.bool_)
        if spins.ndim != 1:
            raise ValueError(
                f"OccupancyState requires a one-dimensional spin array, got {spins.shape}."
            )
        return cls(
            occupied=np.ascontiguousarray(np.flatnonzero(spins), dtype=np.intp),
            unoccupied=np.ascontiguousarray(np.flatnonzero(~spins), dtype=np.intp),
        )

    def copy(self) -> "OccupancyState":
        return OccupancyState(
            occupied=self.occupied.copy(),
            unoccupied=self.unoccupied.copy(),
        )

    def validate(
        self,
        *,
        n_sites: int | None = None,
        expected_occupied: int | None = None,
        spin_array: NDArray[np.bool_] | None = None,
    ) -> None:
        occupied = np.asarray(self.occupied, dtype=np.intp)
        unoccupied = np.asarray(self.unoccupied, dtype=np.intp)
        occupied_list = occupied.astype(int, copy=False).tolist()
        unoccupied_list = unoccupied.astype(int, copy=False).tolist()
        if len(set(occupied_list)) != occupied.size:
            raise ValueError("occupied sites contain duplicates.")
        if len(set(unoccupied_list)) != unoccupied.size:
            raise ValueError("unoccupied sites contain duplicates.")
        if set(occupied_list).intersection(unoccupied_list):
            raise ValueError("occupied and unoccupied sites overlap.")
        if expected_occupied is not None and occupied.size != int(expected_occupied):
            raise ValueError(
                f"Expected {expected_occupied} occupied sites, got {occupied.size}."
            )
        if n_sites is not None:
            all_sites = set(range(int(n_sites)))
            listed_sites = set(occupied_list).union(unoccupied_list)
            if listed_sites != all_sites:
                missing = sorted(all_sites - listed_sites)
                extra = sorted(listed_sites - all_sites)
                raise ValueError(
                    "Occupancy lists must cover every site exactly once. "
                    f"missing={missing[:10]}, extra={extra[:10]}"
                )
        if spin_array is not None:
            expected = OccupancyState.from_spin_array(spin_array)
            if set(occupied_list) != set(
                expected.occupied.astype(int, copy=False).tolist()
            ) or set(unoccupied_list) != set(
                expected.unoccupied.astype(int, copy=False).tolist()
            ):
                raise ValueError("Occupancy lists do not match the spin array.")


@dataclass
class SystemState:
    """Mutable spin state separated from static system metadata."""

    current_system: NDArray[np.bool_] | None = None
    current_energy: float = 0.0
    best_system: NDArray[np.bool_] | None = None
    best_energy: float = 0.0
    occupancy: OccupancyState | None = None

    def copy(self) -> "SystemState":
        return SystemState(
            current_system=(
                None if self.current_system is None else self.current_system.copy()
            ),
            current_energy=float(self.current_energy),
            best_system=None if self.best_system is None else self.best_system.copy(),
            best_energy=float(self.best_energy),
            occupancy=None if self.occupancy is None else self.occupancy.copy(),
        )


@dataclass(frozen=True)
class AnnealSchedule:
    """Immutable simulated-annealing schedule configuration."""

    starting_temp: float = 50.0
    final_temp: float = 0.05
    cooling_rate: float = 0.954993
    min_acceptance: float = 0.00005
    max_runtime_hours: float = 1.0
    max_iterations: int = 100
    starting_steps: int = 100_000

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "AnnealSchedule":
        if not data:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise TypeError(f"Unknown AnnealSchedule field(s): {unknown}.")
        return cls(**dict(data))

    def validate(self) -> None:
        if self.starting_temp <= 0:
            raise ValueError("anneal_schedule.starting_temp must be positive.")
        if self.final_temp < 0:
            raise ValueError("anneal_schedule.final_temp must be nonnegative.")
        if self.cooling_rate <= 0 or self.cooling_rate >= 1:
            raise ValueError("anneal_schedule.cooling_rate must be between 0 and 1.")
        if self.min_acceptance < 0:
            raise ValueError("anneal_schedule.min_acceptance must be nonnegative.")
        if self.max_runtime_hours < 0:
            raise ValueError("anneal_schedule.max_runtime_hours must be nonnegative.")
        if self.max_iterations < 0:
            raise ValueError("anneal_schedule.max_iterations must be nonnegative.")
        if self.starting_steps <= 0:
            raise ValueError("anneal_schedule.starting_steps must be positive.")


@dataclass(frozen=True)
class AnnealRecord:
    iteration: int
    temperature: float
    steps: int
    acceptance_rate: float
    current_energy: float
    best_energy: float


@dataclass
class AnnealData:
    iterations: int = 0
    records: list[AnnealRecord] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "AnnealData":
        if not data:
            return cls()

        records: list[AnnealRecord] = []
        for record in data.get("records", []) or []:
            if isinstance(record, AnnealRecord):
                records.append(record)
            elif isinstance(record, dict):
                records.append(AnnealRecord(**record))

        return cls(
            iterations=int(data.get("iterations", len(records))),
            records=records,
        )

    def append_record(
        self,
        *,
        temperature: float,
        steps: int,
        acceptance_rate: float,
        current_energy: float,
        best_energy: float,
    ) -> None:
        self.records.append(
            AnnealRecord(
                iteration=int(self.iterations),
                temperature=float(temperature),
                steps=int(steps),
                acceptance_rate=float(acceptance_rate),
                current_energy=float(current_energy),
                best_energy=float(best_energy),
            )
        )

    def copy(self) -> "AnnealData":
        return AnnealData(
            iterations=int(self.iterations),
            records=list(self.records),
        )


@dataclass(frozen=True)
class GreedyRecord:
    run: int
    steps: int
    current_energy: float
    best_energy: float
    last_delta: float | None
    converged: bool


@dataclass
class GreedyHistory:
    runs: int = 0
    records: list[GreedyRecord] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "GreedyHistory":
        if not data:
            return cls()

        records: list[GreedyRecord] = []
        for record in data.get("records", []) or []:
            if isinstance(record, GreedyRecord):
                records.append(record)
            elif isinstance(record, dict):
                records.append(GreedyRecord(**record))

        return cls(
            runs=int(data.get("runs", len(records))),
            records=records,
        )

    def append_result(
        self,
        *,
        steps: int,
        current_energy: float,
        best_energy: float,
        last_delta: float | None,
        converged: bool,
    ) -> None:
        self.runs += 1
        self.records.append(
            GreedyRecord(
                run=int(self.runs),
                steps=int(steps),
                current_energy=float(current_energy),
                best_energy=float(best_energy),
                last_delta=None if last_delta is None else float(last_delta),
                converged=bool(converged),
            )
        )

    def copy(self) -> "GreedyHistory":
        return GreedyHistory(
            runs=int(self.runs),
            records=list(self.records),
        )


@dataclass(frozen=True)
class ParallelTemperingRecord:
    run: int
    iterations: int
    temperatures: tuple[float, ...]
    best_energy: float
    cold_energy: float
    final_energies: tuple[float, ...]
    cold_energy_history: tuple[float, ...]
    sweep_acceptance: tuple[tuple[float, ...], ...]
    exchange_attempts: tuple[int, ...]
    exchange_accepts: tuple[int, ...]
    exchange_rates: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "run", int(self.run))
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(
            self,
            "temperatures",
            tuple(float(value) for value in self.temperatures),
        )
        object.__setattr__(self, "best_energy", float(self.best_energy))
        object.__setattr__(self, "cold_energy", float(self.cold_energy))
        object.__setattr__(
            self,
            "final_energies",
            tuple(float(value) for value in self.final_energies),
        )
        object.__setattr__(
            self,
            "cold_energy_history",
            tuple(float(value) for value in self.cold_energy_history),
        )
        object.__setattr__(
            self,
            "sweep_acceptance",
            tuple(
                tuple(float(value) for value in iteration)
                for iteration in self.sweep_acceptance
            ),
        )
        object.__setattr__(
            self,
            "exchange_attempts",
            tuple(int(value) for value in self.exchange_attempts),
        )
        object.__setattr__(
            self,
            "exchange_accepts",
            tuple(int(value) for value in self.exchange_accepts),
        )
        object.__setattr__(
            self,
            "exchange_rates",
            tuple(float(value) for value in self.exchange_rates),
        )


@dataclass
class ParallelTemperingHistory:
    runs: int = 0
    records: list[ParallelTemperingRecord] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "ParallelTemperingHistory":
        if not data:
            return cls()

        records: list[ParallelTemperingRecord] = []
        for record in data.get("records", []) or []:
            if isinstance(record, ParallelTemperingRecord):
                records.append(record)
            elif isinstance(record, dict):
                records.append(ParallelTemperingRecord(**record))

        return cls(
            runs=int(data.get("runs", len(records))),
            records=records,
        )

    def append_run(
        self,
        *,
        iterations: int,
        temperatures,
        best_energy: float,
        cold_energy: float,
        final_energies,
        cold_energy_history,
        sweep_acceptance,
        exchange_attempts,
        exchange_accepts,
        exchange_rates,
    ) -> None:
        self.runs += 1
        self.records.append(
            ParallelTemperingRecord(
                run=int(self.runs),
                iterations=int(iterations),
                temperatures=tuple(temperatures),
                best_energy=float(best_energy),
                cold_energy=float(cold_energy),
                final_energies=tuple(final_energies),
                cold_energy_history=tuple(cold_energy_history),
                sweep_acceptance=tuple(tuple(iteration) for iteration in sweep_acceptance),
                exchange_attempts=tuple(exchange_attempts),
                exchange_accepts=tuple(exchange_accepts),
                exchange_rates=tuple(exchange_rates),
            )
        )

    def copy(self) -> "ParallelTemperingHistory":
        return ParallelTemperingHistory(
            runs=int(self.runs),
            records=list(self.records),
        )


class Geometry(str, Enum):
    SPHERICAL = "spherical"
    EUCLIDEAN = "euclidean"


class ConstraintModel(str, Enum):
    ANTIPODAL = "antipodal"
    INDEPENDENT = "independent"
    FIXED_DENSITY = "fixed-density"


class InteractionModel(str, Enum):
    INVERSE = "inverse"
    INDEPENDENT = "independent"
    OCCUPANCY = "occupancy"


class UpdateRule(str, Enum):
    FLIP = "flip"
    SWAP = "swap"


@dataclass(frozen=True)
class KernelSpec:
    """Serializable specification for a built-in radial interaction kernel."""

    kind: str
    sigma: float | None = None
    cutoff: float | None = None

    def __post_init__(self) -> None:
        aliases = {
            "cosine": "cosine",
            "cell_averaged_hat": "cell_averaged_hat",
            "cell_averaged_cosine": "cell_averaged_cosine",
            "exact": "exact",
            "gaussian": "gaussian",
        }
        key = str(self.kind).lower().replace("-", "_")
        try:
            canonical = aliases[key]
        except KeyError as exc:
            valid = ", ".join(sorted(set(aliases.values())))
            raise ValueError(
                f"Unknown kernel kind {self.kind!r}. Expected one of: {valid}."
            ) from exc
        object.__setattr__(self, "kind", canonical)

        if canonical == "gaussian":
            if self.sigma is None:
                raise ValueError("Gaussian kernels require sigma.")
            sigma = float(self.sigma)
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError("Gaussian sigma must be a positive finite number.")
            object.__setattr__(self, "sigma", sigma)
            if self.cutoff is not None:
                cutoff = float(self.cutoff)
                if not np.isfinite(cutoff) or cutoff <= 0:
                    raise ValueError(
                        "Gaussian cutoff must be positive and finite, or None."
                    )
                object.__setattr__(self, "cutoff", cutoff)
        elif self.sigma is not None or self.cutoff is not None:
            raise ValueError(
                "sigma and cutoff are only valid for Gaussian kernels."
            )

    @classmethod
    def cosine(cls) -> "KernelSpec":
        """Return the compact cosine kernel with support ``2h``."""
        return cls("cosine")

    @classmethod
    def cell_averaged_hat(cls) -> "KernelSpec":
        """Return the cell-averaged hat kernel with support ``3h/2``."""
        return cls("cell_averaged_hat")

    @classmethod
    def cell_averaged_cosine(cls) -> "KernelSpec":
        """Return the cell-averaged cosine kernel with support ``5h/2``."""
        return cls("cell_averaged_cosine")

    @classmethod
    def exact(cls) -> "KernelSpec":
        return cls("exact")

    @classmethod
    def gaussian(
        cls,
        sigma: float,
        cutoff: float | None = 5.0,
    ) -> "KernelSpec":
        return cls("gaussian", sigma=sigma, cutoff=cutoff)

    @classmethod
    def from_value(
        cls,
        value: "KernelSpec | str",
    ) -> "KernelSpec":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        raise TypeError("kernel must be a KernelSpec, string, or callable.")


@dataclass(frozen=True)
class SystemConfig:
    """Orthogonal system description for a supported optimizer/system family."""

    geometry: Geometry
    constraint: ConstraintModel
    interaction_model: InteractionModel
    update_rule: UpdateRule

    def __post_init__(self) -> None:
        supported = {
            (
                Geometry.SPHERICAL,
                ConstraintModel.ANTIPODAL,
                InteractionModel.INVERSE,
                UpdateRule.FLIP,
            ),
            (
                Geometry.SPHERICAL,
                ConstraintModel.INDEPENDENT,
                InteractionModel.INDEPENDENT,
                UpdateRule.FLIP,
            ),
            (
                Geometry.SPHERICAL,
                ConstraintModel.FIXED_DENSITY,
                InteractionModel.OCCUPANCY,
                UpdateRule.SWAP,
            ),
            (
                Geometry.EUCLIDEAN,
                ConstraintModel.FIXED_DENSITY,
                InteractionModel.OCCUPANCY,
                UpdateRule.SWAP,
            ),
        }
        key = (
            self.geometry,
            self.constraint,
            self.interaction_model,
            self.update_rule,
        )
        if key not in supported:
            raise ValueError(
                "Unsupported system configuration: "
                f"geometry={self.geometry.value}, "
                f"constraint={self.constraint.value}, "
                f"interaction_model={self.interaction_model.value}, "
                f"update_rule={self.update_rule.value}."
            )

    @property
    def antipodal(self) -> bool:
        return self.constraint in (
            ConstraintModel.ANTIPODAL,
            ConstraintModel.INDEPENDENT,
        )

    @classmethod
    def inverse(cls) -> "SystemConfig":
        return cls(
            geometry=Geometry.SPHERICAL,
            constraint=ConstraintModel.ANTIPODAL,
            interaction_model=InteractionModel.INVERSE,
            update_rule=UpdateRule.FLIP,
        )

    @classmethod
    def independent(cls) -> "SystemConfig":
        return cls(
            geometry=Geometry.SPHERICAL,
            constraint=ConstraintModel.INDEPENDENT,
            interaction_model=InteractionModel.INDEPENDENT,
            update_rule=UpdateRule.FLIP,
        )

    @classmethod
    def non_antipodal(cls) -> "SystemConfig":
        return cls(
            geometry=Geometry.SPHERICAL,
            constraint=ConstraintModel.FIXED_DENSITY,
            interaction_model=InteractionModel.OCCUPANCY,
            update_rule=UpdateRule.SWAP,
        )

    @classmethod
    def euclidean(cls) -> "SystemConfig":
        return cls(
            geometry=Geometry.EUCLIDEAN,
            constraint=ConstraintModel.FIXED_DENSITY,
            interaction_model=InteractionModel.OCCUPANCY,
            update_rule=UpdateRule.SWAP,
        )


@dataclass
class SystemDefinition:
    """
    Static system metadata derived at construction time.

    For Euclidean systems, `num_spins` is the number of occupied sites, while
    `grid.shape[0]` is the number of candidate lattice sites. The conserved
    occupied volume is fixed to 1.0, so kernel cell length is
    `(1.0 / num_spins) ** (1 / dimension)`.
    """

    grid: NDArray[np.float64]
    config: SystemConfig
    num_spins: int
    dimension: int
    system_shape: int | tuple[int, ...] | None

    @property
    def site_count(self) -> int:
        return int(self.grid.shape[0])

    @property
    def geometry(self) -> Geometry:
        return self.config.geometry

    @property
    def constraint(self) -> ConstraintModel:
        return self.config.constraint

    @property
    def interaction_model(self) -> InteractionModel:
        return self.config.interaction_model

    @property
    def update_rule(self) -> UpdateRule:
        return self.config.update_rule

    @property
    def antipodal(self) -> bool:
        return self.config.antipodal


@dataclass
class NeighborhoodModel:
    """Neighborhood configuration plus cached derived structures."""

    kernel: KernelSpec | Callable[..., Any] | None
    kernel_is_custom: bool = False
    jump_distance: float | tuple[float, ...] | None = None
    jump_weights: tuple[float, ...] | None = None
    distance_metric: Any | None = None
    distance_metric_is_custom: bool = False
    shell_measure: Callable[[float], float] | None = None
    shell_measure_is_custom: bool = False
    radial_bounds: tuple[float, float] | None = None
    neighborhood: NeighborhoodData | None = None


@dataclass
class OptimizationRuntime:
    """Mutable optimization state owned by a system instance."""

    initialized: bool = False
    current_system: NDArray[np.bool_] | None = None
    current_energy: float = 0.0
    best_system: NDArray[np.bool_] | None = None
    best_energy: float = 0.0
    anneal_data: AnnealData = field(default_factory=AnnealData)
    greedy_history: GreedyHistory = field(default_factory=GreedyHistory)
    parallel_tempering_history: ParallelTemperingHistory = field(
        default_factory=ParallelTemperingHistory
    )
    occupancy: OccupancyState | None = None
    temperature: float = field(
        default_factory=lambda: float(AnnealSchedule().starting_temp)
    )

    def snapshot_state(self) -> SystemState:
        return SystemState(
            current_system=(
                None if self.current_system is None else self.current_system.copy()
            ),
            current_energy=float(self.current_energy),
            best_system=None if self.best_system is None else self.best_system.copy(),
            best_energy=float(self.best_energy),
            occupancy=None if self.occupancy is None else self.occupancy.copy(),
        )

    def restore_state(self, state: SystemState) -> None:
        self.current_system = (
            None if state.current_system is None else state.current_system.copy()
        )
        self.current_energy = float(state.current_energy)
        self.best_system = (
            None if state.best_system is None else state.best_system.copy()
        )
        self.best_energy = float(state.best_energy)
        self.occupancy = None if state.occupancy is None else state.occupancy.copy()

    def copy(self) -> "OptimizationRuntime":
        copied = OptimizationRuntime(
            initialized=bool(self.initialized),
            current_system=(
                None if self.current_system is None else self.current_system.copy()
            ),
            current_energy=float(self.current_energy),
            best_system=None if self.best_system is None else self.best_system.copy(),
            best_energy=float(self.best_energy),
            anneal_data=self.anneal_data.copy(),
            greedy_history=self.greedy_history.copy(),
            parallel_tempering_history=self.parallel_tempering_history.copy(),
            occupancy=None if self.occupancy is None else self.occupancy.copy(),
            temperature=float(self.temperature),
        )
        return copied
