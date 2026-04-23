from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray


@dataclass
class Neighborhood:
    positive_spins: NDArray[np.int_]
    positive_contributions: NDArray[np.float64]
    negative_spins: NDArray[np.int_]
    negative_contributions: NDArray[np.float64]
    spin_potential: float


@dataclass
class NonAntipodalNeighborhood:
    neighbor_spins: NDArray[np.int_]
    contributions: NDArray[np.float64]
    spin_potential: float


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
        sorted_neighbor_rows: list[NDArray[np.int64]] = []
        sorted_contribution_rows: list[NDArray[np.float64]] = []
        for row in rows:
            neighbor_row = np.asarray(row.neighbor_spins, dtype=np.int64)
            contribution_row = np.asarray(row.contributions, dtype=np.float64)
            if neighbor_row.shape != contribution_row.shape:
                raise ValueError(
                    "neighbor_spins and contributions must have matching shapes."
                )
            if neighbor_row.size > 1:
                order = np.argsort(neighbor_row, kind="stable")
                neighbor_row = neighbor_row[order]
                contribution_row = contribution_row[order]
            sorted_neighbor_rows.append(np.ascontiguousarray(neighbor_row))
            sorted_contribution_rows.append(np.ascontiguousarray(contribution_row))

        lengths = np.fromiter(
            (len(row) for row in sorted_neighbor_rows),
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
            neighbor_spins = np.concatenate(sorted_neighbor_rows).astype(
                np.int64,
                copy=False,
            )
            contributions = np.concatenate(sorted_contribution_rows).astype(
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


class SystemType(str, Enum):
    INVERSE = "inverse"
    INDEPENDENT = "independent"
    NON_ANTIPODAL = "non-antipodal"
    EUCLIDEAN = "euclidean"


class DeltaType(str, Enum):
    TWO = "two"
    ONE_STAR = "one*"
    ONE_STAR_NAME = "one_star"
    TWO_STAR = "two*"
    TWO_STAR_NAME = "two_star"
    EXACT = "exact"
    EXACT_DELTA = "exact_delta"
    REAL_DELTA = "real_delta"
    DELTA = "delta"
    DIRAC = "dirac"
