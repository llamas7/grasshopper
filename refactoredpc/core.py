from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .system import System


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
        lengths = np.fromiter(
            (len(row.neighbor_spins) for row in rows),
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
            neighbor_spins = np.concatenate(
                [row.neighbor_spins for row in rows]
            ).astype(np.int64, copy=False)
            contributions = np.concatenate(
                [row.contributions for row in rows]
            ).astype(np.float64, copy=False)
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


NeighborhoodLike = Neighborhood | NonAntipodalNeighborhood
NeighborhoodData = list[Neighborhood] | list[NonAntipodalNeighborhood] | NonAntipodalNeighborhoodCSR


@dataclass(frozen=True)
class Engine:
    """Typed bundle of operations for one system update rule."""

    mode: str
    energy: Callable[..., float]
    grid_gen: Callable[..., NDArray[np.float64]]
    update: Callable[..., None]
    suggest: Callable[..., tuple[Any, float]]
    best_update: Callable[..., tuple[Any | None, float]]

    def __getitem__(self, key: str) -> Any:
        """Compatibility shim for older internal dictionary-style access."""
        return getattr(self, key)


@dataclass
class OccupancyState:
    """Cached occupied/unoccupied site lists for fixed-density systems."""

    occupied: list[int] = field(default_factory=list)
    unoccupied: list[int] = field(default_factory=list)

    @classmethod
    def from_spin_array(cls, spin_array: NDArray[np.bool_]) -> "OccupancyState":
        spins = np.asarray(spin_array, dtype=np.bool_)
        if spins.ndim != 1:
            raise ValueError(
                f"OccupancyState requires a one-dimensional spin array, got {spins.shape}."
            )
        return cls(
            occupied=np.flatnonzero(spins).astype(np.intp).tolist(),
            unoccupied=np.flatnonzero(~spins).astype(np.intp).tolist(),
        )

    def copy(self) -> "OccupancyState":
        return OccupancyState(
            occupied=list(self.occupied),
            unoccupied=list(self.unoccupied),
        )

    def validate(
            self,
            *,
            n_sites: int | None = None,
            expected_occupied: int | None = None,
            spin_array: NDArray[np.bool_] | None = None,
    ) -> None:
        occupied = [int(site) for site in self.occupied]
        unoccupied = [int(site) for site in self.unoccupied]
        if len(set(occupied)) != len(occupied):
            raise ValueError("occupied sites contain duplicates.")
        if len(set(unoccupied)) != len(unoccupied):
            raise ValueError("unoccupied sites contain duplicates.")
        if set(occupied).intersection(unoccupied):
            raise ValueError("occupied and unoccupied sites overlap.")
        if expected_occupied is not None and len(occupied) != int(expected_occupied):
            raise ValueError(
                f"Expected {expected_occupied} occupied sites, got {len(occupied)}."
            )
        if n_sites is not None:
            all_sites = set(range(int(n_sites)))
            listed_sites = set(occupied).union(unoccupied)
            if listed_sites != all_sites:
                missing = sorted(all_sites - listed_sites)
                extra = sorted(listed_sites - all_sites)
                raise ValueError(
                    "Occupancy lists must cover every site exactly once. "
                    f"missing={missing[:10]}, extra={extra[:10]}"
                )
        if spin_array is not None:
            expected = OccupancyState.from_spin_array(spin_array)
            if occupied != expected.occupied or unoccupied != expected.unoccupied:
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


@dataclass
class Parameters:
    starting_temp: float = 50
    final_temp: float = 0.05
    cooling_rate: float = 0.954993
    min_acceptance: float = 0.0005
    max_runtime: float = 1  # Hours
    max_iterations: int = 100
    starting_steps: int = 100000
    its_per_save: int = 1
    grid_file: str | None = None
    save_dir: str = "./output_data"
    load_flag: bool = False


@dataclass
class AnnealData:
    steps: int
    temperature: float
    iterations: int = 0
    energies: list[float] = field(default_factory=list)
    update_stats: list[list] = field(default_factory=list)


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
