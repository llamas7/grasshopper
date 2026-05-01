from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import Neighborhood

try:
    from numba import njit
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    njit = None


NUMBA_AVAILABLE = njit is not None


@dataclass(frozen=True)
class _PackedNeighborhood:
    positive_indptr: np.ndarray
    positive_spins: np.ndarray
    positive_contributions: np.ndarray
    negative_indptr: np.ndarray
    negative_spins: np.ndarray
    negative_contributions: np.ndarray
    spin_potential: np.ndarray


_PACKED_NEIGHBORHOOD_CACHE: dict[int, _PackedNeighborhood] = {}


def _pack_rows(
    index_rows: list[np.ndarray],
    contribution_rows: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.fromiter((row.size for row in index_rows), count=len(index_rows), dtype=np.int64)
    indptr = np.empty(len(index_rows) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(lengths, out=indptr[1:])

    if indptr[-1] == 0:
        indices = np.empty(0, dtype=np.int64)
        contributions = np.empty(0, dtype=np.float64)
    else:
        indices = np.concatenate(index_rows).astype(np.int64, copy=False)
        contributions = np.concatenate(contribution_rows).astype(np.float64, copy=False)

    return (
        np.ascontiguousarray(indptr, dtype=np.int64),
        np.ascontiguousarray(indices, dtype=np.int64),
        np.ascontiguousarray(contributions, dtype=np.float64),
    )


def _pack_neighborhood(neighborhood: list[Neighborhood]) -> _PackedNeighborhood:
    cached = _PACKED_NEIGHBORHOOD_CACHE.get(id(neighborhood))
    if cached is not None:
        return cached

    positive_rows = [
        np.ascontiguousarray(row.positive_spins, dtype=np.int64) for row in neighborhood
    ]
    positive_contribution_rows = [
        np.ascontiguousarray(row.positive_contributions, dtype=np.float64)
        for row in neighborhood
    ]
    negative_rows = [
        np.ascontiguousarray(row.negative_spins, dtype=np.int64) for row in neighborhood
    ]
    negative_contribution_rows = [
        np.ascontiguousarray(row.negative_contributions, dtype=np.float64)
        for row in neighborhood
    ]

    positive_indptr, positive_spins, positive_contributions = _pack_rows(
        positive_rows,
        positive_contribution_rows,
    )
    negative_indptr, negative_spins, negative_contributions = _pack_rows(
        negative_rows,
        negative_contribution_rows,
    )

    packed = _PackedNeighborhood(
        positive_indptr=positive_indptr,
        positive_spins=positive_spins,
        positive_contributions=positive_contributions,
        negative_indptr=negative_indptr,
        negative_spins=negative_spins,
        negative_contributions=negative_contributions,
        spin_potential=np.ascontiguousarray(
            np.asarray([row.spin_potential for row in neighborhood], dtype=np.float64)
        ),
    )
    _PACKED_NEIGHBORHOOD_CACHE[id(neighborhood)] = packed
    return packed


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _inverse_recompute_energy_grid(
        current_system,
        positive_indptr,
        positive_spins,
        positive_contributions,
        negative_indptr,
        negative_spins,
        negative_contributions,
        spin_potential,
    ):
        n_spins = current_system.size
        energy_grid = np.empty(n_spins, dtype=np.float64)
        total_energy = 0.0

        for spin_idx in range(n_spins):
            spin_state = current_system[spin_idx]
            row_energy = 0.0

            for edge in range(positive_indptr[spin_idx], positive_indptr[spin_idx + 1]):
                neighbor_idx = positive_spins[edge]
                if spin_state == current_system[neighbor_idx]:
                    row_energy += positive_contributions[edge]

            for edge in range(negative_indptr[spin_idx], negative_indptr[spin_idx + 1]):
                neighbor_idx = negative_spins[edge]
                if spin_state != current_system[neighbor_idx]:
                    row_energy += negative_contributions[edge]

            energy_grid[spin_idx] = spin_potential[spin_idx] - 2.0 * row_energy
            total_energy += row_energy

        return energy_grid, 0.5 * total_energy


    @njit(cache=True)
    def _independent_recompute_energy_grid(
        current_system,
        positive_indptr,
        positive_spins,
        positive_contributions,
        negative_indptr,
        negative_spins,
        negative_contributions,
        spin_potential,
    ):
        n_spins = current_system.shape[0]
        energy_grid = np.empty((n_spins, 2), dtype=np.float64)
        total_energy = 0.0

        for spin_idx in range(n_spins):
            spin_a = current_system[spin_idx, 0]
            spin_b = current_system[spin_idx, 1]
            anti_a = not spin_a
            anti_b = not spin_b
            row_energy_a = 0.0
            row_energy_b = 0.0

            for edge in range(positive_indptr[spin_idx], positive_indptr[spin_idx + 1]):
                neighbor_idx = positive_spins[edge]
                contribution = positive_contributions[edge]
                if anti_a == current_system[neighbor_idx, 1]:
                    row_energy_a += contribution
                if anti_b == current_system[neighbor_idx, 0]:
                    row_energy_b += contribution

            for edge in range(negative_indptr[spin_idx], negative_indptr[spin_idx + 1]):
                neighbor_idx = negative_spins[edge]
                contribution = negative_contributions[edge]
                if anti_a != current_system[neighbor_idx, 1]:
                    row_energy_a += contribution
                if anti_b != current_system[neighbor_idx, 0]:
                    row_energy_b += contribution

            energy_grid[spin_idx, 0] = 0.5 * (spin_potential[spin_idx] - 2.0 * row_energy_a)
            energy_grid[spin_idx, 1] = 0.5 * (spin_potential[spin_idx] - 2.0 * row_energy_b)
            total_energy += row_energy_a

        return energy_grid, 0.5 * total_energy


    @njit(cache=True)
    def _inverse_metropolis_sweep(
        current_system,
        energy_grid,
        positive_indptr,
        positive_spins,
        positive_contributions,
        negative_indptr,
        negative_spins,
        negative_contributions,
        temperature,
        n_steps,
        current_energy,
        best_system,
        best_energy,
        track_best,
        random_seed,
    ):
        np.random.seed(random_seed)
        accepted = 0

        for _ in range(n_steps):
            spin_idx = np.random.randint(current_system.size)
            dE = energy_grid[spin_idx]

            if dE < 0.0:
                if temperature == 0.0 or np.random.random() >= np.exp(dE / temperature):
                    continue

            accepted += 1
            current_system[spin_idx] = not current_system[spin_idx]
            spin_state = current_system[spin_idx]
            energy_grid[spin_idx] = -energy_grid[spin_idx]

            for edge in range(positive_indptr[spin_idx], positive_indptr[spin_idx + 1]):
                neighbor_idx = positive_spins[edge]
                contribution = positive_contributions[edge]
                if spin_state == current_system[neighbor_idx]:
                    energy_grid[neighbor_idx] -= 2.0 * contribution
                else:
                    energy_grid[neighbor_idx] += 2.0 * contribution

            for edge in range(negative_indptr[spin_idx], negative_indptr[spin_idx + 1]):
                neighbor_idx = negative_spins[edge]
                contribution = negative_contributions[edge]
                if spin_state != current_system[neighbor_idx]:
                    energy_grid[neighbor_idx] -= 2.0 * contribution
                else:
                    energy_grid[neighbor_idx] += 2.0 * contribution

            current_energy += dE
            if track_best and current_energy > best_energy:
                best_energy = current_energy
                best_system[:] = current_system

        return accepted / n_steps, current_energy, best_energy


    @njit(cache=True)
    def _independent_metropolis_sweep(
        current_system,
        energy_grid,
        positive_indptr,
        positive_spins,
        positive_contributions,
        negative_indptr,
        negative_spins,
        negative_contributions,
        temperature,
        n_steps,
        current_energy,
        best_system,
        best_energy,
        track_best,
        random_seed,
    ):
        np.random.seed(random_seed)
        accepted = 0

        for _ in range(n_steps):
            spin_idx = np.random.randint(current_system.shape[0])
            system_idx = np.random.randint(2)
            dE = energy_grid[spin_idx, system_idx]

            if dE < 0.0:
                if temperature == 0.0 or np.random.random() >= np.exp(dE / temperature):
                    continue

            accepted += 1
            spin_state = current_system[spin_idx, system_idx]
            other_system = (system_idx + 1) % 2
            current_system[spin_idx, system_idx] = not current_system[spin_idx, system_idx]
            energy_grid[spin_idx, system_idx] = -energy_grid[spin_idx, system_idx]

            for edge in range(positive_indptr[spin_idx], positive_indptr[spin_idx + 1]):
                neighbor_idx = positive_spins[edge]
                contribution = positive_contributions[edge]
                if spin_state == current_system[neighbor_idx, other_system]:
                    energy_grid[neighbor_idx, other_system] -= contribution
                else:
                    energy_grid[neighbor_idx, other_system] += contribution

            for edge in range(negative_indptr[spin_idx], negative_indptr[spin_idx + 1]):
                neighbor_idx = negative_spins[edge]
                contribution = negative_contributions[edge]
                if spin_state != current_system[neighbor_idx, other_system]:
                    energy_grid[neighbor_idx, other_system] -= contribution
                else:
                    energy_grid[neighbor_idx, other_system] += contribution

            current_energy += dE
            if track_best and current_energy > best_energy:
                best_energy = current_energy
                best_system[:, :] = current_system

        return accepted / n_steps, current_energy, best_energy


else:
    _inverse_recompute_energy_grid = None
    _independent_recompute_energy_grid = None
    _inverse_metropolis_sweep = None
    _independent_metropolis_sweep = None


def _get_packed_neighborhood(neighborhood: Any) -> _PackedNeighborhood | None:
    if not NUMBA_AVAILABLE:
        return None
    if not isinstance(neighborhood, list) or not neighborhood:
        return None
    if not isinstance(neighborhood[0], Neighborhood):
        return None
    return _pack_neighborhood(neighborhood)


def compute_spinflip_numba_energy_grid(
    system_states: np.ndarray,
    neighborhood: Any,
) -> tuple[np.ndarray, float] | None:
    packed = _get_packed_neighborhood(neighborhood)
    if packed is None:
        return None

    if system_states.ndim == 1:
        return _inverse_recompute_energy_grid(
            system_states,
            packed.positive_indptr,
            packed.positive_spins,
            packed.positive_contributions,
            packed.negative_indptr,
            packed.negative_spins,
            packed.negative_contributions,
            packed.spin_potential,
        )

    if system_states.ndim == 2 and system_states.shape[1] == 2:
        return _independent_recompute_energy_grid(
            system_states,
            packed.positive_indptr,
            packed.positive_spins,
            packed.positive_contributions,
            packed.negative_indptr,
            packed.negative_spins,
            packed.negative_contributions,
            packed.spin_potential,
        )

    return None


def run_spinflip_numba_sweep(
    system: Any,
    energy_array: np.ndarray,
    T: float,
    n_steps: int,
    track_best: bool,
) -> float | None:
    if not NUMBA_AVAILABLE:
        return None
    if T < 0:
        raise ValueError(f"Temperature must be nonnegative, got {T!r}.")
    if system.current_system is None or system.best_system is None:
        return None

    packed = _get_packed_neighborhood(system.neighborhood)
    if packed is None:
        return None

    random_seed = int(np.random.randint(0, np.iinfo(np.int32).max))
    if system.current_system.ndim == 1 and energy_array.ndim == 1:
        acceptance, current_energy, best_energy = _inverse_metropolis_sweep(
            system.current_system,
            energy_array,
            packed.positive_indptr,
            packed.positive_spins,
            packed.positive_contributions,
            packed.negative_indptr,
            packed.negative_spins,
            packed.negative_contributions,
            float(T),
            int(n_steps),
            float(system.current_energy),
            system.best_system,
            float(system.best_energy),
            bool(track_best),
            random_seed,
        )
    elif (
        system.current_system.ndim == 2
        and system.current_system.shape[1] == 2
        and energy_array.ndim == 2
        and energy_array.shape[1] == 2
    ):
        acceptance, current_energy, best_energy = _independent_metropolis_sweep(
            system.current_system,
            energy_array,
            packed.positive_indptr,
            packed.positive_spins,
            packed.positive_contributions,
            packed.negative_indptr,
            packed.negative_spins,
            packed.negative_contributions,
            float(T),
            int(n_steps),
            float(system.current_energy),
            system.best_system,
            float(system.best_energy),
            bool(track_best),
            random_seed,
        )
    else:
        return None

    system.current_energy = float(current_energy)
    if track_best:
        system.best_energy = float(best_energy)
    return float(acceptance)
