from __future__ import annotations

from typing import Any

import numpy as np

from .core import FixedDensityNeighborhoodCSR

try:
    from numba import njit
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    njit = None


NUMBA_AVAILABLE = njit is not None


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _pair_contribution_sorted(
            indptr,
            neighbor_spins,
            contributions,
            source_spin,
            target_spin,
    ):
        start = indptr[source_spin]
        stop = indptr[source_spin + 1]

        left = start
        right = stop
        while left < right:
            mid = (left + right) // 2
            if neighbor_spins[mid] < target_spin:
                left = mid + 1
            else:
                right = mid

        if left >= stop or neighbor_spins[left] != target_spin:
            return 0.0

        total = 0.0
        idx = left
        while idx < stop and neighbor_spins[idx] == target_spin:
            total += contributions[idx]
            idx += 1
        return total


    @njit(cache=True)
    def _fixed_density_metropolis_sweep(
            current_system,
            energy_grid,
            occupied,
            unoccupied,
            indptr,
            neighbor_spins,
            contributions,
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
            up_idx = np.random.randint(occupied.size)
            down_idx = np.random.randint(unoccupied.size)
            spin_up = occupied[up_idx]
            spin_down = unoccupied[down_idx]

            removed_pair_energy = _pair_contribution_sorted(
                indptr,
                neighbor_spins,
                contributions,
                spin_down,
                spin_up,
            )
            dE = energy_grid[spin_down] - removed_pair_energy - energy_grid[spin_up]

            if dE < 0.0:
                if temperature == 0.0 or np.random.random() >= np.exp(dE / temperature):
                    continue

            accepted += 1
            current_system[spin_up] = False
            current_system[spin_down] = True

            for edge in range(indptr[spin_up], indptr[spin_up + 1]):
                energy_grid[neighbor_spins[edge]] -= contributions[edge]

            for edge in range(indptr[spin_down], indptr[spin_down + 1]):
                energy_grid[neighbor_spins[edge]] += contributions[edge]

            occupied[up_idx] = spin_down
            unoccupied[down_idx] = spin_up
            current_energy += dE

            if track_best and current_energy > best_energy:
                best_energy = current_energy
                best_system[:] = current_system

        return accepted / n_steps, current_energy, best_energy


else:
    _fixed_density_metropolis_sweep = None


def run_fixed_density_numba_sweep(
        system: Any,
        energy_array: np.ndarray,
        T: float,
        n_steps: int,
        track_best: bool,
) -> float | None:
    """Run the optional Numba CSR fixed-density sweep, if available."""
    if not NUMBA_AVAILABLE:
        return None
    if T < 0:
        raise ValueError(f"Temperature must be nonnegative, got {T!r}.")
    if system.occupancy is None:
        return None
    if not isinstance(system.neighborhood, FixedDensityNeighborhoodCSR):
        return None
    if system.current_system is None or system.best_system is None:
        return None
    if system.current_system.ndim != 1 or system.best_system.ndim != 1:
        return None
    if energy_array.ndim != 1:
        return None

    random_seed = int(np.random.randint(0, np.iinfo(np.int32).max))
    acceptance, current_energy, best_energy = _fixed_density_metropolis_sweep(
        system.current_system,
        energy_array,
        system.occupancy.occupied,
        system.occupancy.unoccupied,
        system.neighborhood.indptr,
        system.neighborhood.neighbor_spins,
        system.neighborhood.contributions,
        float(T),
        int(n_steps),
        float(system.current_energy),
        system.best_system,
        float(system.best_energy),
        bool(track_best),
        random_seed,
    )
    system.current_energy = float(current_energy)
    if track_best:
        system.best_energy = float(best_energy)
    return float(acceptance)
