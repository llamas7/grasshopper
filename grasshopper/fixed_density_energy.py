from __future__ import annotations
from .core import FixedDensityNeighborhoodCSR
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import System


def calculate_fixed_density_system_energy(
        system_states: NDArray[np.bool_],
        neighborhood: FixedDensityNeighborhoodCSR,
) -> float:
    row_sums = _csr_row_energies(system_states, neighborhood)
    return float(0.5 * np.sum(row_sums[system_states[: len(neighborhood)]]))


def _csr_row_energies(
        system_states: NDArray[np.bool_],
        neighborhood: FixedDensityNeighborhoodCSR,
) -> NDArray[np.float64]:
    n = len(neighborhood)
    lengths = np.diff(neighborhood.indptr)
    row_sums = np.zeros(n, dtype=np.float64)
    if neighborhood.indptr[-1] == 0:
        return row_sums
    occupied_flat = system_states[neighborhood.neighbor_spins]
    weighted = neighborhood.contributions * occupied_flat
    nonempty_rows = lengths > 0
    row_sums[nonempty_rows] = np.add.reduceat(
        weighted,
        neighborhood.indptr[:-1][nonempty_rows],
    )
    return row_sums


def generate_fixed_density_energy_grid(
        system_states: NDArray[np.bool_],
        neighborhood: FixedDensityNeighborhoodCSR,
):
    return _csr_row_energies(system_states, neighborhood)


def resolve_fixed_density_update(system: "System", update: tuple[int, int]) -> tuple[int, int]:
    """Return the moving occupied and destination unoccupied site indices."""
    occupancy = system.occupancy
    if occupancy is None:
        raise RuntimeError("Fixed-density occupancy has not been initialized.")
    if system.neighborhood is None:
        raise RuntimeError("Fixed-density neighborhood has not been initialized.")

    up_idx, down_idx = update
    if up_idx < 0 or down_idx < 0:
        raise IndexError(f"Fixed-density update indices must be nonnegative: {update!r}.")
    try:
        spin_up = occupancy.occupied[up_idx]
        spin_down = occupancy.unoccupied[down_idx]
    except IndexError as exc:
        raise IndexError(
            f"Invalid fixed-density update {update!r} for "
            f"{len(occupancy.occupied)} occupied and "
            f"{len(occupancy.unoccupied)} unoccupied sites."
        ) from exc

    system_states = system.current_spins
    if system_states is None:
        raise RuntimeError("System has no current spin state.")
    if not bool(system_states[spin_up]):
        raise ValueError(f"Expected spin_up={spin_up} to be occupied.")
    if bool(system_states[spin_down]):
        raise ValueError(f"Expected spin_down={spin_down} to be unoccupied.")

    return spin_up, spin_down


def get_fixed_density_pair_contribution(
        neighborhood: FixedDensityNeighborhoodCSR,
        source_spin: int,
        target_spin: int,
) -> float:
    """Return the source->target contribution only if the two sites interact."""
    target_spin = int(target_spin)
    source_spin = int(source_spin)
    if source_spin < 0:
        source_spin += len(neighborhood)
    if source_spin < 0 or source_spin >= len(neighborhood):
        raise IndexError(source_spin)
    start = int(neighborhood.indptr[source_spin])
    stop = int(neighborhood.indptr[source_spin + 1])
    neighbor_spins = neighborhood.neighbor_spins[start:stop]
    if neighbor_spins.size == 0:
        return 0.0
    left = int(np.searchsorted(neighbor_spins, target_spin, side="left"))
    if left == neighbor_spins.size or int(neighbor_spins[left]) != target_spin:
        return 0.0
    right = int(np.searchsorted(neighbor_spins, target_spin, side="right"))
    return float(neighborhood.contributions[start + left:start + right].sum())


def calculate_fixed_density_move_delta_energy(
        system: System,
        update: tuple[int, int],
        energy_grid: NDArray[np.float64],
) -> float:
    """Energy delta for moving one occupied site to one unoccupied site."""
    spin_up, spin_down = resolve_fixed_density_update(system, update)

    # energy_grid[spin_down] is computed against the current state, where
    # spin_up is still occupied. If these two sites interact, that interaction
    # will not exist after the swap and must not be counted as added energy.
    removed_pair_energy = get_fixed_density_pair_contribution(
        system.neighborhood,
        source_spin=spin_down,
        target_spin=spin_up,
    )
    return float(energy_grid[spin_down] - removed_pair_energy - energy_grid[spin_up])


def update_fixed_density_energy_grid(
        system: System,
        update: tuple[int, int],
        energy_grid: NDArray[np.float64],
):
    up_idx, down_idx = update
    spin_up, spin_down = resolve_fixed_density_update(system, update)
    system_states = system.current_spins
    neighborhood = system.neighborhood
    system_states[spin_up] = not system_states[spin_up]
    system_states[spin_down] = not system_states[spin_down]
    (
        up_neighbors,
        up_contributions,
    ) = (
        neighborhood[spin_up].neighbor_spins,
        neighborhood[spin_up].contributions,
    )
    energy_grid[up_neighbors] -= up_contributions
    (
        down_neighbors,
        down_contributions,
    ) = (
        neighborhood[spin_down].neighbor_spins,
        neighborhood[spin_down].contributions,
    )
    energy_grid[down_neighbors] += down_contributions
    occupancy = system.occupancy
    if occupancy is None:
        raise RuntimeError("Fixed-density occupancy has not been initialized.")
    occupancy.occupied[up_idx] = spin_down
    occupancy.unoccupied[down_idx] = spin_up
