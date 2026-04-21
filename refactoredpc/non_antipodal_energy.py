from __future__ import annotations
from .core import NonAntipodalNeighborhood, NonAntipodalNeighborhoodCSR
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import System


def calculate_non_antipodal_system_potential(
        system_states: NDArray[np.bool_],
        neighborhood: list[NonAntipodalNeighborhood] | NonAntipodalNeighborhoodCSR,
):
    if isinstance(neighborhood, NonAntipodalNeighborhoodCSR):
        return float(0.5 * np.sum(neighborhood.spin_potential[system_states]))
    total_potential = 0
    for i, spin_neighbors in enumerate(neighborhood):
        if system_states[i]:
            total_potential += spin_neighbors.spin_potential
    return total_potential / 2


def calculate_non_antipodal_system_probability(
        system_states: NDArray[np.bool_],
        neighborhood: list[NonAntipodalNeighborhood] | NonAntipodalNeighborhoodCSR,
):
    total_potential = calculate_non_antipodal_system_potential(
        system_states, neighborhood
    )
    total_energy = calculate_non_antipodal_system_energy(system_states, neighborhood)
    if total_potential == 0:
        raise ZeroDivisionError(
            "Cannot calculate non-antipodal probability with zero system potential."
        )
    return total_energy / total_potential


# def calculate_non_antipodal_spin_energy(
#     system_states: NDArray[np.bool_], neighbors: NonAntipodalNeighborhood
# ):
#     neighbor_spins = neighbors.neighbor_spins
#     contributions = neighbors.contributions
#     occupied_bool = system_states[neighbor_spins] != 0
#     spin_energy = np.sum(contributions[occupied_bool])
#     return spin_energy


# def calculate_non_antipodal_system_energy(
#     system_states: NDArray[np.bool_], neighborhood: list[NonAntipodalNeighborhood]
# ):
#     total_energy = 0
#     for i in range(len(neighborhood)):
#         if system_states[i]:
#             total_energy += calculate_non_antipodal_spin_energy(
#                 system_states, neighborhood[i]
#             )
#     return total_energy / 2


def calculate_non_antipodal_spin_energy(
        system_states: NDArray[np.bool_],
        neighbors: NonAntipodalNeighborhood,
) -> float:
    idx = neighbors.neighbor_spins  # int indices
    mask = system_states[idx]  # bool mask (no "!= 0")
    # Sum only where mask is True, without allocating contributions[mask]
    return float(neighbors.contributions.sum(where=mask, initial=0.0))


def calculate_non_antipodal_system_energy(
        system_states: NDArray[np.bool_],
        neighborhood: list[NonAntipodalNeighborhood] | NonAntipodalNeighborhoodCSR,
) -> float:
    if isinstance(neighborhood, NonAntipodalNeighborhoodCSR):
        row_sums = _csr_row_energies(system_states, neighborhood)
        return float(0.5 * np.sum(row_sums[system_states[: len(neighborhood)]]))

    # number of spins/sites
    n = len(neighborhood)

    # Build CSR-like structure: flat neighbor indices & contributions + row pointers
    lengths = np.fromiter(
        (len(nb.neighbor_spins) for nb in neighborhood), count=n, dtype=np.int64
    )
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(lengths, out=indptr[1:])

    neighbors_flat = (
        np.concatenate([nb.neighbor_spins for nb in neighborhood])
        if indptr[-1] > 0
        else np.array([], dtype=np.int64)
    )
    contribs_flat = (
        np.concatenate([nb.contributions for nb in neighborhood])
        if indptr[-1] > 0
        else np.array([], dtype=np.float64)
    )

    # Occupancy of all neighbor edges in one shot (no Python loop)
    occupied_flat = system_states[neighbors_flat]  # bool array
    weighted = contribs_flat * occupied_flat  # 0 where neighbor unoccupied

    # Sum per row (per site i). Empty rows are common in sparse Euclidean
    # neighborhoods, and np.add.reduceat cannot start from an empty slice.
    row_sums = np.zeros(n, dtype=np.float64)
    nonempty_rows = lengths > 0
    if indptr[-1] > 0:
        row_sums[nonempty_rows] = np.add.reduceat(
            weighted, indptr[:-1][nonempty_rows]
        )

    # Original code only adds rows where system_states[i] is True, then halves for double counting
    total_energy = 0.5 * np.sum(row_sums[system_states[:n]])
    return float(total_energy)


def _csr_row_energies(
        system_states: NDArray[np.bool_],
        neighborhood: NonAntipodalNeighborhoodCSR,
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


def generate_non_antipodal_energy_grid(
        system_states: NDArray[np.bool_],
        neighborhood: list[NonAntipodalNeighborhood] | NonAntipodalNeighborhoodCSR,
):
    if isinstance(neighborhood, NonAntipodalNeighborhoodCSR):
        return _csr_row_energies(system_states, neighborhood)
    energy_grid = np.zeros(len(system_states))
    for i, spin_neighbors in enumerate(neighborhood):
        energy_grid[i] = calculate_non_antipodal_spin_energy(
            system_states, spin_neighbors
        )
    return energy_grid


def resolve_non_antipodal_update(system: "System", update: tuple[int, int]) -> tuple[int, int]:
    """Return the moving occupied and destination unoccupied site indices."""
    if system.up_spins is None or system.down_spins is None:
        raise RuntimeError("Non-antipodal spin lists have not been initialized.")
    if system.neighborhood is None:
        raise RuntimeError("Non-antipodal neighborhood has not been initialized.")

    up_idx, down_idx = update
    if up_idx < 0 or down_idx < 0:
        raise IndexError(f"Non-antipodal update indices must be nonnegative: {update!r}.")
    try:
        spin_up = system.up_spins[up_idx]
        spin_down = system.down_spins[down_idx]
    except IndexError as exc:
        raise IndexError(
            f"Invalid non-antipodal update {update!r} for "
            f"{len(system.up_spins)} up spins and {len(system.down_spins)} down spins."
        ) from exc

    system_states = system.current_system
    if system_states is None:
        raise RuntimeError("System has no current spin state.")
    if not bool(system_states[spin_up]):
        raise ValueError(f"Expected spin_up={spin_up} to be occupied.")
    if bool(system_states[spin_down]):
        raise ValueError(f"Expected spin_down={spin_down} to be unoccupied.")

    return spin_up, spin_down


def get_non_antipodal_pair_contribution(
        neighborhood: list[NonAntipodalNeighborhood] | NonAntipodalNeighborhoodCSR,
        source_spin: int,
        target_spin: int,
) -> float:
    """Return the source->target contribution only if the two sites interact."""
    neighbors = neighborhood[source_spin]
    matching_indices = np.flatnonzero(neighbors.neighbor_spins == target_spin)
    if matching_indices.size == 0:
        return 0.0
    return float(neighbors.contributions[matching_indices].sum())


def calculate_non_antipodal_move_delta_energy(
        system: System,
        update: tuple[int, int],
        energy_grid: NDArray[np.float64],
) -> float:
    """Energy delta for moving one occupied site to one unoccupied site."""
    spin_up, spin_down = resolve_non_antipodal_update(system, update)

    # energy_grid[spin_down] is computed against the current state, where
    # spin_up is still occupied. If these two sites interact, that interaction
    # will not exist after the swap and must not be counted as added energy.
    removed_pair_energy = get_non_antipodal_pair_contribution(
        system.neighborhood,
        source_spin=spin_down,
        target_spin=spin_up,
    )
    return float(energy_grid[spin_down] - removed_pair_energy - energy_grid[spin_up])


def update_non_antipodal_energy_grid(
        system: System,
        update: tuple[int, int],
        energy_grid: NDArray[np.float64],
):
    up_idx, down_idx = update
    spin_up, spin_down = resolve_non_antipodal_update(system, update)
    system_states = system.current_system
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
    system.up_spins.append(system.down_spins.pop(down_idx))
    system.down_spins.append(system.up_spins.pop(up_idx))
