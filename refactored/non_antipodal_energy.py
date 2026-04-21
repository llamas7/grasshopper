from __future__ import annotations
from .core import NonAntipodalNeighborhood
import numpy as np
from numpy.typing import NDArray


def calculate_non_antipodal_system_potential(
    system_states: NDArray[np.bool_], neighborhood: list[NonAntipodalNeighborhood]
):
    total_potential = 0
    for i, spin_neighbors in enumerate(neighborhood):
        if system_states[i]:
            total_potential += spin_neighbors.spin_potential
    return total_potential / 2


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
    neighborhood: List[NonAntipodalNeighborhood],
) -> float:
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

    # Sum per row (per site i) using reduceat with CSR pointers
    row_sums = np.add.reduceat(weighted, indptr[:-1])  # shape (n,)

    # Original code only adds rows where system_states[i] is True, then halves for double counting
    total_energy = 0.5 * np.sum(row_sums[system_states[:n]])
    return float(total_energy)


def generate_non_antipodal_energy_grid(
    system_states: NDArray[np.bool_], neighborhood: list[NonAntipodalNeighborhood]
):
    energy_grid = np.zeros(len(system_states))
    for i, spin_neighbors in enumerate(neighborhood):
        energy_grid[i] = calculate_non_antipodal_spin_energy(
            system_states, spin_neighbors
        )
    return energy_grid


def update_non_antipodal_energy_grid(
    system: System,
    update: tuple[int, int],
    energy_grid: NDArray[np.bool_],
):
    up_idx, down_idx = update
    spin_up = system.up_spins[up_idx]
    spin_down = system.down_spins[down_idx]
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
