from __future__ import annotations
from .core import Neighborhood
from .complementary_energy import calculate_spin_energy, calculate_spin_delta_energy
import numpy as np
from numpy.typing import NDArray


def calculate_independent_system_energy(
        system_states: NDArray[np.bool_], neighborhood: list[Neighborhood]
):
    systemA = system_states[:, 0]
    systemB = system_states[:, 1]
    total_energyA = 0
    for i, spin in enumerate(systemA):
        total_energyA += calculate_spin_energy(systemB, not spin, neighborhood[i])
    return total_energyA / 2


def generate_independent_energy_grid(
        system_states: NDArray[np.bool_], neighborhood: list[Neighborhood]
):
    systemA = system_states[:, 0]
    systemB = system_states[:, 1]
    energy_grid = np.zeros((len(system_states), 2))
    for i, neighbors in enumerate(neighborhood):
        energy_grid[i, 0] = calculate_spin_delta_energy(
            systemB, not systemA[i], neighbors
        )
        energy_grid[i, 1] = calculate_spin_delta_energy(
            systemA, not systemB[i], neighbors
        )
    return energy_grid / 2


def update_independent_energy_grid(
        system: System,
        update: tuple(int, int),
        energy_grid: NDArray[np.float64],
):
    system_states = system.current_system
    spin_num, system_num = update
    neighbors = system.neighborhood[spin_num]
    aligned_spins = neighbors.positive_spins
    aligned_contributions = neighbors.positive_contributions
    non_aligned_spins = neighbors.negative_spins
    non_aligned_contributions = neighbors.negative_contributions
    spin_state = system_states[spin_num, system_num]  # Looking at anti correlations
    other_system = (system_num + 1) % 2
    system_states[spin_num, system_num] = not system_states[spin_num, system_num]
    energy_grid[spin_num, system_num] = -energy_grid[spin_num, system_num]
    matching_bool = spin_state == system_states[aligned_spins, other_system]

    energy_grid[aligned_spins, other_system] -= aligned_contributions * (
        matching_bool
    ) - aligned_contributions * np.logical_not(matching_bool)
    opposing_bool = spin_state != system_states[non_aligned_spins, other_system]
    energy_grid[non_aligned_spins, other_system] -= non_aligned_contributions * (
        opposing_bool
    ) - non_aligned_contributions * np.logical_not(opposing_bool)
