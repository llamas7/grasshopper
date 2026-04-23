from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .core import Neighborhood


def calculate_spin_energy(
        system_states: NDArray[np.bool_], spin_state: bool, neighbors: Neighborhood
):
    spin_energy = 0
    aligned_spins = neighbors.positive_spins
    aligned_contributions = neighbors.positive_contributions
    non_aligned_spins = neighbors.negative_spins
    non_aligned_contributions = neighbors.negative_contributions
    spin_energy += np.dot(
        spin_state == system_states[aligned_spins], aligned_contributions
    )
    spin_energy += np.dot(
        spin_state != system_states[non_aligned_spins], non_aligned_contributions
    )
    return spin_energy


def calculate_system_energy(
        system_states: NDArray[np.bool_], neighborhood: list[Neighborhood]
):
    system_energy = 0
    for i in range(len(neighborhood)):
        system_energy += calculate_spin_energy(
            system_states, system_states[i], neighborhood[i]
        )

    return system_energy / 2


def calculate_spin_delta_energy(
        system_states: NDArray[np.bool_], spin_state: bool, neighbors: Neighborhood
):
    current_energy = 0
    aligned_spins = neighbors.positive_spins
    aligned_contributions = neighbors.positive_contributions
    non_aligned_spins = neighbors.negative_spins
    non_aligned_contributions = neighbors.negative_contributions
    spin_potential = neighbors.spin_potential
    current_energy += np.dot(
        spin_state == system_states[aligned_spins], aligned_contributions
    )
    current_energy += np.dot(
        spin_state != system_states[non_aligned_spins], non_aligned_contributions
    )
    return spin_potential - 2 * current_energy


def generate_energy_grid(
        system_states: NDArray[np.bool_], neighborhood: list[Neighborhood]
):
    energy_grid = np.zeros(len(system_states))
    for i, neighbors in enumerate(neighborhood):
        energy_grid[i] = calculate_spin_delta_energy(
            system_states, system_states[i], neighbors
        )
    return energy_grid


def update_energy_grid(
        system: System, spin_index: int, energy_grid: NDArray[np.float64]
):
    neighbors = system.neighborhood[spin_index]
    system_states = system.current_system
    aligned_spins = neighbors.positive_spins
    aligned_contributions = neighbors.positive_contributions
    non_aligned_spins = neighbors.negative_spins
    non_aligned_contributions = neighbors.negative_contributions

    system_states[spin_index] = not system_states[spin_index]
    spin_state = system_states[spin_index]
    energy_grid[spin_index] = -energy_grid[spin_index]
    matching_bool = spin_state == system_states[aligned_spins]
    # The factor of 2 is because we SUBTRACT energy from current state and ADD energy to antipodal state
    energy_grid[aligned_spins] -= 2 * (
            aligned_contributions * (matching_bool)
            - aligned_contributions * np.logical_not(matching_bool)
    )
    opposing_bool = spin_state != system_states[non_aligned_spins]
    energy_grid[non_aligned_spins] -= 2 * (
            non_aligned_contributions * (opposing_bool)
            - non_aligned_contributions * np.logical_not(opposing_bool)
    )
