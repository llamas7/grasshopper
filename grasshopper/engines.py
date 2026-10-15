from __future__ import annotations
from numpy.typing import NDArray
import numpy as np
from .core import Engine, SystemKind
from .complementary_energy import (
    generate_energy_grid,
    update_energy_grid,
    calculate_system_energy,
)
from .independent_energy import (
    generate_independent_energy_grid,
    update_independent_energy_grid,
    calculate_independent_system_energy,
)
from .fixed_density_energy import (
    generate_fixed_density_energy_grid,
    update_fixed_density_energy_grid,
    calculate_fixed_density_system_energy,
    calculate_fixed_density_move_delta_energy,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import System


def complementary_suggest(system: System, energy_array: NDArray[np.float64]):
    suggested_update = np.random.randint(system.num_spins)
    energy_delta = energy_array[suggested_update]
    return suggested_update, energy_delta


def complementary_best_update(system: System, energy_array: NDArray[np.float64]):
    if energy_array.size == 0:
        return None, float("-inf")
    suggested_update = int(np.argmax(energy_array))
    energy_delta = float(energy_array[suggested_update])
    return suggested_update, energy_delta


def ind_suggest(system: System, energy_array: NDArray[np.float64]):
    suggested_spin = np.random.randint(system.num_spins)
    suggested_system = np.random.randint(2)
    suggested_update = (suggested_spin, suggested_system)
    energy_delta = energy_array[suggested_update]
    return suggested_update, energy_delta


def ind_best_update(system: System, energy_array: NDArray[np.float64]):
    if energy_array.size == 0:
        return None, float("-inf")
    suggested_update = tuple(
        int(i)
        for i in np.unravel_index(
            int(np.argmax(energy_array)),
            energy_array.shape,
        )
    )
    energy_delta = float(energy_array[suggested_update])
    return suggested_update, energy_delta


def na_suggest(system: System, energy_array: NDArray[np.float64]):
    occupancy = system.occupancy
    if (
            occupancy is None
            or occupancy.occupied.size == 0
            or occupancy.unoccupied.size == 0
    ):
        raise ValueError(
            "Fixed-density moves require at least one occupied and one "
            "unoccupied site."
        )
    suggested_spin = np.random.randint(len(occupancy.occupied))
    suggested_move = np.random.randint(len(occupancy.unoccupied))
    suggested_update = (suggested_spin, suggested_move)
    energy_delta = calculate_fixed_density_move_delta_energy(
        system, suggested_update, energy_array
    )
    return suggested_update, energy_delta


def na_best_update(system: System, energy_array: NDArray[np.float64]):
    occupancy = system.occupancy
    if occupancy is None:
        raise RuntimeError("Fixed-density occupancy has not been initialized.")
    if occupancy.occupied.size == 0 or occupancy.unoccupied.size == 0:
        return None, float("-inf")

    up_site_indices = occupancy.occupied
    down_site_indices = occupancy.unoccupied
    up_idx = int(np.argmin(energy_array[up_site_indices]))
    down_idx = int(np.argmax(energy_array[down_site_indices]))
    suggested_update = (up_idx, down_idx)
    energy_delta = calculate_fixed_density_move_delta_energy(
        system,
        suggested_update,
        energy_array,
    )
    return suggested_update, float(energy_delta)


COMPLEMENTARY_ENGINE = Engine(
    mode="complementary",
    energy=calculate_system_energy,
    grid_gen=generate_energy_grid,
    update=update_energy_grid,
    suggest=complementary_suggest,
    best_update=complementary_best_update,
)
INDEPENDENT_ENGINE = Engine(
    mode="independent",
    energy=calculate_independent_system_energy,
    grid_gen=generate_independent_energy_grid,
    update=update_independent_energy_grid,
    suggest=ind_suggest,
    best_update=ind_best_update,
)
OCCUPANCY_ENGINE = Engine(
    mode="occupancy",
    energy=calculate_fixed_density_system_energy,
    grid_gen=generate_fixed_density_energy_grid,
    update=update_fixed_density_energy_grid,
    suggest=na_suggest,
    best_update=na_best_update,
)

ENGINE_BY_KIND = {
    SystemKind.COMPLEMENTARY: COMPLEMENTARY_ENGINE,
    SystemKind.INDEPENDENT: INDEPENDENT_ENGINE,
    SystemKind.FIXED_DENSITY: OCCUPANCY_ENGINE,
    SystemKind.EUCLIDEAN: OCCUPANCY_ENGINE,
}


def get_engine(kind: SystemKind) -> Engine:
    return ENGINE_BY_KIND[SystemKind(kind)]
