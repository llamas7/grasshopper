from __future__ import annotations
from numpy.typing import NDArray
import numpy as np
from .inverse_energy import (
    generate_energy_grid,
    update_energy_grid,
    calculate_system_energy,
)
from .independent_energy import (
    generate_independent_energy_grid,
    update_independent_energy_grid,
    calculate_independent_system_energy,
)
from .non_antipodal_energy import (
    generate_non_antipodal_energy_grid,
    update_non_antipodal_energy_grid,
    calculate_non_antipodal_system_energy,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import System


def inv_suggest(system: System, energy_array: NDArray[np.float64]):
    suggested_update = np.random.randint(system.num_spins)
    energy_delta = energy_array[suggested_update]
    return suggested_update, energy_delta


def ind_suggest(system: System, energy_array: NDArray[np.float64]):
    suggested_spin = np.random.randint(system.num_spins)
    suggested_system = np.random.randint(2)
    suggested_update = (suggested_spin, suggested_system)
    energy_delta = energy_array[suggested_update]
    return suggested_update, energy_delta


def na_suggest(system: System, energy_array: NDArray[np.float64]):
    suggested_spin = np.random.randint(len(system.up_spins))
    suggested_move = np.random.randint(len(system.down_spins))
    suggested_update = (suggested_spin, suggested_move)
    energy_delta = (
        energy_array[system.down_spins[suggested_move]]
        - energy_array[system.up_spins[suggested_spin]]
    )
    return suggested_update, energy_delta


INVERSE_ENGINE = {
    "mode": "inverse",
    "energy": calculate_system_energy,
    "grid_gen": generate_energy_grid,
    "update": update_energy_grid,
    "suggest": inv_suggest,
}
INDEPENDENT_ENGINE = {
    "mode": "independent",
    "energy": calculate_independent_system_energy,
    "grid_gen": generate_independent_energy_grid,
    "update": update_independent_energy_grid,
    "suggest": ind_suggest,
}
NONANTIPODAL_ENGINE = {
    "mode": "non-antipodal",
    "energy": calculate_non_antipodal_system_energy,
    "grid_gen": generate_non_antipodal_energy_grid,
    "update": update_non_antipodal_energy_grid,
    "suggest": na_suggest,
}
