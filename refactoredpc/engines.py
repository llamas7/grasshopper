from __future__ import annotations
from numpy.typing import NDArray
import numpy as np
from .core import Engine, SystemType
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
    calculate_non_antipodal_move_delta_energy,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import System


def inv_suggest(system: System, energy_array: NDArray[np.float64]):
    suggested_update = np.random.randint(system.num_spins)
    energy_delta = energy_array[suggested_update]
    return suggested_update, energy_delta


def inv_best_update(system: System, energy_array: NDArray[np.float64]):
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
    if not system.up_spins or not system.down_spins:
        raise ValueError(
            "Non-antipodal moves require at least one occupied and one "
            "unoccupied site."
        )
    suggested_spin = np.random.randint(len(system.up_spins))
    suggested_move = np.random.randint(len(system.down_spins))
    suggested_update = (suggested_spin, suggested_move)
    energy_delta = calculate_non_antipodal_move_delta_energy(
        system, suggested_update, energy_array
    )
    return suggested_update, energy_delta


def na_best_update(system: System, energy_array: NDArray[np.float64]):
    if system.up_spins is None or system.down_spins is None:
        raise RuntimeError("Non-antipodal spin lists have not been initialized.")
    if not system.up_spins or not system.down_spins:
        return None, float("-inf")

    up_site_indices = np.asarray(system.up_spins, dtype=np.intp)
    down_site_indices = np.asarray(system.down_spins, dtype=np.intp)
    up_idx = int(np.argmin(energy_array[up_site_indices]))
    down_idx = int(np.argmax(energy_array[down_site_indices]))
    suggested_update = (up_idx, down_idx)
    energy_delta = calculate_non_antipodal_move_delta_energy(
        system,
        suggested_update,
        energy_array,
    )
    return suggested_update, float(energy_delta)


INVERSE_ENGINE = Engine(
    mode="inverse",
    energy=calculate_system_energy,
    grid_gen=generate_energy_grid,
    update=update_energy_grid,
    suggest=inv_suggest,
    best_update=inv_best_update,
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
    energy=calculate_non_antipodal_system_energy,
    grid_gen=generate_non_antipodal_energy_grid,
    update=update_non_antipodal_energy_grid,
    suggest=na_suggest,
    best_update=na_best_update,
)

NONANTIPODAL_ENGINE = Engine(
    mode="non-antipodal",
    energy=OCCUPANCY_ENGINE.energy,
    grid_gen=OCCUPANCY_ENGINE.grid_gen,
    update=OCCUPANCY_ENGINE.update,
    suggest=OCCUPANCY_ENGINE.suggest,
    best_update=OCCUPANCY_ENGINE.best_update,
)

EUCLIDEAN_ENGINE = Engine(
    mode="euclidean",
    energy=OCCUPANCY_ENGINE.energy,
    grid_gen=OCCUPANCY_ENGINE.grid_gen,
    update=OCCUPANCY_ENGINE.update,
    suggest=OCCUPANCY_ENGINE.suggest,
    best_update=OCCUPANCY_ENGINE.best_update,
)

ENGINE_BY_TYPE = {
    SystemType.INVERSE: INVERSE_ENGINE,
    SystemType.INDEPENDENT: INDEPENDENT_ENGINE,
    SystemType.NON_ANTIPODAL: NONANTIPODAL_ENGINE,
    SystemType.EUCLIDEAN: EUCLIDEAN_ENGINE,
}

ENGINES = {system_type.value: engine for system_type, engine in ENGINE_BY_TYPE.items()}


def get_engine(system_type: SystemType | str) -> Engine:
    if isinstance(system_type, str):
        system_type = SystemType(system_type.lower().replace("_", "-"))
    return ENGINE_BY_TYPE[system_type]
