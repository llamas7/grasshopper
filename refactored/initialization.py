from __future__ import annotations
from .engines import INVERSE_ENGINE, INDEPENDENT_ENGINE, NONANTIPODAL_ENGINE
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING
from .core import SystemType
from .cogs import cogwheel_interpd
from .plot_system import convert_ss
from functools import partial

if TYPE_CHECKING:
    from .system import System

ENGINES = {
    "inverse": INVERSE_ENGINE,
    "independent": INDEPENDENT_ENGINE,
    "non-antipodal": NONANTIPODAL_ENGINE,
}


def build_spin_lists(system: System):
    all_spins = np.arange(len(system.grid))
    up_spins = all_spins[system.current_system].tolist()
    down_spins = np.delete(all_spins, up_spins).tolist()
    system.up_spins = up_spins
    system.down_spins = down_spins


def init_system(system: System):
    engine = ENGINES[system.system_type.value]
    calculate_energy = engine["energy"]
    system.current_energy = calculate_energy(system.current_system, system.neighborhood)
    system.best_energy = calculate_energy(system.best_system, system.neighborhood)

    if system.current_energy > system.best_energy:
        system.best_system = system.current_system.copy()
        system.best_energy = system.current_energy
    if not system.antipodal:
        build_spin_lists(system)


def init_random(system: System):
    if system.system_type.value != "non-antipodal":
        system.current_system = np.random.randint(
            2, size=system.system_shape, dtype="bool"
        )
    else:
        all_spins = np.arange(len(system.grid))
        idx = np.random.choice(len(all_spins), size=system.num_spins, replace=False)
        up_spins = all_spins[idx].tolist()
        system.current_system = np.zeros(len(system.grid), dtype=bool)
        system.current_system[up_spins] = True

    system.best_system = system.current_system.copy()
    init_system(system)


def init_random_hemisphere(system: System, seed=None):
    np.random.seed(seed)
    norm_vector = np.random.uniform(-1, 1, 3)
    while np.linalg.norm(norm_vector) >= 1:
        norm_vector = np.random.uniform(-1, 1, 3)
    norm_vector = norm_vector / np.linalg.norm(norm_vector)
    current_system = (system.grid[: len(system.neighborhood)] @ norm_vector) > 0
    if system.system_type.value == "independent":
        current_system = np.column_stack((current_system, ~current_system))
    system.current_system = current_system.copy()
    system.best_system = current_system.copy()
    init_system(system)


def init_from_array(system: System, spin_array: NDArray[np.bool_]) -> None:
    n_sites = int(system.grid.shape[0])

    # dtype guard
    if spin_array.dtype != np.bool_:
        raise TypeError(f"spin_array must be boolean, got {spin_array.dtype!r}")

    # expected shape per system type
    if system.system_type == SystemType.INVERSE:
        if n_sites % 2 != 0:
            raise ValueError("INVERSE system requires an even number of sites.")
        expected_len = n_sites // 2
        if not system.antipodal:
            raise ValueError("INVERSE system expects antipodal=True, but got False.")
        allowed_shapes = [(expected_len,), (expected_len, 1)]

    elif system.system_type == SystemType.NON_ANTIPODAL:
        expected_len = n_sites
        if system.antipodal:
            raise ValueError(
                "NON_ANTIPODAL system expects antipodal=False, but got True."
            )
        allowed_shapes = [(expected_len,), (expected_len, 1)]

    elif system.system_type == SystemType.INDEPENDENT:
        if n_sites % 2 != 0:
            raise ValueError("INDEPENDENT system requires an even number of sites.")
        expected_len = n_sites // 2
        allowed_shapes = [(expected_len, 2)]

    else:
        raise ValueError(f"Unknown system type: {system.system_type!r}")

    shape = spin_array.shape

    # Strict shape check + targeted normalization
    if system.system_type in (SystemType.INVERSE, SystemType.NON_ANTIPODAL):
        if shape == (expected_len,):
            normalized = spin_array
        elif shape == (expected_len, 1):
            normalized = spin_array.reshape(expected_len)
        else:
            raise ValueError(
                f"Input array shape {shape} does not match allowed shapes "
                f"{allowed_shapes} for system type {system.system_type.value}."
            )

    else:  # INDEPENDENT
        if shape == (expected_len, 2):
            normalized = spin_array
        else:
            raise ValueError(
                f"Input array shape {shape} does not match allowed shape "
                f"{allowed_shapes} for system type {system.system_type.value}."
            )

    # Store exactly once; ensure no aliasing and contiguous memory
    system.current_system = np.ascontiguousarray(normalized, dtype=np.bool_).copy()
    system.best_system = system.current_system.copy()
    init_system(system)


def init_triangle_cogwheel(
        system: System,
        num_cogs: int,
        cog_height: float,
) -> None:
    if system.antipodal and num_cogs % 2 == 0:
        raise ValueError("Antipodal cogwheels require an odd number of cogs.")
    interped = cogwheel_interpd(num_cogs, cog_height)
    grid_ss = convert_ss(system.grid)

    if system.antipodal:
        system_size = len(grid_ss) // 2
    else:
        system_size = len(grid_ss)
    spin_array = (grid_ss[:, 1] > interped(grid_ss[:, 0]))[:system_size]

    if system.system_type.value == "independent":
        shifted_x = (grid_ss[:, 0] + np.pi / num_cogs + np.pi) % (2 * np.pi) - np.pi
        second_array = (grid_ss[:, 1] < interped(shifted_x))[: len(grid_ss) // 2]
        spin_array = np.column_stack((system, second_array))
    system.current_system = spin_array.copy()
    system.best_system = spin_array.copy()
    init_system(system)


def init_cosine_cogwheel(
        system: System,
        num_cogs: int,
        cog_height: float,
) -> None:
    if system.antipodal and num_cogs % 2 == 0:
        raise ValueError("Antipodal cogwheels require an odd number of cogs.")

    def cos_perturbation(x, amplitude=0.1, frequency=3):
        return amplitude * np.cos(frequency * x)

    grid_ss = convert_ss(system.grid)
    cogwheel = partial(cos_perturbation, amplitude=cog_height, frequency=num_cogs)
    if system.antipodal:
        system_size = len(grid_ss) // 2
    else:
        system_size = len(grid_ss)

    spin_array = (grid_ss[:, 1] > cogwheel(grid_ss[:, 0]) + np.pi / 2)[:system_size]
    if system.system_type.value == "independent":
        second_system = (
                                grid_ss[:, 1] < cogwheel(grid_ss[:, 0] + (np.pi / num_cogs)) + np.pi / 2
                        )[: len(grid_ss) // 2]
        spin_array = np.column_stack((spin_array, second_system))
    system.current_system = spin_array.copy()
    system.best_system = spin_array.copy()
    init_system(system)
