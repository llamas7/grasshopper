from __future__ import annotations
from .engines import get_engine
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING
from .core import ConstraintModel, Geometry
from .cogs import cogwheel_interpd
from .plot_system import convert_ss
from functools import partial

if TYPE_CHECKING:
    from .system import System


def sync_occupancy(system: System):
    if system.current_system is None:
        raise RuntimeError("Cannot build occupancy lists before initialization.")
    system.sync_occupancy_from_spins()


def init_system(system: System):
    engine = get_engine(system.config)
    calculate_energy = engine.energy
    system.current_energy = calculate_energy(system.current_system, system.neighborhood)
    system.best_energy = calculate_energy(system.best_system, system.neighborhood)

    if system.current_energy > system.best_energy:
        system.best_system = system.current_system.copy()
        system.best_energy = system.current_energy
    if not system.antipodal:
        sync_occupancy(system)


def init_random(system: System):
    if system.constraint_model == ConstraintModel.FIXED_DENSITY:
        all_spins = np.arange(len(system.grid))
        idx = np.random.choice(len(all_spins), size=system.num_spins, replace=False)
        occupied_sites = all_spins[idx].tolist()
        system.current_system = np.zeros(len(system.grid), dtype=bool)
        system.current_system[occupied_sites] = True
    else:
        system.current_system = np.random.randint(
            2, size=system.system_shape, dtype="bool"
        )

    system.best_system = system.current_system.copy()
    init_system(system)


def init_random_hemisphere(system: System, seed=None):
    if system.geometry == Geometry.EUCLIDEAN:
        raise ValueError("hemisphere_init() requires a spherical grid.")
    np.random.seed(seed)
    dimension = int(system.grid.shape[1])
    norm_vector = np.random.uniform(-1, 1, dimension)
    while np.linalg.norm(norm_vector) >= 1:
        norm_vector = np.random.uniform(-1, 1, dimension)
    norm_vector = norm_vector / np.linalg.norm(norm_vector)
    current_system = (system.grid[: len(system.neighborhood)] @ norm_vector) > 0
    if system.constraint_model == ConstraintModel.INDEPENDENT:
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
    if system.constraint_model == ConstraintModel.ANTIPODAL:
        if n_sites % 2 != 0:
            raise ValueError("Antipodal systems require an even number of sites.")
        expected_len = n_sites // 2
        if not system.antipodal:
            raise ValueError("Antipodal system expects antipodal=True, but got False.")
        allowed_shapes = [(expected_len,), (expected_len, 1)]

    elif system.constraint_model == ConstraintModel.FIXED_DENSITY:
        expected_len = n_sites
        if system.antipodal:
            raise ValueError(
                f"{system.system_type.value} system expects antipodal=False, but got True."
            )
        allowed_shapes = [(expected_len,), (expected_len, 1)]

    elif system.constraint_model == ConstraintModel.INDEPENDENT:
        if n_sites % 2 != 0:
            raise ValueError("Independent antipodal systems require an even number of sites.")
        expected_len = n_sites // 2
        allowed_shapes = [(expected_len, 2)]

    else:
        raise ValueError(f"Unknown system config: {system.config!r}")

    shape = spin_array.shape

    # Strict shape check + targeted normalization
    if system.constraint_model in (
        ConstraintModel.ANTIPODAL,
        ConstraintModel.FIXED_DENSITY,
    ):
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

    if system.geometry == Geometry.EUCLIDEAN:
        occupied = int(np.count_nonzero(normalized))
        if occupied != system.num_spins:
            raise ValueError(
                f"Euclidean spin_array has {occupied} occupied sites, "
                f"but system.num_spins is {system.num_spins}."
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
    if system.geometry == Geometry.EUCLIDEAN:
        raise ValueError("Cogwheel initializers require a spherical grid.")
    if system.antipodal and num_cogs % 2 == 0:
        raise ValueError("Antipodal cogwheels require an odd number of cogs.")
    interped = cogwheel_interpd(num_cogs, cog_height)
    grid_ss = convert_ss(system.grid)

    if system.antipodal:
        system_size = len(grid_ss) // 2
    else:
        system_size = len(grid_ss)
    spin_array = (grid_ss[:, 1] > interped(grid_ss[:, 0]))[:system_size]

    if system.constraint_model == ConstraintModel.INDEPENDENT:
        shifted_x = (grid_ss[:, 0] + np.pi / num_cogs + np.pi) % (2 * np.pi) - np.pi
        second_array = (grid_ss[:, 1] < interped(shifted_x))[: len(grid_ss) // 2]
        spin_array = np.column_stack((spin_array, second_array))
    system.current_system = spin_array.copy()
    system.best_system = spin_array.copy()
    init_system(system)


def init_cosine_cogwheel(
        system: System,
        num_cogs: int,
        cog_height: float,
) -> None:
    if system.geometry == Geometry.EUCLIDEAN:
        raise ValueError("Cogwheel initializers require a spherical grid.")
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
    if system.constraint_model == ConstraintModel.INDEPENDENT:
        second_system = (
                                grid_ss[:, 1] < cogwheel(grid_ss[:, 0] + (np.pi / num_cogs)) + np.pi / 2
                        )[: len(grid_ss) // 2]
        spin_array = np.column_stack((spin_array, second_system))
    system.current_system = spin_array.copy()
    system.best_system = spin_array.copy()
    init_system(system)
