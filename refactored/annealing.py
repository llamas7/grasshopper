from __future__ import annotations
import time
import numpy as np
from .core import Parameters, SystemType, NeighborhoodLike
from .inverse_energy import update_energy_grid
from .non_antipodal_energy import update_non_antipodal_energy_grid
from .independent_energy import update_independent_energy_grid
from .engines import INVERSE_ENGINE, INDEPENDENT_ENGINE, NONANTIPODAL_ENGINE

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import System
    from .euclid_system import EuclidSystem

ENGINES = {
    "inverse": INVERSE_ENGINE,
    "independent": INDEPENDENT_ENGINE,
    "non-antipodal": NONANTIPODAL_ENGINE,
}


def acceptance_probability(dE: float, T: float) -> bool:
    """Metropolis acceptance test at temperature T."""
    if dE >= 0:
        return True
    return np.random.rand() < np.exp(dE / max(T, 1e-12))


def metropolis_sweep(
    system: "System | EuclidSystem",
    energy_array: np.ndarray,
    T: float,
    n_steps: int,
    engine: dict,
    track_best: bool = True,
) -> float:
    """Run a Metropolis sweep at fixed temperature.

    Parameters
    ----------
    system : System
        Current system object, modified in-place.
    energy_array : np.ndarray
        Energy grid associated with current_system; modified in-place.
    T : float
        Temperature for this sweep.
    n_steps : int
        Number of MC steps.
    engine : dict
        Engine dict for this system_type with "suggest" and "update".
    track_best : bool, optional
        If True, updates system.best_system / best_energy.

    Returns
    -------
    better : int
        Number of accepted energy-increasing moves (ΔE >= 0).
    worse : int
        Number of accepted energy-decreasing moves (ΔE < 0).
    nothing : int
        Number of rejected proposals.
    acceptance : float
        (better + worse) / n_steps
    """
    suggest = engine["suggest"]
    update = engine["update"]

    accepted = 0

    for _ in range(n_steps):
        suggested_update, dE = suggest(system, energy_array)

        if dE >= 0:
            # Always accept non-worsening moves
            accepted += 1
            update(system, suggested_update, energy_array)
            system.current_energy += dE
            if track_best and system.current_energy > system.best_energy:
                system.best_energy = system.current_energy
                system.best_system = system.current_system.copy()
        else:
            # Worsening move: accept with Metropolis prob
            if acceptance_probability(dE, T):
                accepted += 1
                update(system, suggested_update, energy_array)
                system.current_energy += dE

    acceptance = accepted / max(1, n_steps)
    return acceptance


def thermalize_system(
    system: "System | EuclidSystem",
    target_acceptance: float = 0.6,
    tolerance: float = 0.05,
    steps_per_temp: int = 1_000,
    max_adjustments: int = 20,
    verbose: bool = False,
    k_p: float = 0.5,  # proportional gain in log(T)
) -> None:
    """Thermalize system to reach a target Metropolis acceptance rate.

    Uses short Metropolis sweeps at fixed T and adjusts T using a
    log-space proportional controller until the empirical acceptance rate
    is within `tolerance` of `target_acceptance`.

    Parameters
    ----------
    system : System
        System object (must have system.system_type and system.neighborhood).
        Uses and updates system.anneal_data.temperature.
    target_acceptance : float, optional
        Desired acceptance probability for SA start, by default 0.6.
    tolerance : float, optional
        Allowed deviation |acc - target|, by default 0.05.
    steps_per_temp : int, optional
        MC steps per temperature probe, by default 1000.
    max_adjustments : int, optional
        Maximum temperature adjustments, by default 20.
    verbose : bool, optional
        If True, prints detailed progress, by default False.
    k_p : float, optional
        Proportional gain in log(T) update: T_new = T * exp(-k_p * (acc - target)),
        by default 0.5.

    Returns
    -------
    float
        Final temperature after thermalization.
    """
    engine = ENGINES[system.system_type.value]
    grid_gen = engine["grid_gen"]

    # Ensure energy_array matches current configuration
    energy_array = grid_gen(system.current_system, system.neighborhood)

    # Initialize temperature if unset or invalid
    if system.anneal_data.temperature <= 0:
        system.anneal_data.temperature = 1.0

    # Clamp range for safety
    min_temp = 1e-6
    max_temp = 1e6

    for i in range(max_adjustments):
        T = float(system.anneal_data.temperature)

        # One probe sweep at current T (do NOT track best during thermalization)
        acceptance = metropolis_sweep(
            system=system,
            energy_array=energy_array,
            T=T,
            n_steps=steps_per_temp,
            engine=engine,
            track_best=False,
        )

        if verbose:
            print(
                f"[thermalize] iter={i}, T={T:.4g}, "
                f"acceptance={acceptance:.4f}, target={target_acceptance:.4f}, "
                f"steps={steps_per_temp}"
            )

        # Check if we are within the target band
        if abs(acceptance - target_acceptance) <= tolerance:
            break

        # Log-space proportional update:
        #   log T_{new} = log T_{old} - k_p * (acc - target)
        # so if acc > target, T goes down; if acc < target, T goes up.
        log_T = np.log(max(T, min_temp))
        log_T_new = log_T - k_p * (acceptance - target_acceptance)
        T_new = float(np.exp(log_T_new))

        # Clamp to a sane range
        system.anneal_data.temperature = float(np.clip(T_new, min_temp, max_temp))


def simulated_annealing(
    system: "System | EuclidSystem",
    save: bool = True,
    verbose: bool = False,
    reinit: bool = False,
) -> None:
    """Main simulated annealing loop.

    Uses the current system.anneal_data.temperature as starting T
    (you can call `thermalize_system` beforehand to tune it).

    Parameters
    ----------
    system : System
        System object, mutated in-place.
    save : bool, optional
        Whether to save, by default True.
    """
    engine = ENGINES[system.system_type.value]
    mode = engine["mode"]  # kept for future use if needed
    grid_gen = engine["grid_gen"]
    system_energy_fn = engine["energy"]

    # Build initial energy grid consistent with current_system
    energy_array = grid_gen(system.current_system, system.neighborhood)

    start_time = time.time()
    acc = 1.0
    while (
        system.anneal_data.temperature > system.parameters.final_temp
        and system.anneal_data.iterations < system.parameters.max_iterations
        and (time.time() - start_time) / 60**2 < system.parameters.max_runtime
        and acc > system.parameters.min_acceptance
    ):
        T = system.anneal_data.temperature
        n_steps = system.anneal_data.steps

        acc = metropolis_sweep(
            system=system,
            energy_array=energy_array,
            T=T,
            n_steps=n_steps,
            engine=engine,
            track_best=True,
        )

        # Log stats
        if verbose:
            print(
                f"[anneal] iter={system.anneal_data.iterations}, T={T:.4g}, "
                f"acceptance={acc:.4g}, "
                f"steps={n_steps}"
            )
        system.anneal_data.update_stats.append([T, acc])
        # Cooling schedule

        system.anneal_data.steps = int(
            system.anneal_data.steps / system.parameters.cooling_rate
        )
        system.anneal_data.temperature *= system.parameters.cooling_rate
        system.anneal_data.iterations += 1
        system.anneal_data.energies.append(system.current_energy)

        if reinit:
            # Reinitialize energy to correct for any numerical drift
            energy_array = grid_gen(system.current_system, system.neighborhood)
            system.current_energy = system_energy_fn(
                system.current_system, system.neighborhood
            )
            system.best_energy = system_energy_fn(
                system.best_system, system.neighborhood
            )
        # Periodic save
        if save and (
            system.anneal_data.iterations % system.parameters.its_per_save == 0
        ):
            system.save(system.parameters.save_dir)

    # Final save
    if save:
        system.save(system.parameters.save_dir)
