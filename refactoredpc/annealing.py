from __future__ import annotations
import time
import numpy as np
from .core import Engine
from .engines import get_engine

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import System


def acceptance_probability(dE: float, T: float) -> bool:
    """Metropolis acceptance test at temperature T."""
    if dE >= 0:
        return True
    if T < 0:
        raise ValueError(f"Temperature must be nonnegative, got {T!r}.")
    if T == 0:
        return False
    return np.random.rand() < np.exp(dE / T)


def _require_annealable(system: "System") -> None:
    if system.current_system is None:
        raise RuntimeError("System has no current spin state. Initialize it first.")
    if system.best_system is None:
        raise RuntimeError("System has no best spin state. Initialize it first.")
    if system.neighborhood is None:
        raise RuntimeError("System has no neighborhood. Call compute_neighborhood() first.")
    if system.anneal_data is None:
        raise RuntimeError("System has no anneal_data.")


def _validate_schedule(system: "System", save: bool = False) -> None:
    params = system.parameters
    data = system.anneal_data
    if params.cooling_rate <= 0 or params.cooling_rate >= 1:
        raise ValueError(
            "parameters.cooling_rate must be between 0 and 1 for simulated annealing."
        )
    if params.max_iterations < 0:
        raise ValueError("parameters.max_iterations must be nonnegative.")
    if params.max_runtime < 0:
        raise ValueError("parameters.max_runtime must be nonnegative.")
    if data.steps <= 0:
        raise ValueError(f"anneal_data.steps must be positive, got {data.steps!r}.")
    if save and params.its_per_save <= 0:
        raise ValueError("parameters.its_per_save must be positive when save=True.")


def _recompute_energy_state(system: "System", engine: Engine) -> np.ndarray:
    """Rebuild the incremental energy grid and exact scalar energies."""
    grid_gen = engine.grid_gen
    system_energy_fn = engine.energy
    energy_array = grid_gen(system.current_system, system.neighborhood)
    system.current_energy = system_energy_fn(system.current_system, system.neighborhood)
    system.best_energy = system_energy_fn(system.best_system, system.neighborhood)
    if system.current_energy > system.best_energy:
        system.best_energy = system.current_energy
        system.best_system = system.current_system.copy()
    return energy_array


def metropolis_sweep(
        system: "System",
        energy_array: np.ndarray,
        T: float,
        n_steps: int,
        engine: Engine,
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
    engine : Engine
        Engine for this system_type.
    track_best : bool, optional
        If True, updates system.best_system / best_energy.

    Returns
    -------
    acceptance : float
        Accepted proposals divided by n_steps.
    """
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps!r}.")

    suggest = engine.suggest
    update = engine.update

    accepted = 0

    for _ in range(n_steps):
        suggested_update, dE = suggest(system, energy_array)

        if not acceptance_probability(dE, T):
            continue

        accepted += 1
        update(system, suggested_update, energy_array)
        system.current_energy += dE
        if track_best and system.current_energy > system.best_energy:
            system.best_energy = system.current_energy
            system.best_system = system.current_system.copy()

    return accepted / n_steps


def thermalize_system(
        system: "System",
        target_acceptance: float = 0.6,
        tolerance: float = 0.05,
        steps_per_temp: int = 1_000,
        max_adjustments: int = 20,
        verbose: bool = False,
        k_p: float = 0.5,  # proportional gain in log(T)
) -> float:
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
    _require_annealable(system)
    if steps_per_temp <= 0:
        raise ValueError(f"steps_per_temp must be positive, got {steps_per_temp!r}.")
    if max_adjustments < 0:
        raise ValueError(f"max_adjustments must be nonnegative, got {max_adjustments!r}.")
    if not 0 <= target_acceptance <= 1:
        raise ValueError(
            f"target_acceptance must be between 0 and 1, got {target_acceptance!r}."
        )
    if tolerance < 0:
        raise ValueError(f"tolerance must be nonnegative, got {tolerance!r}.")

    engine = get_engine(system.system_type)
    grid_gen = engine.grid_gen

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
        system.anneal_data.temperature = float(
            np.clip(T_new, min_temp, max_temp)
        )

    return float(system.anneal_data.temperature)


def simulated_annealing(
        system: "System",
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
    verbose : bool, optional
        Whether to print annealing diagnostics, by default False.
    reinit : bool, optional
        If True, recompute the energy grid and scalar energies after each
        temperature step to correct numerical or incremental-update drift.
    """
    _require_annealable(system)
    _validate_schedule(system, save=save)

    engine = get_engine(system.system_type)
    grid_gen = engine.grid_gen

    # Build initial energy grid consistent with current_system
    energy_array = _recompute_energy_state(system, engine) if reinit else grid_gen(
        system.current_system, system.neighborhood
    )

    start_time = time.time()
    acc = 1.0
    while (
            system.anneal_data.temperature > system.parameters.final_temp
            and system.anneal_data.iterations < system.parameters.max_iterations
            and (time.time() - start_time) / 60 ** 2 < system.parameters.max_runtime
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

        system.anneal_data.steps = max(
            1,
            int(system.anneal_data.steps / system.parameters.cooling_rate),
        )
        system.anneal_data.temperature *= system.parameters.cooling_rate
        system.anneal_data.iterations += 1
        if reinit:
            energy_array = _recompute_energy_state(system, engine)
        system.anneal_data.energies.append(system.current_energy)

        # Periodic save
        if save and (
                system.anneal_data.iterations % system.parameters.its_per_save == 0
        ):
            system.save(system.parameters.save_dir)

    # Final save
    if save:
        system.save(system.parameters.save_dir)
