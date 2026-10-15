from __future__ import annotations
from pathlib import Path
import time
import numpy as np
from .core import AnnealSchedule, Engine
from .engines import get_engine
from .fixed_density_numba import run_fixed_density_numba_sweep
from .spinflip_numba import (
    compute_spinflip_numba_energy_grid,
    run_spinflip_numba_sweep,
)
from .rng import Seed, restore_numpy_state, seed_numpy

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
    if system.current_spins is None:
        raise RuntimeError("System has no current spin state. Initialize it first.")
    if system.best_spins is None:
        raise RuntimeError("System has no best spin state. Initialize it first.")
    if system.neighborhood is None:
        raise RuntimeError("System has no neighborhood. Call compute_neighborhood() first.")
    if system.anneal_data is None:
        raise RuntimeError("System has no anneal_data.")


def _validate_schedule(
        schedule: AnnealSchedule,
        *,
        save_interval: int,
) -> None:
    schedule.validate()
    if save_interval <= 0:
        raise ValueError(f"save_interval must be positive, got {save_interval!r}.")


def _recompute_energy_state(system: "System", engine: Engine) -> np.ndarray:
    """Rebuild the incremental energy grid and exact scalar energies."""
    accelerated_current = compute_spinflip_numba_energy_grid(
        system.current_spins,
        system.neighborhood,
    )
    accelerated_best = compute_spinflip_numba_energy_grid(
        system.best_spins,
        system.neighborhood,
    )
    if accelerated_current is not None and accelerated_best is not None:
        energy_array, current_energy = accelerated_current
        _, best_energy = accelerated_best
        system.current_energy = float(current_energy)
        system.best_energy = float(best_energy)
        if system.current_energy > system.best_energy:
            system.best_energy = system.current_energy
            system.best_spins = system.current_spins.copy()
        return energy_array

    grid_gen = engine.grid_gen
    system_energy_fn = engine.energy
    energy_array = grid_gen(system.current_spins, system.neighborhood)
    system.current_energy = system_energy_fn(system.current_spins, system.neighborhood)
    system.best_energy = system_energy_fn(system.best_spins, system.neighborhood)
    if system.current_energy > system.best_energy:
        system.best_energy = system.current_energy
        system.best_spins = system.current_spins.copy()
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
        Energy grid associated with current_spins; modified in-place.
    T : float
        Temperature for this sweep.
    n_steps : int
        Number of MC steps.
    engine : Engine
        Engine for this system configuration.
    track_best : bool, optional
        If True, updates system.best_spins / best_energy.

    Returns
    -------
    acceptance : float
        Accepted proposals divided by n_steps.
    """
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps!r}.")

    accelerated_acceptance = run_fixed_density_numba_sweep(
        system=system,
        energy_array=energy_array,
        T=T,
        n_steps=n_steps,
        track_best=track_best,
    )
    if accelerated_acceptance is not None:
        return accelerated_acceptance

    accelerated_acceptance = run_spinflip_numba_sweep(
        system=system,
        energy_array=energy_array,
        T=T,
        n_steps=n_steps,
        track_best=track_best,
    )
    if accelerated_acceptance is not None:
        return accelerated_acceptance

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
            system.best_spins = system.current_spins.copy()

    return accepted / n_steps


def thermalize_system(
        system: "System",
        target_acceptance: float = 0.6,
        tolerance: float = 0.05,
        steps_per_temp: int = 1_000,
        max_adjustments: int = 20,
        verbose: bool = False,
        k_p: float = 0.5,  # proportional gain in log(T)
        seed: Seed = None,
) -> float:
    """Thermalize system to reach a target Metropolis acceptance rate.

    Uses short Metropolis sweeps at fixed T and adjusts T using a
    log-space proportional controller until the empirical acceptance rate
    is within `tolerance` of `target_acceptance`.

    Parameters
    ----------
    system : System
        System object (must have system.kind and system.neighborhood).
        Uses and updates system.temperature.
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

    rng_state = seed_numpy(seed)
    try:
        engine = get_engine(system.kind)
        grid_gen = engine.grid_gen

        # Ensure energy_array matches current configuration
        accelerated_energy = compute_spinflip_numba_energy_grid(
            system.current_spins,
            system.neighborhood,
        )
        energy_array = (
            accelerated_energy[0]
            if accelerated_energy is not None
            else grid_gen(system.current_spins, system.neighborhood)
        )

        # Clamp range for safety
        min_temp = 1e-6
        max_temp = 1e6

        for i in range(max_adjustments):
            T = float(system.temperature)

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
            system.temperature = float(np.clip(T_new, min_temp, max_temp))

        return float(system.temperature)
    finally:
        restore_numpy_state(rng_state)


def simulated_annealing(
        system: "System",
        *,
        schedule: AnnealSchedule,
        save_path: str | Path | None = None,
        save_interval: int = 1,
        verbose: bool = False,
        reinit: bool = False,
        seed: Seed = None,
) -> None:
    """Main simulated annealing loop.

    Uses the provided schedule's starting temperature as starting T
    (you can call `thermalize_system` beforehand to tune it).

    Parameters
    ----------
    system : System
        System object, mutated in-place.
    save_path : str | Path | None, optional
        If provided, save periodic and final NetCDF state to this path.
    save_interval : int, optional
        Save every N temperature iterations when save_path is provided.
    verbose : bool, optional
        Whether to print annealing diagnostics, by default False.
    reinit : bool, optional
        If True, recompute the energy grid and scalar energies after each
        temperature step to correct numerical or incremental-update drift.
    """
    _require_annealable(system)
    _validate_schedule(schedule, save_interval=save_interval)

    rng_state = seed_numpy(seed)
    try:
        engine = get_engine(system.kind)
        grid_gen = engine.grid_gen

        # Build initial energy grid consistent with current_spins
        if reinit:
            energy_array = _recompute_energy_state(system, engine)
        else:
            accelerated_energy = compute_spinflip_numba_energy_grid(
                system.current_spins,
                system.neighborhood,
            )
            energy_array = (
                accelerated_energy[0]
                if accelerated_energy is not None
                else grid_gen(system.current_spins, system.neighborhood)
            )

        start_time = time.time()
        acc = 1.0
        iterations_run = 0
        temperature = float(schedule.starting_temp)
        system.temperature = max(float(temperature), 1e-12)
        steps = int(schedule.starting_steps)
        while (
                temperature > schedule.final_temp
                and iterations_run < schedule.max_iterations
                and (time.time() - start_time) / 60 ** 2
                < schedule.max_runtime_hours
                and acc > schedule.min_acceptance
        ):
            T = temperature
            n_steps = steps

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
            # Cooling schedule

            steps = max(
                1,
                int(steps / schedule.cooling_rate),
            )
            temperature *= schedule.cooling_rate
            system.temperature = max(float(temperature), 1e-12)
            system.anneal_data.iterations += 1
            iterations_run += 1
            if reinit:
                energy_array = _recompute_energy_state(system, engine)
            system.anneal_data.append_record(
                temperature=float(T),
                steps=int(n_steps),
                acceptance_rate=float(acc),
                current_energy=float(system.current_energy),
                best_energy=float(system.best_energy),
            )

            # Periodic save
            if save_path is not None and iterations_run % save_interval == 0:
                system.save(save_path)

        # Final save
        system.temperature = max(float(temperature), 1e-12)
        if save_path is not None:
            system.save(save_path)
    finally:
        restore_numpy_state(rng_state)
