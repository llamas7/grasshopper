from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .annealing import metropolis_sweep
from .core import (
    Engine,
    NonAntipodalNeighborhoodCSR,
    OccupancyState,
    SystemConfig,
)
from .engines import get_engine
from .initialization import init_from_array, init_random, sync_occupancy
from .rng import Seed, restore_numpy_state, seed_numpy
from .spinflip_numba import compute_spinflip_numba_energy_grid

if TYPE_CHECKING:
    from .system import System


_WORKER_CONTEXT: dict[str, Any] | None = None


@dataclass(frozen=True)
class _SharedArraySpec:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class _SharedCSRSpec:
    indptr: _SharedArraySpec
    neighbor_spins: _SharedArraySpec
    contributions: _SharedArraySpec
    spin_potential: _SharedArraySpec


@dataclass(frozen=True)
class ParallelTemperingResult:
    iterations: int
    temperatures: np.ndarray
    best_energy: float
    cold_energy: float
    final_energies: list[float]
    sweep_acceptance: list[list[float]]
    exchange_attempts: list[int]
    exchange_accepts: list[int]
    exchange_rates: list[float]


@dataclass
class _WorkerSystem:
    config: SystemConfig
    neighborhood: Any
    num_spins: int
    antipodal: bool
    current_system: np.ndarray
    current_energy: float
    best_system: np.ndarray
    best_energy: float
    occupancy: OccupancyState | None


@dataclass(frozen=True)
class _ReplicaTask:
    current_system: np.ndarray
    current_energy: float
    occupancy: OccupancyState | None
    energy_grid: np.ndarray
    temperature: float
    sweep_steps: int
    reinit: bool
    seed: Seed


@dataclass(frozen=True)
class _ReplicaSweepResult:
    current_system: np.ndarray
    current_energy: float
    occupancy: OccupancyState | None
    energy_grid: np.ndarray
    acceptance: float


def _share_array(array: np.ndarray) -> tuple[_SharedArraySpec, shared_memory.SharedMemory]:
    array = np.ascontiguousarray(array)
    shm = shared_memory.SharedMemory(create=True, size=max(1, int(array.nbytes)))
    view = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    view[...] = array
    return (
        _SharedArraySpec(
            name=shm.name,
            shape=tuple(int(dim) for dim in array.shape),
            dtype=array.dtype.str,
        ),
        shm,
    )


def _share_csr_neighborhood(
        neighborhood: NonAntipodalNeighborhoodCSR,
) -> tuple[_SharedCSRSpec, list[shared_memory.SharedMemory]]:
    specs: list[_SharedArraySpec] = []
    handles: list[shared_memory.SharedMemory] = []
    for array in (
            neighborhood.indptr,
            neighborhood.neighbor_spins,
            neighborhood.contributions,
            neighborhood.spin_potential,
    ):
        spec, handle = _share_array(array)
        specs.append(spec)
        handles.append(handle)
    return (
        _SharedCSRSpec(
            indptr=specs[0],
            neighbor_spins=specs[1],
            contributions=specs[2],
            spin_potential=specs[3],
        ),
        handles,
    )


def _attach_shared_array(
        spec: _SharedArraySpec,
) -> tuple[np.ndarray, shared_memory.SharedMemory]:
    shm = shared_memory.SharedMemory(name=spec.name)
    array = np.ndarray(spec.shape, dtype=np.dtype(spec.dtype), buffer=shm.buf)
    return array, shm


def _attach_worker_neighborhood(
        neighborhood_context: Any,
) -> tuple[Any, list[shared_memory.SharedMemory]]:
    if not isinstance(neighborhood_context, _SharedCSRSpec):
        return neighborhood_context, []

    indptr, indptr_shm = _attach_shared_array(neighborhood_context.indptr)
    neighbor_spins, neighbor_spins_shm = _attach_shared_array(
        neighborhood_context.neighbor_spins
    )
    contributions, contributions_shm = _attach_shared_array(
        neighborhood_context.contributions
    )
    spin_potential, spin_potential_shm = _attach_shared_array(
        neighborhood_context.spin_potential
    )
    return (
        NonAntipodalNeighborhoodCSR(
            indptr=indptr,
            neighbor_spins=neighbor_spins,
            contributions=contributions,
            spin_potential=spin_potential,
        ),
        [indptr_shm, neighbor_spins_shm, contributions_shm, spin_potential_shm],
    )


def _release_shared_memory(handles: list[shared_memory.SharedMemory]) -> None:
    for handle in handles:
        try:
            handle.close()
        except FileNotFoundError:
            pass
        try:
            handle.unlink()
        except FileNotFoundError:
            pass


def _init_worker_context(
        config: SystemConfig,
        neighborhood_context: Any,
        num_spins: int,
        antipodal: bool,
) -> None:
    global _WORKER_CONTEXT
    neighborhood, shared_handles = _attach_worker_neighborhood(neighborhood_context)
    _WORKER_CONTEXT = {
        "config": config,
        "neighborhood": neighborhood,
        "num_spins": int(num_spins),
        "antipodal": bool(antipodal),
        "shared_memory_handles": shared_handles,
    }


def _require_parallel_tempering_ready(system: "System") -> None:
    if system.current_system is None:
        raise RuntimeError("System has no current spin state. Initialize it first.")
    if system.best_system is None:
        raise RuntimeError("System has no best spin state. Initialize it first.")
    if system.neighborhood is None:
        raise RuntimeError("System has no neighborhood. Call compute_neighborhood() first.")


def _temperature_ladder(
        system: "System",
        temperatures: Sequence[float] | None,
        num_replicas: int,
        cold_temp: float,
        hot_temp: float | None,
) -> np.ndarray:
    if temperatures is not None:
        ladder = np.asarray(temperatures, dtype=float)
        if ladder.ndim != 1:
            raise ValueError("temperatures must be a one-dimensional sequence.")
        if ladder.size < 2:
            raise ValueError("Parallel tempering requires at least two temperatures.")
    else:
        if num_replicas < 2:
            raise ValueError("num_replicas must be at least 2.")
        cold = max(float(cold_temp), 1e-12)
        hot = max(float(system.temperature if hot_temp is None else hot_temp), cold)
        if hot <= cold:
            hot = cold * 10.0
        ladder = np.geomspace(cold, hot, int(num_replicas))

    if not np.all(np.isfinite(ladder)):
        raise ValueError("temperatures must be finite.")
    if np.any(ladder <= 0):
        raise ValueError("temperatures must be positive for parallel tempering.")
    return np.sort(ladder)


def _clone_replica(system: "System") -> "System":
    replica = system.clone()
    replica.current_system = system.current_system.copy()
    replica.best_system = system.current_system.copy()
    replica.current_energy = float(system.current_energy)
    replica.best_energy = float(system.current_energy)
    replica.set_occupancy(system.occupancy)
    return replica


def _make_replicas(
        system: "System",
        engine: Engine,
        num_replicas: int,
        initial_states: Sequence[np.ndarray] | None,
) -> list["System"]:
    if initial_states is not None and len(initial_states) != num_replicas:
        raise ValueError(
            f"initial_states must contain one state per replica "
            f"({num_replicas}), got {len(initial_states)}."
        )

    replicas: list[System] = []
    if initial_states is None:
        replicas.append(_clone_replica(system))
        for _ in range(1, num_replicas):
            replica = _clone_replica(system)
            init_random(replica)
            replicas.append(replica)
    else:
        for state in initial_states:
            replica = _clone_replica(system)
            init_from_array(replica, np.asarray(state))
            replicas.append(replica)

    system_energy_fn = engine.energy
    for replica in replicas:
        replica.current_energy = float(
            system_energy_fn(replica.current_system, replica.neighborhood)
        )
        replica.best_system = replica.current_system.copy()
        replica.best_energy = replica.current_energy
        if not replica.antipodal:
            sync_occupancy(replica)
    return replicas


def _recompute_replica(replica: "System", engine: Engine) -> np.ndarray:
    accelerated = compute_spinflip_numba_energy_grid(
        replica.current_system,
        replica.neighborhood,
    )
    if accelerated is not None:
        energy_grid, current_energy = accelerated
        replica.current_energy = float(current_energy)
    else:
        energy_grid = engine.grid_gen(replica.current_system, replica.neighborhood)
        replica.current_energy = float(
            engine.energy(replica.current_system, replica.neighborhood)
        )
    if not replica.antipodal:
        sync_occupancy(replica)
    return energy_grid


def _sync_worker_occupancy(system: _WorkerSystem) -> None:
    system.occupancy = OccupancyState.from_spin_array(system.current_system)


def _make_worker_task(
        replica: "System",
        energy_grid: np.ndarray,
        temperature: float,
        sweep_steps: int,
        reinit: bool,
        seed: Seed,
) -> _ReplicaTask:
    return _ReplicaTask(
        current_system=replica.current_system,
        current_energy=float(replica.current_energy),
        occupancy=None if replica.occupancy is None else replica.occupancy.copy(),
        energy_grid=energy_grid,
        temperature=float(temperature),
        sweep_steps=int(sweep_steps),
        reinit=bool(reinit),
        seed=seed,
    )


def _task_seed(base_seed: Seed, iteration: int, replica_index: int) -> int | None:
    if base_seed is None:
        return None
    return int(
        np.random.SeedSequence(
            [int(base_seed), int(iteration), int(replica_index)]
        ).generate_state(1, dtype=np.uint32)[0]
    )


def _run_replica_sweep_task(task: _ReplicaTask) -> _ReplicaSweepResult:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Parallel tempering worker context has not been initialized.")
    if task.seed is not None:
        np.random.seed(task.seed)

    system = _WorkerSystem(
        config=_WORKER_CONTEXT["config"],
        neighborhood=_WORKER_CONTEXT["neighborhood"],
        num_spins=_WORKER_CONTEXT["num_spins"],
        antipodal=_WORKER_CONTEXT["antipodal"],
        current_system=task.current_system.copy(),
        current_energy=float(task.current_energy),
        best_system=task.current_system.copy(),
        best_energy=float(task.current_energy),
        occupancy=None if task.occupancy is None else task.occupancy.copy(),
    )
    energy_grid = task.energy_grid.copy()
    engine = get_engine(system.config)
    acceptance = metropolis_sweep(
        system=system,
        energy_array=energy_grid,
        T=task.temperature,
        n_steps=task.sweep_steps,
        engine=engine,
        track_best=False,
    )
    if task.reinit:
        accelerated = compute_spinflip_numba_energy_grid(
            system.current_system,
            system.neighborhood,
        )
        if accelerated is not None:
            energy_grid, current_energy = accelerated
            system.current_energy = float(current_energy)
        else:
            energy_grid = engine.grid_gen(system.current_system, system.neighborhood)
            system.current_energy = float(
                engine.energy(system.current_system, system.neighborhood)
            )
        if not system.antipodal:
            _sync_worker_occupancy(system)

    return _ReplicaSweepResult(
        current_system=system.current_system,
        current_energy=float(system.current_energy),
        occupancy=None if system.occupancy is None else system.occupancy.copy(),
        energy_grid=energy_grid,
        acceptance=float(acceptance),
    )


def _apply_sweep_result(replica: "System", result: _ReplicaSweepResult) -> None:
    replica.current_system = result.current_system
    replica.current_energy = float(result.current_energy)
    replica.set_occupancy(result.occupancy)


def _accept_exchange(
        left_energy: float,
        right_energy: float,
        left_temperature: float,
        right_temperature: float,
) -> bool:
    log_acceptance = (
            (1.0 / left_temperature - 1.0 / right_temperature)
            * (right_energy - left_energy)
    )
    return log_acceptance >= 0 or np.random.rand() < np.exp(log_acceptance)


def _copy_replica_state(source: "System", target: "System") -> None:
    target.current_system = source.current_system.copy()
    target.current_energy = float(source.current_energy)
    target.set_occupancy(source.occupancy)


def _set_best(system: "System", best_system: np.ndarray, best_energy: float) -> None:
    system.best_system = best_system.copy()
    system.best_energy = float(best_energy)


def _parallel_tempering_impl(
        system: "System",
        *,
        temperatures: Sequence[float] | None = None,
        num_replicas: int = 8,
        iterations: int = 100,
        sweep_steps: int = 100_000,
        exchange_interval: int = 1,
        initial_states: Sequence[np.ndarray] | None = None,
        cold_temp: float = 0.05,
        hot_temp: float | None = None,
        reinit: bool = False,
        track_best: bool = True,
        save_path: str | Path | None = None,
        seed: Seed = None,
        n_workers: int = 1,
        verbose: bool = False,
) -> ParallelTemperingResult:
    """Run replica-exchange Monte Carlo over a fixed temperature ladder.

    With n_workers > 1, each worker process is initialized once with the
    immutable neighborhood/engine context. Per-iteration tasks send only replica
    state, occupancy, energy grids, and temperature data.
    """
    _require_parallel_tempering_ready(system)
    iterations = int(iterations)
    sweep_steps = int(sweep_steps)
    exchange_interval = int(exchange_interval)
    n_workers = int(n_workers)
    if iterations < 0:
        raise ValueError("iterations must be nonnegative.")
    if sweep_steps <= 0:
        raise ValueError("sweep_steps must be positive.")
    if exchange_interval <= 0:
        raise ValueError("exchange_interval must be positive.")
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1.")

    ladder = _temperature_ladder(
        system,
        temperatures,
        num_replicas,
        cold_temp,
        hot_temp,
    )
    num_replicas = int(ladder.size)
    engine = get_engine(system.config)
    replicas = _make_replicas(system, engine, num_replicas, initial_states)
    energy_grids = [_recompute_replica(replica, engine) for replica in replicas]

    best_energy = float(system.best_energy)
    best_system = system.best_system.copy()
    for replica in replicas:
        if replica.current_energy > best_energy:
            best_energy = float(replica.current_energy)
            best_system = replica.current_system.copy()

    sweep_acceptance: list[list[float]] = []
    cold_energy_history: list[float] = []
    exchange_attempts = [0 for _ in range(num_replicas - 1)]
    exchange_accepts = [0 for _ in range(num_replicas - 1)]

    executor = None
    shared_memory_handles: list[shared_memory.SharedMemory] = []
    if n_workers > 1 and iterations > 0:
        neighborhood_context = system.neighborhood
        if isinstance(system.neighborhood, NonAntipodalNeighborhoodCSR):
            neighborhood_context, shared_memory_handles = _share_csr_neighborhood(
                system.neighborhood
            )
        executor = ProcessPoolExecutor(
            max_workers=min(n_workers, num_replicas),
            initializer=_init_worker_context,
            initargs=(
                system.config,
                neighborhood_context,
                system.num_spins,
                system.antipodal,
            ),
        )

    try:
        for iteration in range(iterations):
            iteration_acceptance: list[float] = []
            if executor is None:
                for idx, (replica, temperature) in enumerate(zip(replicas, ladder)):
                    acceptance = metropolis_sweep(
                        system=replica,
                        energy_array=energy_grids[idx],
                        T=float(temperature),
                        n_steps=sweep_steps,
                        engine=engine,
                        track_best=False,
                    )
                    iteration_acceptance.append(float(acceptance))
                    if reinit:
                        energy_grids[idx] = _recompute_replica(replica, engine)

                    if track_best and replica.current_energy > best_energy:
                        best_energy = float(replica.current_energy)
                        best_system = replica.current_system.copy()
            else:
                tasks = [
                    _make_worker_task(
                        replica=replica,
                        energy_grid=energy_grids[idx],
                        temperature=float(temperature),
                        sweep_steps=sweep_steps,
                        reinit=reinit,
                        seed=_task_seed(seed, iteration, idx),
                    )
                    for idx, (replica, temperature) in enumerate(zip(replicas, ladder))
                ]
                results = list(executor.map(_run_replica_sweep_task, tasks))
                for idx, result in enumerate(results):
                    _apply_sweep_result(replicas[idx], result)
                    energy_grids[idx] = result.energy_grid
                    iteration_acceptance.append(float(result.acceptance))

                    if track_best and replicas[idx].current_energy > best_energy:
                        best_energy = float(replicas[idx].current_energy)
                        best_system = replicas[idx].current_system.copy()

            if iteration % exchange_interval == 0:
                start_pair = iteration % 2
                for left in range(start_pair, num_replicas - 1, 2):
                    right = left + 1
                    exchange_attempts[left] += 1
                    if _accept_exchange(
                            replicas[left].current_energy,
                            replicas[right].current_energy,
                            float(ladder[left]),
                            float(ladder[right]),
                    ):
                        replicas[left], replicas[right] = (
                            replicas[right],
                            replicas[left],
                        )
                        energy_grids[left], energy_grids[right] = (
                            energy_grids[right],
                            energy_grids[left],
                        )
                        exchange_accepts[left] += 1

            sweep_acceptance.append(iteration_acceptance)
            cold_energy_history.append(float(replicas[0].current_energy))

            if verbose:
                attempted = sum(exchange_attempts)
                accepted = sum(exchange_accepts)
                exchange_rate = accepted / attempted if attempted else 0.0
                print(
                    f"[parallel_tempering] iter={iteration}, "
                    f"cold_E={replicas[0].current_energy:.6g}, "
                    f"best_E={best_energy:.6g}, exchange={exchange_rate:.4g}"
                )
    finally:
        if executor is not None:
            executor.shutdown()
        _release_shared_memory(shared_memory_handles)

    _copy_replica_state(replicas[0], system)
    if track_best and best_energy > system.best_energy:
        _set_best(system, best_system, best_energy)
    elif track_best and system.best_system is None:
        _set_best(system, best_system, best_energy)

    exchange_rates = [
        accepts / attempts if attempts else 0.0
        for accepts, attempts in zip(exchange_accepts, exchange_attempts)
    ]
    system.parallel_tempering_history.append_run(
        iterations=int(iterations),
        temperatures=ladder,
        best_energy=float(system.best_energy if track_best else best_energy),
        cold_energy=float(replicas[0].current_energy),
        final_energies=[float(replica.current_energy) for replica in replicas],
        cold_energy_history=cold_energy_history,
        sweep_acceptance=sweep_acceptance,
        exchange_attempts=exchange_attempts,
        exchange_accepts=exchange_accepts,
        exchange_rates=exchange_rates,
    )

    if save_path is not None:
        system.save(save_path)

    return ParallelTemperingResult(
        iterations=iterations,
        temperatures=ladder,
        best_energy=float(best_energy),
        cold_energy=float(replicas[0].current_energy),
        final_energies=[float(replica.current_energy) for replica in replicas],
        sweep_acceptance=sweep_acceptance,
        exchange_attempts=exchange_attempts,
        exchange_accepts=exchange_accepts,
        exchange_rates=exchange_rates,
    )


def parallel_tempering(
        system: "System",
        *,
        temperatures: Sequence[float] | None = None,
        num_replicas: int = 8,
        iterations: int = 100,
        sweep_steps: int = 100_000,
        exchange_interval: int = 1,
        initial_states: Sequence[np.ndarray] | None = None,
        cold_temp: float = 0.05,
        hot_temp: float | None = None,
        reinit: bool = False,
        track_best: bool = True,
        save_path: str | Path | None = None,
        seed: Seed = None,
        n_workers: int = 1,
        verbose: bool = False,
) -> ParallelTemperingResult:
    """Run replica-exchange Monte Carlo with optional reproducible seeding."""
    rng_state = seed_numpy(seed)
    try:
        return _parallel_tempering_impl(
            system,
            temperatures=temperatures,
            num_replicas=num_replicas,
            iterations=iterations,
            sweep_steps=sweep_steps,
            exchange_interval=exchange_interval,
            initial_states=initial_states,
            cold_temp=cold_temp,
            hot_temp=hot_temp,
            reinit=reinit,
            track_best=track_best,
            save_path=save_path,
            seed=seed,
            n_workers=n_workers,
            verbose=verbose,
        )
    finally:
        restore_numpy_state(rng_state)
