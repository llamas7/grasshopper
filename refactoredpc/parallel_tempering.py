from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .annealing import metropolis_sweep
from .core import Engine, OccupancyState
from .engines import get_engine
from .initialization import build_spin_lists, init_from_array, init_random

if TYPE_CHECKING:
    from .system import System


_WORKER_CONTEXT: dict[str, Any] | None = None


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
    system_type: Any
    neighborhood: Any
    num_spins: int
    antipodal: bool
    current_system: np.ndarray
    current_energy: float
    best_system: np.ndarray
    best_energy: float
    up_spins: list[int] | None
    down_spins: list[int] | None


@dataclass(frozen=True)
class _ReplicaTask:
    current_system: np.ndarray
    current_energy: float
    up_spins: list[int] | None
    down_spins: list[int] | None
    energy_grid: np.ndarray
    temperature: float
    sweep_steps: int
    reinit: bool
    seed: int | None


@dataclass(frozen=True)
class _ReplicaSweepResult:
    current_system: np.ndarray
    current_energy: float
    up_spins: list[int] | None
    down_spins: list[int] | None
    energy_grid: np.ndarray
    acceptance: float


def _init_worker_context(
        system_type: Any,
        neighborhood: Any,
        num_spins: int,
        antipodal: bool,
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "system_type": system_type,
        "neighborhood": neighborhood,
        "num_spins": int(num_spins),
        "antipodal": bool(antipodal),
    }


def _require_parallel_tempering_ready(system: "System") -> None:
    if system.current_system is None:
        raise RuntimeError("System has no current spin state. Initialize it first.")
    if system.best_system is None:
        raise RuntimeError("System has no best spin state. Initialize it first.")
    if system.neighborhood is None:
        raise RuntimeError("System has no neighborhood. Call compute_neighborhood() first.")
    if system.anneal_data is None:
        raise RuntimeError("System has no anneal_data.")


def _temperature_ladder(
        system: "System",
        temperatures: Sequence[float] | None,
        num_replicas: int,
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
        cold = max(float(system.parameters.final_temp), 1e-12)
        hot = max(float(system.anneal_data.temperature), cold)
        if hot <= cold:
            hot = cold * 10.0
        ladder = np.geomspace(cold, hot, int(num_replicas))

    if not np.all(np.isfinite(ladder)):
        raise ValueError("temperatures must be finite.")
    if np.any(ladder <= 0):
        raise ValueError("temperatures must be positive for parallel tempering.")
    return np.sort(ladder)


def _clone_replica(system: "System") -> "System":
    replica = copy.copy(system)
    replica.current_system = system.current_system.copy()
    replica.best_system = system.current_system.copy()
    replica.current_energy = float(system.current_energy)
    replica.best_energy = float(system.current_energy)
    if system.up_spins is not None and system.down_spins is not None:
        replica.set_occupancy(
            OccupancyState(
                occupied=list(system.up_spins),
                unoccupied=list(system.down_spins),
            )
        )
    else:
        replica.set_occupancy(None)
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
            build_spin_lists(replica)
    return replicas


def _recompute_replica(replica: "System", engine: Engine) -> np.ndarray:
    energy_grid = engine.grid_gen(replica.current_system, replica.neighborhood)
    replica.current_energy = float(
        engine.energy(replica.current_system, replica.neighborhood)
    )
    if not replica.antipodal:
        build_spin_lists(replica)
    return energy_grid


def _build_spin_lists_from_state(system: _WorkerSystem) -> None:
    occupancy = OccupancyState.from_spin_array(system.current_system)
    system.up_spins = occupancy.occupied
    system.down_spins = occupancy.unoccupied


def _make_worker_task(
        replica: "System",
        energy_grid: np.ndarray,
        temperature: float,
        sweep_steps: int,
        reinit: bool,
        seed: int | None,
) -> _ReplicaTask:
    return _ReplicaTask(
        current_system=replica.current_system,
        current_energy=float(replica.current_energy),
        up_spins=list(replica.up_spins) if replica.up_spins is not None else None,
        down_spins=(
            list(replica.down_spins) if replica.down_spins is not None else None
        ),
        energy_grid=energy_grid,
        temperature=float(temperature),
        sweep_steps=int(sweep_steps),
        reinit=bool(reinit),
        seed=seed,
    )


def _task_seed(base_seed: int | None, iteration: int, replica_index: int) -> int | None:
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
        system_type=_WORKER_CONTEXT["system_type"],
        neighborhood=_WORKER_CONTEXT["neighborhood"],
        num_spins=_WORKER_CONTEXT["num_spins"],
        antipodal=_WORKER_CONTEXT["antipodal"],
        current_system=task.current_system.copy(),
        current_energy=float(task.current_energy),
        best_system=task.current_system.copy(),
        best_energy=float(task.current_energy),
        up_spins=list(task.up_spins) if task.up_spins is not None else None,
        down_spins=list(task.down_spins) if task.down_spins is not None else None,
    )
    energy_grid = task.energy_grid.copy()
    engine = get_engine(system.system_type)
    acceptance = metropolis_sweep(
        system=system,
        energy_array=energy_grid,
        T=task.temperature,
        n_steps=task.sweep_steps,
        engine=engine,
        track_best=False,
    )
    if task.reinit:
        energy_grid = engine.grid_gen(system.current_system, system.neighborhood)
        system.current_energy = float(
            engine.energy(system.current_system, system.neighborhood)
        )
        if not system.antipodal:
            _build_spin_lists_from_state(system)

    return _ReplicaSweepResult(
        current_system=system.current_system,
        current_energy=float(system.current_energy),
        up_spins=list(system.up_spins) if system.up_spins is not None else None,
        down_spins=list(system.down_spins) if system.down_spins is not None else None,
        energy_grid=energy_grid,
        acceptance=float(acceptance),
    )


def _apply_sweep_result(replica: "System", result: _ReplicaSweepResult) -> None:
    replica.current_system = result.current_system
    replica.current_energy = float(result.current_energy)
    if result.up_spins is not None and result.down_spins is not None:
        replica.set_occupancy(
            OccupancyState(
                occupied=list(result.up_spins),
                unoccupied=list(result.down_spins),
            )
        )
    else:
        replica.set_occupancy(None)


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
    if source.up_spins is not None and source.down_spins is not None:
        target.set_occupancy(
            OccupancyState(
                occupied=list(source.up_spins),
                unoccupied=list(source.down_spins),
            )
        )
    else:
        target.set_occupancy(None)


def _set_best(system: "System", best_system: np.ndarray, best_energy: float) -> None:
    system.best_system = best_system.copy()
    system.best_energy = float(best_energy)


def parallel_tempering(
        system: "System",
        *,
        temperatures: Sequence[float] | None = None,
        num_replicas: int = 8,
        iterations: int | None = None,
        sweep_steps: int | None = None,
        exchange_interval: int = 1,
        initial_states: Sequence[np.ndarray] | None = None,
        reinit: bool = True,
        track_best: bool = True,
        save: bool = False,
        seed: int | None = None,
        n_workers: int = 1,
        verbose: bool = False,
) -> ParallelTemperingResult:
    """Run replica-exchange Monte Carlo over a fixed temperature ladder.

    With n_workers > 1, each worker process is initialized once with the
    immutable neighborhood/engine context. Per-iteration tasks send only replica
    state, spin lists, energy grids, and temperature data.
    """
    _require_parallel_tempering_ready(system)
    if seed is not None:
        np.random.seed(seed)
    if iterations is None:
        iterations = int(system.parameters.max_iterations)
    if sweep_steps is None:
        sweep_steps = int(system.anneal_data.steps)
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

    ladder = _temperature_ladder(system, temperatures, num_replicas)
    num_replicas = int(ladder.size)
    engine = get_engine(system.system_type)
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
    if n_workers > 1 and iterations > 0:
        executor = ProcessPoolExecutor(
            max_workers=min(n_workers, num_replicas),
            initializer=_init_worker_context,
            initargs=(
                system.system_type,
                system.neighborhood,
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

    _copy_replica_state(replicas[0], system)
    if track_best and best_energy > system.best_energy:
        _set_best(system, best_system, best_energy)
    elif track_best and system.best_system is None:
        _set_best(system, best_system, best_energy)

    system.anneal_data.iterations += iterations
    system.anneal_data.energies.extend(cold_energy_history)

    if save:
        system.save(system.parameters.save_dir)

    exchange_rates = [
        accepts / attempts if attempts else 0.0
        for accepts, attempts in zip(exchange_accepts, exchange_attempts)
    ]
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
