from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .core import Engine
from .engines import get_engine

if TYPE_CHECKING:
    from .system import System


@dataclass(frozen=True)
class GreedyResult:
    steps: int
    current_energy: float
    best_energy: float
    last_delta: float | None
    converged: bool


def _require_greedy_ready(system: "System") -> None:
    if system.current_system is None:
        raise RuntimeError("System has no current spin state. Initialize it first.")
    if system.best_system is None:
        raise RuntimeError("System has no best spin state. Initialize it first.")
    if system.neighborhood is None:
        raise RuntimeError(
            "System has no neighborhood. Call compute_neighborhood() first."
        )


def _recompute_energy_state(
    system: "System",
    engine: Engine,
    track_best: bool,
) -> np.ndarray:
    grid_gen = engine.grid_gen
    system_energy_fn = engine.energy
    energy_array = grid_gen(system.current_system, system.neighborhood)
    system.current_energy = float(
        system_energy_fn(system.current_system, system.neighborhood)
    )
    system.best_energy = float(
        system_energy_fn(system.best_system, system.neighborhood)
    )
    if track_best and system.current_energy > system.best_energy:
        system.best_energy = system.current_energy
        system.best_system = system.current_system.copy()
    return energy_array


def greedy_optimization(
    system: "System",
    *,
    max_steps: int | None = None,
    min_delta: float = 0.0,
    track_best: bool = True,
    reinit: bool = False,
    verbose: bool = False,
) -> GreedyResult:
    """Run deterministic greedy optimization using the active system engine.

    At each step, the engine's greedy update is selected. The update is
    applied only when its energy delta is strictly larger than `min_delta`.
    When `reinit=True`, the exact energy state is recomputed only at the
    beginning and end of the optimization run.
    """
    _require_greedy_ready(system)
    if max_steps is not None and max_steps < 0:
        raise ValueError(f"max_steps must be nonnegative or None, got {max_steps!r}.")
    if min_delta < 0:
        raise ValueError(f"min_delta must be nonnegative, got {min_delta!r}.")

    engine = get_engine(system.kind)
    best_update = engine.best_update
    update = engine.update

    energy_array = (
        _recompute_energy_state(system, engine, track_best)
        if reinit
        else engine.grid_gen(system.current_system, system.neighborhood)
    )

    steps = 0
    last_delta = None
    converged = False

    while max_steps is None or steps < max_steps:
        suggested_update, dE = best_update(system, energy_array)
        if suggested_update is None:
            converged = True
            break

        dE = float(dE)
        if dE <= min_delta:
            converged = True
            break

        update(system, suggested_update, energy_array)
        system.current_energy += dE

        if track_best and system.current_energy > system.best_energy:
            system.best_energy = system.current_energy
            system.best_system = system.current_system.copy()

        steps += 1
        last_delta = dE

        if verbose:
            print(
                f"[greedy] step={steps}, dE={dE:.6g}, "
                f"energy={system.current_energy:.6g}"
            )

    if reinit:
        _recompute_energy_state(system, engine, track_best)

    system.greedy_history.append_result(
        steps=int(steps),
        current_energy=float(system.current_energy),
        best_energy=float(system.best_energy),
        last_delta=last_delta,
        converged=converged,
    )

    return GreedyResult(
        steps=steps,
        current_energy=float(system.current_energy),
        best_energy=float(system.best_energy),
        last_delta=last_delta,
        converged=converged,
    )
