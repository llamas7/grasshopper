#!/usr/bin/env python3
"""Scan Gaussian widths and detect numerical morphology transitions.

This script is intentionally a numerical discovery tool, not a proof of phase
boundaries. For a fixed jump distance, grid length, and occupied spin count N,
it scans eta = sigma / d, optimizes several independent states at each eta,
computes coarse morphology diagnostics, and reports persistent label changes as
transition intervals.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

cache_root = Path(tempfile.gettempdir()) / "grasshopper-gaussian-transition-cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import grasshopper as gh


SCHEMA_VERSION = 1
LABELS = ("cogwheel", "dihedral/asymmetric", "radial disk", "annulus", "uncertain")
NUMERICAL_WIDTH_FACTOR = math.sqrt(4.0 / 3.0 - 8.0 / math.pi**2)


@dataclass(frozen=True)
class TransitionCase:
    jump_distance: float
    grid_length: int
    num_spins: int
    eta: float
    run_index: int

    @property
    def sigma(self) -> float:
        return self.eta * self.jump_distance

    @property
    def key(self) -> str:
        return (
            f"d{self.jump_distance:.9f}"
            f"__L{self.grid_length}"
            f"__N{self.num_spins}"
            f"__eta{self.eta:.9f}"
            f"__run{self.run_index:03d}"
        )


@dataclass(frozen=True)
class EtaSummary:
    eta: float
    records: tuple[dict[str, Any], ...]
    best: dict[str, Any]
    labels: tuple[str, ...]

    @property
    def ambiguous(self) -> bool:
        return len(self.labels) > 1


@dataclass(frozen=True)
class ScheduleDecision:
    eta: float
    run_index: int
    reason: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("Expected a finite nonnegative number.")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a finite positive number.")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Gaussian-broadened Grasshopper transition scans."
    )
    subparsers = parser.add_subparsers(dest="command")

    transitions = subparsers.add_parser(
        "transitions",
        help="scan eta=sigma/d and report numerical morphology crossovers",
    )
    transitions.add_argument("--jump-distance", type=positive_float, required=True)
    transitions.add_argument("--grid-length", type=positive_int, default=200)
    transitions.add_argument("--num-spins", type=positive_int, default=10_000)
    transitions.add_argument(
        "--eta-min",
        type=nonnegative_float,
        default=None,
        help=(
            "Smallest eta=sigma/d. Defaults to the Gaussian width matching "
            "the compact cosine kernel, sqrt(4/3 - 8/pi^2)/(d sqrt(N))."
        ),
    )
    transitions.add_argument("--eta-max", type=nonnegative_float, default=0.6)
    transitions.add_argument("--eta-step", type=positive_float, default=0.025)
    transitions.add_argument(
        "--eta",
        dest="explicit_etas",
        type=nonnegative_float,
        action="append",
        default=None,
        help="Specific eta value to include. May be repeated.",
    )
    transitions.add_argument("--runs", type=positive_int, default=3)
    transitions.add_argument("--seed", type=int, default=20260707)
    transitions.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Gauss_data") / "transitions",
    )
    transitions.add_argument("--overwrite", action="store_true")
    transitions.add_argument("--fail-fast", action="store_true")
    transitions.add_argument("--verbose", action="store_true")
    transitions.add_argument("--plot", action="store_true")

    transitions.add_argument("--gaussian-cutoff", type=positive_float, default=5.0)
    transitions.add_argument("--unbounded-gaussian", action="store_true")
    transitions.add_argument(
        "--initialization",
        choices=("mixed", "random", "disk", "disk-jitter", "annulus"),
        default="mixed",
    )
    transitions.add_argument("--jitter-fraction", type=nonnegative_float, default=0.08)
    transitions.add_argument("--annulus-radius", type=positive_float, default=None)

    transitions.add_argument("--workers", type=positive_int, default=1)
    transitions.add_argument("--replicas", type=positive_int, default=8)
    transitions.add_argument("--thermalize-steps", type=positive_int, default=None)
    transitions.add_argument("--thermalize-adjustments", type=int, default=10)
    transitions.add_argument("--target-acceptance", type=positive_float, default=0.6)
    transitions.add_argument("--tempering-iterations", type=positive_int, default=100)
    transitions.add_argument("--sweep-steps", type=positive_int, default=None)
    transitions.add_argument("--greedy-steps", type=positive_int, default=None)
    transitions.add_argument(
        "--cold-temp-factor",
        type=positive_float,
        default=1.0e-4,
        help="Cold parallel-tempering temperature as a fraction of thermalized T.",
    )

    transitions.add_argument("--m-max", type=positive_int, default=24)
    transitions.add_argument("--radial-bin-width", type=positive_float, default=None)
    transitions.add_argument(
        "--inner-density-threshold",
        type=positive_float,
        default=0.35,
    )
    transitions.add_argument("--inner-sustain-bins", type=positive_int, default=2)
    transitions.add_argument("--cog-threshold", type=positive_float, default=0.18)
    transitions.add_argument(
        "--asymmetry-threshold",
        type=positive_float,
        default=0.08,
    )
    transitions.add_argument(
        "--annulus-inner-radius-over-d",
        type=positive_float,
        default=0.12,
    )
    transitions.add_argument("--stability-points", type=positive_int, default=2)
    transitions.add_argument(
        "--uniform",
        action="store_true",
        help="Evaluate the complete eta grid instead of scheduling adaptively.",
    )
    transitions.add_argument("--initial-eta-points", type=positive_int, default=7)
    transitions.add_argument("--max-eta-points", type=positive_int, default=32)
    transitions.add_argument("--eta-tolerance", type=positive_float, default=None)
    transitions.add_argument("--exploration-width", type=positive_float, default=None)
    transitions.add_argument("--max-runs-per-eta", type=positive_int, default=None)
    transitions.add_argument("--explore-every", type=positive_int, default=4)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        raise SystemExit(2)
    if args.command == "transitions":
        validate_transition_args(parser, args)
    return args


def validate_transition_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.eta_max < resolved_eta_min(args):
        parser.error("--eta-max must be greater than or equal to --eta-min.")
    if args.grid_length * args.grid_length < args.num_spins:
        parser.error("--grid-length squared must be at least --num-spins.")
    if args.thermalize_adjustments < 0:
        parser.error("--thermalize-adjustments cannot be negative.")
    if args.inner_density_threshold > 1:
        parser.error("--inner-density-threshold must be at most 1.")
    if args.cog_threshold > 1 or args.asymmetry_threshold > 1:
        parser.error("mode thresholds must be at most 1.")
    if args.initial_eta_points < 2:
        parser.error("--initial-eta-points must be at least 2.")
    if args.max_eta_points < args.initial_eta_points:
        parser.error("--max-eta-points must be at least --initial-eta-points.")
    if args.max_runs_per_eta is None:
        args.max_runs_per_eta = 2 * args.runs
    elif args.max_runs_per_eta < args.runs:
        parser.error("--max-runs-per-eta must be at least --runs.")
    if args.eta_tolerance is None:
        args.eta_tolerance = args.eta_step / 8.0
    if args.exploration_width is None:
        args.exploration_width = 2.0 * args.eta_step
    if args.explicit_etas:
        eta_min = resolved_eta_min(args)
        if any(value < eta_min or value > args.eta_max for value in args.explicit_etas):
            parser.error("Every --eta value must lie between --eta-min and --eta-max.")
    if len(initial_eta_grid(args)) > args.max_eta_points:
        parser.error(
            "--max-eta-points must accommodate all initial and explicit eta values."
        )


def eta_grid(args: argparse.Namespace) -> list[float]:
    values: list[float] = []
    eta_min = resolved_eta_min(args)
    eta = eta_min
    tolerance = args.eta_step * 1.0e-6
    while eta <= args.eta_max + tolerance:
        values.append(round(float(eta), 10))
        eta += args.eta_step
    if args.explicit_etas:
        values.extend(round(float(value), 10) for value in args.explicit_etas)
    return sorted(set(values))


def initial_eta_grid(args: argparse.Namespace) -> list[float]:
    eta_min = resolved_eta_min(args)
    values = np.linspace(
        eta_min,
        args.eta_max,
        num=args.initial_eta_points,
        dtype=float,
    ).tolist()
    if args.explicit_etas:
        values.extend(float(value) for value in args.explicit_etas)
    return sorted({round(float(value), 10) for value in values})


def record_matches_scan(record: dict[str, Any], args: argparse.Namespace) -> bool:
    eta = float(record.get("eta", math.nan))
    return (
        math.isclose(
            float(record.get("jump_distance", math.nan)),
            float(args.jump_distance),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and int(record.get("grid_length", -1)) == int(args.grid_length)
        and int(record.get("num_spins", -1)) == int(args.num_spins)
        and resolved_eta_min(args) <= eta <= args.eta_max
    )


def compatible_records(
    records: Iterable[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    return [record for record in records if record_matches_scan(record, args)]


def summarize_etas(records: Iterable[dict[str, Any]]) -> dict[float, EtaSummary]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for record in records:
        eta = round(float(record["eta"]), 10)
        grouped.setdefault(eta, []).append(record)

    summaries: dict[float, EtaSummary] = {}
    for eta, eta_records in grouped.items():
        ordered = sorted(eta_records, key=lambda row: int(row["run_index"]))
        best = max(
            ordered,
            key=lambda row: (float(row["probability"]), -int(row["run_index"])),
        )
        labels = tuple(
            sorted({str(record["morphology_label"]) for record in ordered})
        )
        summaries[eta] = EtaSummary(
            eta=eta,
            records=tuple(ordered),
            best=best,
            labels=labels,
        )
    return summaries


def next_run_index(summary: EtaSummary | None) -> int:
    if summary is None or not summary.records:
        return 0
    return max(int(record["run_index"]) for record in summary.records) + 1


def _intervals(
    summaries: dict[float, EtaSummary],
) -> list[tuple[float, EtaSummary, EtaSummary]]:
    ordered = [summaries[eta] for eta in sorted(summaries)]
    return [
        (right.eta - left.eta, left, right)
        for left, right in zip(ordered, ordered[1:])
    ]


def _midpoint(left: float, right: float) -> float:
    return round((float(left) + float(right)) / 2.0, 10)


def _widest_eligible_interval(
    intervals: Iterable[tuple[float, EtaSummary, EtaSummary]],
    *,
    minimum_width: float,
    existing_etas: set[float],
    boundary: bool,
) -> tuple[float, EtaSummary, EtaSummary] | None:
    eligible = []
    for width, left, right in intervals:
        is_boundary = (
            left.best["morphology_label"] != right.best["morphology_label"]
            or left.best["morphology_label"] == "uncertain"
            or right.best["morphology_label"] == "uncertain"
        )
        if is_boundary != boundary or width <= minimum_width:
            continue
        if _midpoint(left.eta, right.eta) in existing_etas:
            continue
        eligible.append((width, left, right))
    if not eligible:
        return None
    return min(eligible, key=lambda item: (-item[0], item[1].eta))


def choose_adaptive_work(
    args: argparse.Namespace,
    records: Sequence[dict[str, Any]],
    *,
    boundary_since_exploration: int = 0,
) -> ScheduleDecision | None:
    summaries = summarize_etas(records)
    anchors = initial_eta_grid(args)

    for eta in anchors:
        summary = summaries.get(eta)
        if summary is None or len(summary.records) < args.runs:
            return ScheduleDecision(
                eta=eta,
                run_index=next_run_index(summary),
                reason="initial_anchor",
            )

    for eta in sorted(summaries):
        summary = summaries[eta]
        if summary.ambiguous and len(summary.records) < args.max_runs_per_eta:
            return ScheduleDecision(
                eta=eta,
                run_index=next_run_index(summary),
                reason="ambiguous_replicate",
            )

    existing_etas = set(summaries)
    if len(existing_etas) >= args.max_eta_points:
        return None

    intervals = _intervals(summaries)
    exploration = _widest_eligible_interval(
        intervals,
        minimum_width=args.exploration_width,
        existing_etas=existing_etas,
        boundary=False,
    )
    if boundary_since_exploration >= args.explore_every and exploration is not None:
        _, left, right = exploration
        return ScheduleDecision(
            eta=_midpoint(left.eta, right.eta),
            run_index=0,
            reason="periodic_exploration",
        )

    boundary_interval = _widest_eligible_interval(
        intervals,
        minimum_width=args.eta_tolerance,
        existing_etas=existing_etas,
        boundary=True,
    )
    if boundary_interval is not None:
        _, left, right = boundary_interval
        return ScheduleDecision(
            eta=_midpoint(left.eta, right.eta),
            run_index=0,
            reason="boundary_refinement",
        )

    if exploration is not None:
        _, left, right = exploration
        return ScheduleDecision(
            eta=_midpoint(left.eta, right.eta),
            run_index=0,
            reason="interval_exploration",
        )
    return None


def deterministic_seed(base_seed: int, key: str, stream: int) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8, person=b"gh-tran").digest()
    case_number = int.from_bytes(digest, byteorder="little", signed=False)
    sequence = np.random.SeedSequence([int(base_seed), case_number, int(stream)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def resolved_eta_min(args: argparse.Namespace) -> float:
    if args.eta_min is not None:
        return float(args.eta_min)
    return NUMERICAL_WIDTH_FACTOR / (args.jump_distance * math.sqrt(args.num_spins))


def disk_state(grid: NDArray[np.float64], num_spins: int) -> NDArray[np.bool_]:
    radii_squared = np.einsum("ij,ij->i", grid, grid)
    selected = np.argpartition(radii_squared, num_spins - 1)[:num_spins]
    state = np.zeros(len(grid), dtype=np.bool_)
    state[selected] = True
    return state


def annulus_state(
    grid: NDArray[np.float64],
    num_spins: int,
    target_radius: float,
) -> NDArray[np.bool_]:
    radii = np.linalg.norm(grid, axis=1)
    selected = np.argpartition(np.abs(radii - target_radius), num_spins - 1)[:num_spins]
    state = np.zeros(len(grid), dtype=np.bool_)
    state[selected] = True
    return state


def jitter_state(
    state: NDArray[np.bool_],
    fraction: float,
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    jittered = state.copy()
    occupied = np.flatnonzero(jittered)
    empty = np.flatnonzero(~jittered)
    swaps = min(len(occupied), len(empty), int(round(fraction * len(occupied))))
    if swaps <= 0:
        return jittered
    jittered[rng.choice(occupied, size=swaps, replace=False)] = False
    jittered[rng.choice(empty, size=swaps, replace=False)] = True
    return jittered


def initialization_for_run(requested: str, run_index: int) -> str:
    if requested != "mixed":
        return requested
    return ("disk", "disk-jitter", "annulus", "random")[run_index % 4]


def initialize_system(system: gh.System, args: argparse.Namespace, case: TransitionCase) -> str:
    kind = initialization_for_run(args.initialization, case.run_index)
    seed = deterministic_seed(args.seed, case.key, 1)
    if kind == "random":
        system.random_init(seed=seed)
    elif kind == "disk":
        system.array_init(disk_state(system.grid, system.num_spins))
    elif kind == "disk-jitter":
        rng = np.random.default_rng(seed)
        system.array_init(
            jitter_state(disk_state(system.grid, system.num_spins), args.jitter_fraction, rng)
        )
    elif kind == "annulus":
        radius = args.annulus_radius
        if radius is None:
            radius = max(args.jump_distance, math.sqrt(args.num_spins / math.pi) / math.sqrt(args.num_spins))
        system.array_init(annulus_state(system.grid, system.num_spins, radius))
    else:
        raise ValueError(f"Unsupported initialization {kind!r}.")
    return kind


def build_system(args: argparse.Namespace, eta: float) -> gh.System:
    system = gh.System.square_lattice(
        width=args.grid_length,
        height=args.grid_length,
        num_spins=args.num_spins,
        center=True,
    )
    if eta > 0:
        cutoff = None if args.unbounded_gaussian else args.gaussian_cutoff
        kernel = gh.KernelSpec.gaussian(sigma=eta * args.jump_distance, cutoff=cutoff)
    else:
        kernel = gh.KernelSpec.cosine()
    system.compute_neighborhood(
        jump_distance=args.jump_distance,
        kernel=kernel,
        distance_metric="euclidean",
        silent=not args.verbose,
    )
    return system


def optimize_system(system: gh.System, args: argparse.Namespace, case: TransitionCase) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "thermalized_temperature": None,
        "tempering_iterations": 0,
        "tempering_sweep_steps": 0,
        "greedy_steps": 0,
        "greedy_converged": None,
    }
    thermalize_steps = args.thermalize_steps or max(100, 5 * args.num_spins)
    sweep_steps = args.sweep_steps or max(100, 20 * args.num_spins)

    if args.thermalize_adjustments > 0:
        temperature = system.thermalize(
            target_acceptance=args.target_acceptance,
            steps_per_temp=thermalize_steps,
            max_adjustments=args.thermalize_adjustments,
            verbose=args.verbose,
            seed=deterministic_seed(args.seed, case.key, 2),
        )
    else:
        temperature = float(system.temperature)
    summary["thermalized_temperature"] = float(temperature)

    if args.tempering_iterations > 0:
        cold_temp = max(float(temperature) * args.cold_temp_factor, 1.0e-12)
        result = system.parallel_temper(
            num_replicas=args.replicas,
            iterations=args.tempering_iterations,
            sweep_steps=sweep_steps,
            cold_temp=cold_temp,
            seed=deterministic_seed(args.seed, case.key, 3),
            n_workers=args.workers,
            verbose=args.verbose,
        )
        summary["tempering_iterations"] = int(args.tempering_iterations)
        summary["tempering_sweep_steps"] = int(sweep_steps)
        sweep_acceptance = np.asarray(result.sweep_acceptance, dtype=float)
        exchange_rates = np.asarray(result.exchange_rates, dtype=float)
        summary["tempering_mean_acceptance"] = finite_or_none(
            np.mean(sweep_acceptance) if sweep_acceptance.size else None
        )
        summary["tempering_mean_exchange_rate"] = finite_or_none(
            np.mean(exchange_rates) if exchange_rates.size else None
        )

    greedy_result = system.greedy(
        max_steps=args.greedy_steps,
        min_delta=0.0,
        track_best=True,
        reinit=True,
        verbose=args.verbose,
    )
    summary["greedy_steps"] = int(greedy_result.steps)
    summary["greedy_converged"] = bool(greedy_result.converged)

    system.array_init(np.asarray(system.best_spins, dtype=np.bool_).copy())
    return summary


def radial_profile(
    radii: NDArray[np.float64],
    occupancy: NDArray[np.bool_],
    bin_width: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    indices = np.floor(radii / bin_width).astype(np.int64)
    bin_count = int(indices.max(initial=0)) + 1
    counts = np.bincount(indices, minlength=bin_count).astype(np.int64)
    occupied = np.bincount(indices, weights=occupancy.astype(float), minlength=bin_count)
    density = np.zeros(bin_count, dtype=float)
    populated = counts > 0
    density[populated] = occupied[populated] / counts[populated]
    centers = (np.arange(bin_count, dtype=float) + 0.5) * bin_width
    return centers, density, counts, indices


def smooth_density(density: NDArray[np.float64], counts: NDArray[np.int64]) -> NDArray[np.float64]:
    if len(density) < 3:
        return density.copy()
    smoothed = density.copy()
    for index in range(1, len(density) - 1):
        weight_sum = counts[index - 1] + counts[index] + counts[index + 1]
        if weight_sum > 0:
            smoothed[index] = (
                density[index - 1] * counts[index - 1]
                + density[index] * counts[index]
                + density[index + 1] * counts[index + 1]
            ) / weight_sum
    return np.clip(smoothed, 0.0, 1.0)


def estimate_inner_radius(
    density: NDArray[np.float64],
    counts: NDArray[np.int64],
    bin_width: float,
    *,
    threshold: float,
    sustain_bins: int,
) -> float:
    populated = np.flatnonzero(counts > 0)
    for position, bin_index in enumerate(populated):
        window = populated[position : position + sustain_bins]
        if len(window) < sustain_bins:
            break
        if np.any(np.diff(window) != 1):
            continue
        if np.all(density[window] >= threshold):
            return 0.0 if bin_index == populated[0] else float(bin_index * bin_width)
    return float("nan")


def angular_modes(
    angles: NDArray[np.float64],
    radial_indices: NDArray[np.int64],
    radial_density: NDArray[np.float64],
    occupancy: NDArray[np.bool_],
    m_max: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    modes = np.arange(2, m_max + 1, dtype=np.int64)
    fluctuation = occupancy.astype(float) - radial_density[radial_indices]
    denominator = float(np.sum(np.abs(fluctuation)))
    if denominator <= np.finfo(float).eps:
        return modes, np.zeros_like(modes, dtype=float)
    values = np.array(
        [abs(np.sum(fluctuation * np.exp(1j * mode * angles))) / denominator for mode in modes],
        dtype=float,
    )
    return modes, values


def component_count_square(
    occupancy: NDArray[np.bool_],
    grid_length: int,
) -> tuple[int, int]:
    if occupancy.size != grid_length * grid_length:
        return -1, -1
    grid = occupancy.reshape((grid_length, grid_length))
    seen = np.zeros_like(grid, dtype=np.bool_)
    components = 0
    largest = 0
    while True:
        unvisited = np.argwhere(grid & ~seen)
        if unvisited.size == 0:
            break
        start_y, start_x = unvisited[0]
        components += 1
        size = 0
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        while stack:
            y, x = stack.pop()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < grid_length and 0 <= nx < grid_length and grid[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        largest = max(largest, size)
    return components, largest


def morphology_diagnostics(
    grid: NDArray[np.float64],
    occupancy: NDArray[np.bool_],
    args: argparse.Namespace,
) -> dict[str, Any]:
    occupied_points = grid[occupancy]
    if occupied_points.size == 0:
        raise ValueError("Cannot diagnose an empty state.")

    center = np.mean(occupied_points, axis=0)
    relative = grid - center
    occupied_relative = relative[occupancy]
    radii = np.linalg.norm(relative, axis=1)
    occupied_radii = radii[occupancy]
    angles = np.arctan2(relative[:, 1], relative[:, 0])
    cell_length = args.num_spins ** -0.5
    bin_width = args.radial_bin_width or cell_length

    _, raw_density, counts, radial_indices = radial_profile(radii, occupancy, bin_width)
    density = smooth_density(raw_density, counts)
    modes, mode_values = angular_modes(
        angles,
        radial_indices,
        density,
        occupancy,
        args.m_max,
    )
    dominant_index = int(np.argmax(mode_values))
    dominant_mode = int(modes[dominant_index])
    anisotropy = float(mode_values[dominant_index])
    sorted_modes = np.sort(mode_values)
    second_mode = float(sorted_modes[-2]) if len(sorted_modes) > 1 else 0.0

    covariance = np.cov(occupied_relative.T, bias=True)
    eigenvalues = np.linalg.eigvalsh(covariance)
    inertia_total = float(np.sum(eigenvalues))
    inertia_anisotropy = (
        0.0 if inertia_total <= np.finfo(float).eps else float((eigenvalues[-1] - eigenvalues[0]) / inertia_total)
    )

    inner_radius = estimate_inner_radius(
        density,
        counts,
        bin_width,
        threshold=args.inner_density_threshold,
        sustain_bins=args.inner_sustain_bins,
    )
    components, largest_component = component_count_square(occupancy, args.grid_length)
    largest_fraction = largest_component / int(np.count_nonzero(occupancy)) if largest_component >= 0 else None

    metrics: dict[str, Any] = {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "dominant_mode": dominant_mode,
        "anisotropy_strength": anisotropy,
        "second_mode_strength": second_mode,
        "mode_prominence": None if second_mode <= np.finfo(float).eps else float(anisotropy / second_mode),
        "radial_symmetry_score": float(max(0.0, 1.0 - anisotropy)),
        "inner_radius": finite_or_none(inner_radius),
        "inner_radius_over_d": finite_or_none(inner_radius / args.jump_distance),
        "occupied_radius_mean": float(np.mean(occupied_radii)),
        "occupied_radius_std": float(np.std(occupied_radii)),
        "occupied_radius_q01": float(np.quantile(occupied_radii, 0.01)),
        "occupied_radius_q99": float(np.quantile(occupied_radii, 0.99)),
        "inertia_anisotropy": inertia_anisotropy,
        "component_count": components,
        "largest_component_fraction": finite_or_none(largest_fraction),
    }
    label, reason = classify_morphology(metrics, args)
    metrics["morphology_label"] = label
    metrics["classification_reason"] = reason
    return metrics


def classify_morphology(metrics: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    inner_over_d = metrics.get("inner_radius_over_d")
    anisotropy = float(metrics["anisotropy_strength"])
    mode = int(metrics["dominant_mode"])
    components = int(metrics["component_count"])
    largest_fraction = metrics.get("largest_component_fraction")

    if inner_over_d is not None and inner_over_d >= args.annulus_inner_radius_over_d:
        return "annulus", "persistent low-density center"
    if anisotropy >= args.cog_threshold and mode >= 3:
        return "cogwheel", f"dominant angular mode m={mode}"
    if anisotropy >= args.asymmetry_threshold or metrics["inertia_anisotropy"] >= args.asymmetry_threshold:
        return "dihedral/asymmetric", "anisotropic angular or inertia diagnostic"
    if components in (0, 1) or (largest_fraction is not None and largest_fraction >= 0.95):
        return "radial disk", "weak angular modes and no detected central hole"
    return "uncertain", "diagnostics do not cleanly match a coarse morphology"


def run_case(args: argparse.Namespace, case: TransitionCase) -> dict[str, Any]:
    record_path = record_file(args.output_dir, case)
    if record_path.exists() and not args.overwrite:
        with record_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    started = time.time()
    system = build_system(args, case.eta)
    initialization = initialize_system(system, args, case)
    optimizer = optimize_system(system, args, case)
    best_state = np.asarray(system.best_spins, dtype=np.bool_).copy()
    probability = float(system.probability(best_state, normalization="continuum"))
    graph_probability = float(system.probability(best_state, normalization="graph"))
    diagnostics = morphology_diagnostics(system.grid, best_state, args)
    state_path = state_file(args.output_dir, case)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        state_path,
        state=best_state.astype(np.uint8),
        grid=system.grid,
        jump_distance=case.jump_distance,
        eta=case.eta,
        sigma=case.sigma,
        grid_length=case.grid_length,
        num_spins=case.num_spins,
    )
    image_path = image_file(args.output_dir, case)
    save_state_png(system.grid, best_state, image_path)

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_key": case.key,
        "jump_distance": case.jump_distance,
        "grid_length": case.grid_length,
        "num_spins": case.num_spins,
        "eta": case.eta,
        "sigma": case.sigma,
        "run_index": case.run_index,
        "seed": deterministic_seed(args.seed, case.key, 1),
        "initialization": initialization,
        "probability": probability,
        "graph_probability": graph_probability,
        "state_path": str(state_path),
        "image_path": str(image_path),
        "elapsed_seconds": time.time() - started,
        **optimizer,
        **diagnostics,
    }
    atomic_write_json(record_path, record)
    return record


def record_file(output_dir: Path, case: TransitionCase) -> Path:
    return output_dir / "records_json" / f"{case.key}.json"


def state_file(output_dir: Path, case: TransitionCase) -> Path:
    return output_dir / "states" / f"{case.key}.npz"


def image_file(output_dir: Path, case: TransitionCase) -> Path:
    return output_dir / "images" / f"{case.key}.png"


def save_state_png(
    grid: NDArray[np.float64],
    state: NDArray[np.bool_],
    path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency path
        print(f"Skipping state PNG {path}: matplotlib is unavailable ({exc}).")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    occupied = grid[state]
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(occupied[:, 0], occupied[:, 1], s=1.0, c="black", linewidths=0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()
    axis.margins(0.02)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def atomic_write_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def best_by_eta(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[float, dict[str, Any]] = {}
    for record in records:
        eta = float(record["eta"])
        if eta not in best or float(record["probability"]) > float(best[eta]["probability"]):
            best[eta] = record
    return [best[eta] for eta in sorted(best)]


def detect_transitions(best_rows: Sequence[dict[str, Any]], stability_points: int) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    rows = list(sorted(best_rows, key=lambda row: float(row["eta"])))
    for index in range(1, len(rows)):
        before = rows[index - 1]
        after = rows[index]
        if before["morphology_label"] == after["morphology_label"]:
            continue
        stable_before = label_is_stable(rows, index - 1, -1, stability_points)
        stable_after = label_is_stable(rows, index, 1, stability_points)
        transitions.append(
            {
                "from_label": before["morphology_label"],
                "to_label": after["morphology_label"],
                "eta_low": float(before["eta"]),
                "eta_high": float(after["eta"]),
                "sigma_low": float(before["sigma"]),
                "sigma_high": float(after["sigma"]),
                "stable_before": stable_before,
                "stable_after": stable_after,
                "stable": bool(stable_before and stable_after),
                "from_probability": float(before["probability"]),
                "to_probability": float(after["probability"]),
                "from_dominant_mode": int(before["dominant_mode"]),
                "to_dominant_mode": int(after["dominant_mode"]),
                "from_anisotropy_strength": float(before["anisotropy_strength"]),
                "to_anisotropy_strength": float(after["anisotropy_strength"]),
            }
        )
    return transitions


def label_is_stable(
    rows: Sequence[dict[str, Any]],
    start: int,
    direction: int,
    stability_points: int,
) -> bool:
    label = rows[start]["morphology_label"]
    for offset in range(stability_points):
        index = start + direction * offset
        if index < 0 or index >= len(rows):
            return False
        if rows[index]["morphology_label"] != label:
            return False
    return True


def write_outputs(args: argparse.Namespace, records: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = list(sorted(records, key=lambda row: (float(row["eta"]), int(row["run_index"]))))
    best_rows = best_by_eta(all_rows)
    transitions = detect_transitions(best_rows, args.stability_points)

    fieldnames = sorted({key for row in all_rows for key in row})
    best_fieldnames = sorted({key for row in best_rows for key in row})
    atomic_write_csv(args.output_dir / "records.csv", all_rows, fieldnames)
    atomic_write_csv(args.output_dir / "best_by_eta.csv", best_rows, best_fieldnames)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "jump_distance": args.jump_distance,
        "grid_length": args.grid_length,
        "num_spins": args.num_spins,
        "eta_min": resolved_eta_min(args),
        "eta_min_source": (
            "explicit"
            if args.eta_min is not None
            else "cosine_kernel_gaussian_equivalent"
        ),
        "stability_points": args.stability_points,
        "transitions": transitions,
    }
    adaptive_path = args.output_dir / "adaptive_scan.json"
    if adaptive_path.exists():
        with adaptive_path.open("r", encoding="utf-8") as handle:
            payload["scan"] = json.load(handle)
    atomic_write_json(args.output_dir / "transitions.json", payload)
    if args.plot:
        write_quicklook_plot(args, best_rows)
    return best_rows, transitions


def write_quicklook_plot(args: argparse.Namespace, best_rows: Sequence[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency path
        print(f"Skipping plot: matplotlib is unavailable ({exc}).")
        return
    etas = [float(row["eta"]) for row in best_rows]
    probabilities = [float(row["probability"]) for row in best_rows]
    anisotropies = [float(row["anisotropy_strength"]) for row in best_rows]
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(etas, probabilities, marker="o")
    axes[0].set_ylabel("probability")
    axes[1].plot(etas, anisotropies, marker="o")
    axes[1].set_ylabel("anisotropy")
    axes[1].set_xlabel("eta = sigma / d")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "quicklook.png", dpi=180)
    plt.close(fig)


def scheduler_snapshot(
    args: argparse.Namespace,
    records: Sequence[dict[str, Any]],
    history: Sequence[dict[str, Any]],
    *,
    stop_reason: str | None,
) -> dict[str, Any]:
    summaries = summarize_etas(records)
    unresolved_boundaries = []
    unexplored_intervals = []
    for width, left, right in _intervals(summaries):
        interval = {
            "eta_low": left.eta,
            "eta_high": right.eta,
            "width": width,
            "low_label": left.best["morphology_label"],
            "high_label": right.best["morphology_label"],
        }
        is_boundary = (
            interval["low_label"] != interval["high_label"]
            or interval["low_label"] == "uncertain"
            or interval["high_label"] == "uncertain"
        )
        if is_boundary and width > args.eta_tolerance:
            unresolved_boundaries.append(interval)
        elif not is_boundary and width > args.exploration_width:
            unexplored_intervals.append(interval)

    return {
        "mode": "uniform" if args.uniform else "adaptive",
        "scan": {
            "jump_distance": args.jump_distance,
            "grid_length": args.grid_length,
            "num_spins": args.num_spins,
            "eta_min": resolved_eta_min(args),
            "eta_max": args.eta_max,
        },
        "settings": {
            "initial_eta_points": args.initial_eta_points,
            "max_eta_points": args.max_eta_points,
            "eta_tolerance": args.eta_tolerance,
            "exploration_width": args.exploration_width,
            "runs": args.runs,
            "max_runs_per_eta": args.max_runs_per_eta,
            "explore_every": args.explore_every,
        },
        "distinct_eta_count": len(summaries),
        "evaluation_count": len(records),
        "stop_reason": stop_reason,
        "unresolved_boundaries": unresolved_boundaries,
        "unexplored_intervals": unexplored_intervals,
        "ambiguous_etas": [
            {
                "eta": summary.eta,
                "labels": list(summary.labels),
                "runs": len(summary.records),
            }
            for summary in summaries.values()
            if summary.ambiguous
        ],
        "history": list(history),
    }


def write_scheduler_snapshot(
    args: argparse.Namespace,
    records: Sequence[dict[str, Any]],
    history: Sequence[dict[str, Any]],
    *,
    stop_reason: str | None,
) -> dict[str, Any]:
    snapshot = scheduler_snapshot(
        args,
        records,
        history,
        stop_reason=stop_reason,
    )
    atomic_write_json(args.output_dir / "adaptive_scan.json", snapshot)
    return snapshot


def scheduler_payload_matches(
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    scan = payload.get("scan", {})
    return (
        math.isclose(
            float(scan.get("jump_distance", math.nan)),
            float(args.jump_distance),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and int(scan.get("grid_length", -1)) == int(args.grid_length)
        and int(scan.get("num_spins", -1)) == int(args.num_spins)
        and math.isclose(
            float(scan.get("eta_min", math.nan)),
            resolved_eta_min(args),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(scan.get("eta_max", math.nan)),
            float(args.eta_max),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )


def load_scheduler_payload(args: argparse.Namespace) -> dict[str, Any] | None:
    path = args.output_dir / "adaptive_scan.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if scheduler_payload_matches(payload, args) else None


def load_scheduler_history(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = load_scheduler_payload(args)
    if payload is None:
        return []
    return list(payload.get("history", []))


def boundary_refinements_since_exploration(
    history: Sequence[dict[str, Any]],
) -> int:
    count = 0
    for entry in reversed(history):
        if entry.get("outcome") != "completed":
            continue
        reason = entry.get("reason")
        if reason in ("periodic_exploration", "interval_exploration"):
            break
        if reason == "boundary_refinement":
            count += 1
    return count


def remove_matching_scan(args: argparse.Namespace) -> None:
    for record_path in sorted((args.output_dir / "records_json").glob("*.json")):
        try:
            with record_path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not record_matches_scan(record, args):
            continue
        case_key = str(record.get("case_key", record_path.stem))
        for path in (
            record_path,
            args.output_dir / "states" / f"{case_key}.npz",
            args.output_dir / "images" / f"{case_key}.png",
        ):
            path.unlink(missing_ok=True)


def _record_decision(
    history: list[dict[str, Any]],
    decision: ScheduleDecision,
    *,
    outcome: str,
    error: str | None = None,
) -> None:
    entry = {
        "sequence": len(history),
        "eta": decision.eta,
        "run_index": decision.run_index,
        "reason": decision.reason,
        "outcome": outcome,
    }
    if error is not None:
        entry["error"] = error
    history.append(entry)


def run_uniform_scan(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    history = load_scheduler_history(args)
    known = {
        (round(float(record["eta"]), 10), int(record["run_index"]))
        for record in records
    }
    for eta in eta_grid(args):
        for run_index in range(args.runs):
            if (eta, run_index) in known:
                continue
            decision = ScheduleDecision(eta, run_index, "uniform_grid")
            case = TransitionCase(
                args.jump_distance,
                args.grid_length,
                args.num_spins,
                eta,
                run_index,
            )
            try:
                print(f"eta={eta:.6f} run={run_index + 1}/{args.runs}")
                records.append(run_case(args, case))
                _record_decision(history, decision, outcome="completed")
            except Exception as exc:
                _record_decision(
                    history,
                    decision,
                    outcome="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                if args.fail_fast:
                    raise
                print(
                    f"WARNING: failed eta={eta:.6f} run={run_index}",
                    file=sys.stderr,
                )
    write_scheduler_snapshot(
        args,
        records,
        history,
        stop_reason="uniform_grid_complete",
    )
    return write_outputs(args, records)


def run_adaptive_scan(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    history = load_scheduler_history(args)
    boundary_since_exploration = boundary_refinements_since_exploration(history)
    stop_reason = "converged"

    while True:
        decision = choose_adaptive_work(
            args,
            records,
            boundary_since_exploration=boundary_since_exploration,
        )
        if decision is None:
            if len(summarize_etas(records)) >= args.max_eta_points:
                stop_reason = "max_eta_points"
            break

        case = TransitionCase(
            args.jump_distance,
            args.grid_length,
            args.num_spins,
            decision.eta,
            decision.run_index,
        )
        print(
            f"eta={decision.eta:.6f} run={decision.run_index + 1} "
            f"reason={decision.reason}"
        )
        try:
            record = run_case(args, case)
            records.append(record)
            _record_decision(history, decision, outcome="completed")
        except Exception as exc:
            _record_decision(
                history,
                decision,
                outcome="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            if args.fail_fast:
                raise
            print(
                f"WARNING: failed eta={decision.eta:.6f} "
                f"run={decision.run_index}",
                file=sys.stderr,
            )
            stop_reason = "evaluation_failed"
            break

        if decision.reason == "boundary_refinement":
            boundary_since_exploration += 1
        elif decision.reason in ("periodic_exploration", "interval_exploration"):
            boundary_since_exploration = 0

        write_scheduler_snapshot(
            args,
            records,
            history,
            stop_reason=None,
        )
        write_outputs(args, records)

    write_scheduler_snapshot(
        args,
        records,
        history,
        stop_reason=stop_reason,
    )
    return write_outputs(args, records)


def run_transitions(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        remove_matching_scan(args)
        if load_scheduler_payload(args) is not None:
            (args.output_dir / "adaptive_scan.json").unlink(missing_ok=True)
    records = compatible_records(load_records(args.output_dir), args)
    if args.uniform:
        best_rows, transitions = run_uniform_scan(args, records)
    else:
        best_rows, transitions = run_adaptive_scan(args, records)

    print(f"Wrote {args.output_dir / 'records.csv'}")
    print(f"Wrote {args.output_dir / 'best_by_eta.csv'}")
    print(f"Wrote {args.output_dir / 'transitions.json'}")
    print(f"Wrote {args.output_dir / 'adaptive_scan.json'}")
    if transitions:
        for transition in transitions:
            marker = "stable" if transition["stable"] else "candidate"
            print(
                f"{marker}: {transition['from_label']} -> {transition['to_label']} "
                f"eta in [{transition['eta_low']:.6f}, {transition['eta_high']:.6f}]"
            )
    elif best_rows:
        print("No morphology label changes detected in this eta range.")


def load_records(output_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((output_dir / "records_json").glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    return records


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "transitions":
        run_transitions(args)


if __name__ == "__main__":
    main()
