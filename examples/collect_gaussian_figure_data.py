#!/usr/bin/env python3
"""Collect planar Grasshopper data for the Gaussian-broadening figures.

The script runs two complementary experiment families:

1. ``gaussian``: scan the physical width ``sigma / d`` at fixed occupied
   resolution ``K``. The point ``sigma / d = 0`` is evaluated with the compact
   cosine delta regularization and serves as the sharp-kernel reference.
2. ``resolution``: run the sharp cosine kernel at several values of ``K``.
   These data test the effective-width comparison

       sigma_num / d = sqrt(4/3 - 8/pi^2) / (d sqrt(K)).

Each independent optimization writes an atomic JSON record and a compressed
NumPy state artifact. The output directory also contains deduplicated grids,
``records.csv`` with every run, and ``best_by_case.csv`` with the highest
continuum probability found for each parameter point. Existing records are
skipped by default, so interrupted calculations can be resumed safely.

Example pilot run:

    python examples/collect_gaussian_figure_data.py \
        --scan all \
        --occupied-count 900 \
        --resolution-counts 400 625 900 1225 \
        --runs 4

The defaults are deliberately moderate. A dissertation-quality scan should
increase ``--occupied-count``, ``--runs``, and the optimizer effort after a
pilot calculation has identified the useful width range.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

cache_root = Path(tempfile.gettempdir()) / "grasshopper-gaussian-data-cache"
matplotlib_cache = cache_root / "matplotlib"
xdg_cache = cache_root / "xdg"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

import grasshopper as gh


NUMERICAL_WIDTH_FACTOR = math.sqrt(4.0 / 3.0 - 8.0 / math.pi**2)
DEFAULT_REGIMES = (
    ("cogwheel", 0.40),
    ("critical", 1.0 / math.sqrt(math.pi)),
)
DEFAULT_ETAS = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60)
DEFAULT_RESOLUTION_COUNTS = (400, 625, 900, 1225)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Regime:
    name: str
    jump_distance: float


@dataclass(frozen=True)
class ExperimentCase:
    scan: str
    regime: str
    jump_distance: float
    occupied_count: int
    eta: float
    sigma: float
    run_index: int

    @property
    def key(self) -> str:
        return (
            f"{self.scan}__{slug(self.regime)}"
            f"__d{self.jump_distance:.9f}"
            f"__K{self.occupied_count}"
            f"__eta{self.eta:.9f}"
            f"__run{self.run_index:03d}"
        )


@dataclass(frozen=True)
class LatticeShape:
    width: int
    height: int
    candidate_count: int


@dataclass
class OptimizationSummary:
    thermalized_temperature: float | None = None
    anneal_iterations: int = 0
    anneal_mean_acceptance: float | None = None
    tempering_iterations: int = 0
    tempering_mean_acceptance: float | None = None
    tempering_mean_exchange_rate: float | None = None
    greedy_steps: int = 0
    greedy_converged: bool | None = None


def slug(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value.strip().lower()
    )
    return cleaned.strip("-") or "case"


def parse_regime(value: str) -> Regime:
    try:
        name, distance_text = value.rsplit(":", 1)
        distance = float(distance_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "Regimes must have the form NAME:JUMP_DISTANCE."
        ) from exc
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("A regime name cannot be empty.")
    if not math.isfinite(distance) or distance <= 0:
        raise argparse.ArgumentTypeError(
            "A regime jump distance must be positive and finite."
        )
    return Regime(name=name, jump_distance=distance)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(
            "Expected a finite, nonnegative number."
        )
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect optimized states and diagnostics for the planar "
            "Gaussian-broadening and resolution-collapse figures."
        )
    )
    parser.add_argument(
        "--scan",
        choices=("all", "gaussian", "resolution"),
        default="all",
        help="Experiment family to run.",
    )
    parser.add_argument(
        "--regime",
        action="append",
        type=parse_regime,
        default=None,
        metavar="NAME:D",
        help=(
            "Named jump-distance regime. May be repeated. Defaults to "
            "cogwheel:0.40 and critical:1/sqrt(pi)."
        ),
    )
    parser.add_argument(
        "--etas",
        nargs="+",
        type=nonnegative_float,
        default=list(DEFAULT_ETAS),
        help="Physical Gaussian widths sigma/d for the Gaussian scan.",
    )
    parser.add_argument(
        "--occupied-count",
        type=positive_int,
        default=900,
        help="Occupied-spin count K for the fixed-resolution Gaussian scan.",
    )
    parser.add_argument(
        "--resolution-counts",
        nargs="+",
        type=positive_int,
        default=list(DEFAULT_RESOLUTION_COUNTS),
        help="Occupied-spin counts K for the sharp-kernel resolution scan.",
    )
    parser.add_argument(
        "--candidate-factor",
        type=float,
        default=4.0,
        help=(
            "Requested ratio M/K of candidate to occupied sites. "
            "The realized ratio is reported exactly."
        ),
    )
    parser.add_argument(
        "--lattice",
        choices=("square", "rectangular", "triangular", "hexagonal", "honeycomb"),
        default="square",
    )
    parser.add_argument("--runs", type=positive_int, default=4)
    parser.add_argument(
        "--gaussian-cutoff",
        type=float,
        default=5.0,
        help="Gaussian truncation in standard deviations.",
    )
    parser.add_argument(
        "--unbounded-gaussian",
        action="store_true",
        help="Use an exact all-pairs Gaussian instead of finite cutoff pruning.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("anneal", "parallel-tempering"),
        default="anneal",
    )
    parser.add_argument(
        "--initialization",
        choices=("mixed", "random", "disk", "disk-jitter", "annulus"),
        default="mixed",
        help=(
            "Initialization family. 'mixed' cycles through disk, disk-jitter, "
            "annulus, and random seeds across independent runs."
        ),
    )
    parser.add_argument(
        "--jitter-fraction",
        type=nonnegative_float,
        default=0.08,
        help="Fraction of occupied sites swapped in disk-jitter initialization.",
    )
    parser.add_argument(
        "--annulus-radius",
        type=nonnegative_float,
        default=0.55,
        help="Target radius used only to seed annular candidates.",
    )
    parser.add_argument(
        "--thermalize-steps",
        type=positive_int,
        default=None,
        help="Steps per adaptive temperature probe; defaults to 5K.",
    )
    parser.add_argument(
        "--thermalize-adjustments",
        type=int,
        default=10,
        help="Set to zero to skip adaptive initial-temperature calibration.",
    )
    parser.add_argument(
        "--target-acceptance",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--sweep-steps",
        type=positive_int,
        default=None,
        help="Metropolis proposals per annealing/PT sweep; defaults to 20K.",
    )
    parser.add_argument("--anneal-iterations", type=positive_int, default=60)
    parser.add_argument("--final-temperature", type=nonnegative_float, default=0.01)
    parser.add_argument("--min-acceptance", type=nonnegative_float, default=1e-5)
    parser.add_argument("--max-runtime-hours", type=float, default=4.0)
    parser.add_argument("--replicas", type=positive_int, default=12)
    parser.add_argument("--tempering-iterations", type=positive_int, default=60)
    parser.add_argument("--cold-temperature", type=float, default=0.01)
    parser.add_argument("--exchange-interval", type=positive_int, default=1)
    parser.add_argument("--workers", type=positive_int, default=1)
    parser.add_argument(
        "--greedy-steps",
        type=int,
        default=5_000,
        help="Maximum zero-temperature refinement steps; set to zero to skip.",
    )
    parser.add_argument("--m-max", type=positive_int, default=32)
    parser.add_argument(
        "--radial-bin-width",
        type=float,
        default=None,
        help="Radial profile width. Defaults to the physical cell length h.",
    )
    parser.add_argument(
        "--radial-smoothing-width",
        type=float,
        default=None,
        help=(
            "Gaussian smoothing width for the radial mean used in Q_m. "
            "Defaults to 2h. The unsmoothed profile is also saved."
        ),
    )
    parser.add_argument(
        "--inner-density-threshold",
        type=float,
        default=0.50,
        help="Radial-density threshold used for the reported inner radius.",
    )
    parser.add_argument(
        "--inner-sustain-bins",
        type=positive_int,
        default=3,
        help="Consecutive populated radial bins required at the threshold.",
    )
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/output/gaussian_figure_data"),
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Also save the library's NetCDF checkpoint for every completed run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute cases whose JSON records already exist.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return args


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if not math.isfinite(args.candidate_factor) or args.candidate_factor <= 1.0:
        parser.error("--candidate-factor must be finite and greater than 1.")
    if (
        not args.unbounded_gaussian
        and (
            not math.isfinite(args.gaussian_cutoff)
            or args.gaussian_cutoff <= 0
        )
    ):
        parser.error("--gaussian-cutoff must be positive and finite.")
    if args.thermalize_adjustments < 0:
        parser.error("--thermalize-adjustments cannot be negative.")
    if args.greedy_steps < 0:
        parser.error("--greedy-steps cannot be negative.")
    if not 0 < args.target_acceptance < 1:
        parser.error("--target-acceptance must lie strictly between zero and one.")
    if args.cold_temperature <= 0 or not math.isfinite(args.cold_temperature):
        parser.error("--cold-temperature must be positive and finite.")
    if args.max_runtime_hours <= 0 or not math.isfinite(args.max_runtime_hours):
        parser.error("--max-runtime-hours must be positive and finite.")
    if args.radial_bin_width is not None and (
        not math.isfinite(args.radial_bin_width)
        or args.radial_bin_width <= 0
    ):
        parser.error("--radial-bin-width must be positive and finite.")
    if args.radial_smoothing_width is not None and (
        not math.isfinite(args.radial_smoothing_width)
        or args.radial_smoothing_width <= 0
    ):
        parser.error("--radial-smoothing-width must be positive and finite.")
    if not 0 < args.inner_density_threshold <= 1:
        parser.error("--inner-density-threshold must lie in (0, 1].")
    if any(not math.isfinite(eta) for eta in args.etas):
        parser.error("--etas must contain only finite values.")
    args.etas = sorted(set(float(value) for value in args.etas))
    args.resolution_counts = sorted(set(int(value) for value in args.resolution_counts))


def default_regimes() -> list[Regime]:
    return [
        Regime(name=name, jump_distance=distance)
        for name, distance in DEFAULT_REGIMES
    ]


def build_cases(
    *,
    scan: str,
    regimes: Sequence[Regime],
    etas: Sequence[float],
    occupied_count: int,
    resolution_counts: Sequence[int],
    runs: int,
) -> list[ExperimentCase]:
    cases: list[ExperimentCase] = []
    if scan in ("all", "gaussian"):
        for regime in regimes:
            for eta in etas:
                for run_index in range(runs):
                    cases.append(
                        ExperimentCase(
                            scan="gaussian",
                            regime=regime.name,
                            jump_distance=regime.jump_distance,
                            occupied_count=occupied_count,
                            eta=float(eta),
                            sigma=float(eta * regime.jump_distance),
                            run_index=run_index,
                        )
                    )
    if scan in ("all", "resolution"):
        for regime in regimes:
            for count in resolution_counts:
                for run_index in range(runs):
                    cases.append(
                        ExperimentCase(
                            scan="resolution",
                            regime=regime.name,
                            jump_distance=regime.jump_distance,
                            occupied_count=int(count),
                            eta=0.0,
                            sigma=0.0,
                            run_index=run_index,
                        )
                    )
    return cases


def lattice_shape(
    occupied_count: int,
    candidate_factor: float,
    lattice: str,
) -> LatticeShape:
    target_sites = max(occupied_count + 1, math.ceil(candidate_factor * occupied_count))
    sites_per_cell = 2 if lattice in ("hexagonal", "honeycomb") else 1
    target_cells = math.ceil(target_sites / sites_per_cell)
    width = math.ceil(math.sqrt(target_cells))
    height = math.ceil(target_cells / width)
    candidate_count = width * height * sites_per_cell
    return LatticeShape(
        width=width,
        height=height,
        candidate_count=candidate_count,
    )


def build_system(
    case: ExperimentCase,
    args: argparse.Namespace,
) -> tuple[gh.System, LatticeShape]:
    shape = lattice_shape(
        case.occupied_count,
        args.candidate_factor,
        args.lattice,
    )
    system = gh.System.regular_lattice(
        args.lattice,
        shape.width,
        shape.height,
        num_spins=case.occupied_count,
        center=True,
    )
    cutoff = None if args.unbounded_gaussian else float(args.gaussian_cutoff)
    if case.sigma > 0:
        kernel = gh.KernelSpec.gaussian(case.sigma, cutoff=cutoff)
    else:
        kernel = gh.KernelSpec.cosine()
    system.compute_neighborhood(
        jump_distance=case.jump_distance,
        kernel=kernel,
        distance_metric="euclidean",
        silent=not args.verbose,
    )
    return system, shape


def deterministic_seed(base_seed: int, case_key: str, stream: int) -> int:
    digest = hashlib.blake2b(
        case_key.encode("utf-8"),
        digest_size=8,
        person=b"gh-gauss",
    ).digest()
    case_number = int.from_bytes(digest, byteorder="little", signed=False)
    sequence = np.random.SeedSequence([int(base_seed), case_number, int(stream)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def disk_state(
    grid: NDArray[np.float64],
    occupied_count: int,
) -> NDArray[np.bool_]:
    radii_squared = np.einsum("ij,ij->i", grid, grid)
    selected = np.argpartition(radii_squared, occupied_count - 1)[:occupied_count]
    state = np.zeros(len(grid), dtype=np.bool_)
    state[selected] = True
    return state


def annulus_state(
    grid: NDArray[np.float64],
    occupied_count: int,
    target_radius: float,
) -> NDArray[np.bool_]:
    radii = np.linalg.norm(grid, axis=1)
    scores = np.abs(radii - target_radius)
    selected = np.argpartition(scores, occupied_count - 1)[:occupied_count]
    state = np.zeros(len(grid), dtype=np.bool_)
    state[selected] = True
    return state


def jitter_state(
    state: NDArray[np.bool_],
    fraction: float,
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    jittered = np.asarray(state, dtype=np.bool_).copy()
    occupied = np.flatnonzero(jittered)
    unoccupied = np.flatnonzero(~jittered)
    swaps = min(
        len(occupied),
        len(unoccupied),
        int(round(fraction * len(occupied))),
    )
    if swaps <= 0:
        return jittered
    remove = rng.choice(occupied, size=swaps, replace=False)
    add = rng.choice(unoccupied, size=swaps, replace=False)
    jittered[remove] = False
    jittered[add] = True
    return jittered


def initialization_for_run(requested: str, run_index: int) -> str:
    if requested != "mixed":
        return requested
    cycle = ("disk", "disk-jitter", "annulus", "random")
    return cycle[run_index % len(cycle)]


def initialize_system(
    system: gh.System,
    *,
    kind: str,
    seed: int,
    jitter_fraction: float,
    annulus_radius: float,
) -> None:
    if kind == "random":
        system.random_init(seed=seed)
        return
    if kind == "disk":
        system.array_init(disk_state(system.grid, system.num_spins))
        return
    if kind == "disk-jitter":
        rng = np.random.default_rng(seed)
        state = jitter_state(
            disk_state(system.grid, system.num_spins),
            jitter_fraction,
            rng,
        )
        system.array_init(state)
        return
    if kind == "annulus":
        system.array_init(
            annulus_state(system.grid, system.num_spins, annulus_radius)
        )
        return
    raise ValueError(f"Unsupported initialization {kind!r}.")


def cooling_rate(start: float, final: float, iterations: int) -> float:
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if start <= 0:
        raise ValueError("start must be positive.")
    effective_final = min(max(final, 1e-12), start * (1.0 - 1e-9))
    return float((effective_final / start) ** (1.0 / iterations))


def optimize_system(
    system: gh.System,
    args: argparse.Namespace,
    case: ExperimentCase,
) -> OptimizationSummary:
    summary = OptimizationSummary()
    thermalize_steps = (
        args.thermalize_steps
        if args.thermalize_steps is not None
        else max(100, 5 * case.occupied_count)
    )
    sweep_steps = (
        args.sweep_steps
        if args.sweep_steps is not None
        else max(100, 20 * case.occupied_count)
    )

    if args.thermalize_adjustments > 0:
        summary.thermalized_temperature = system.thermalize(
            target_acceptance=args.target_acceptance,
            steps_per_temp=thermalize_steps,
            max_adjustments=args.thermalize_adjustments,
            verbose=args.verbose,
            seed=deterministic_seed(args.seed, case.key, 1),
        )
    else:
        summary.thermalized_temperature = float(system.temperature)

    if args.optimizer == "anneal":
        start_temp = float(system.temperature)
        rate = cooling_rate(
            start_temp,
            args.final_temperature,
            args.anneal_iterations,
        )
        system.anneal(
            starting_temp=start_temp,
            final_temp=min(args.final_temperature, start_temp * (1.0 - 1e-9)),
            cooling_rate=rate,
            min_acceptance=args.min_acceptance,
            max_runtime_hours=args.max_runtime_hours,
            max_iterations=args.anneal_iterations,
            steps=sweep_steps,
            verbose=args.verbose,
            reinit=False,
            seed=deterministic_seed(args.seed, case.key, 2),
        )
        records = system.anneal_data.records
        summary.anneal_iterations = len(records)
        if records:
            summary.anneal_mean_acceptance = float(
                np.mean([record.acceptance_rate for record in records])
            )
    else:
        hot_temp = max(float(system.temperature), args.cold_temperature * 1.01)
        result = system.parallel_temper(
            num_replicas=args.replicas,
            iterations=args.tempering_iterations,
            sweep_steps=sweep_steps,
            exchange_interval=args.exchange_interval,
            cold_temp=args.cold_temperature,
            hot_temp=hot_temp,
            reinit=False,
            track_best=True,
            seed=deterministic_seed(args.seed, case.key, 2),
            n_workers=args.workers,
            verbose=args.verbose,
        )
        summary.tempering_iterations = int(result.iterations)
        if result.sweep_acceptance:
            summary.tempering_mean_acceptance = float(
                np.mean(np.asarray(result.sweep_acceptance, dtype=float))
            )
        if result.exchange_rates:
            summary.tempering_mean_exchange_rate = float(
                np.mean(np.asarray(result.exchange_rates, dtype=float))
            )

    # Refine from the best configuration, not merely the final chain state.
    best_state = np.asarray(system.best_spins, dtype=np.bool_).copy()
    system.array_init(best_state)
    if args.greedy_steps > 0:
        result = system.greedy(
            max_steps=args.greedy_steps,
            min_delta=0.0,
            track_best=True,
            reinit=True,
            verbose=args.verbose,
        )
        summary.greedy_steps = int(result.steps)
        summary.greedy_converged = bool(result.converged)

    # array_init performs an exact final energy reconstruction.
    system.array_init(np.asarray(system.best_spins, dtype=np.bool_).copy())
    return summary


def radial_profile(
    radii: NDArray[np.float64],
    occupancy: NDArray[np.bool_],
    bin_width: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    indices = np.floor(radii / bin_width).astype(np.int64)
    bin_count = int(indices.max(initial=0)) + 1
    counts = np.bincount(indices, minlength=bin_count).astype(np.int64)
    occupied = np.bincount(
        indices,
        weights=occupancy.astype(float),
        minlength=bin_count,
    )
    density = np.full(bin_count, np.nan, dtype=float)
    populated = counts > 0
    density[populated] = occupied[populated] / counts[populated]
    centers = (np.arange(bin_count, dtype=float) + 0.5) * bin_width
    return centers, density, counts, indices


def smooth_radial_density(
    density: NDArray[np.float64],
    counts: NDArray[np.int64],
    *,
    bin_width: float,
    smoothing_width: float,
) -> NDArray[np.float64]:
    occupied_counts = np.nan_to_num(density, nan=0.0) * counts
    sigma_bins = smoothing_width / bin_width
    smoothed_counts = gaussian_filter1d(
        counts.astype(float),
        sigma=sigma_bins,
        mode="nearest",
    )
    smoothed_occupied = gaussian_filter1d(
        occupied_counts,
        sigma=sigma_bins,
        mode="nearest",
    )
    smoothed = np.zeros_like(smoothed_counts)
    populated = smoothed_counts > np.finfo(float).eps
    smoothed[populated] = smoothed_occupied[populated] / smoothed_counts[populated]
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
    if populated.size == 0:
        return float("nan")

    for start_position, bin_index in enumerate(populated):
        window = populated[start_position : start_position + sustain_bins]
        if window.size < sustain_bins:
            break
        if np.any(np.diff(window) != 1):
            continue
        if np.all(density[window] >= threshold):
            if bin_index == populated[0]:
                return 0.0
            return float(bin_index * bin_width)
    return float("nan")


def angular_modes(
    angles: NDArray[np.float64],
    radial_indices: NDArray[np.int64],
    radial_density: NDArray[np.float64],
    occupancy: NDArray[np.bool_],
    m_max: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    fluctuation = occupancy.astype(float) - radial_density[radial_indices]
    denominator = float(np.sum(np.abs(fluctuation)))
    modes = np.arange(2, m_max + 1, dtype=np.int64)
    if denominator <= np.finfo(float).eps:
        return modes, np.zeros_like(modes, dtype=float)
    values = np.array(
        [
            abs(np.sum(fluctuation * np.exp(1j * mode * angles))) / denominator
            for mode in modes
        ],
        dtype=float,
    )
    return modes, values


def boundary_margin(
    grid: NDArray[np.float64],
    occupancy: NDArray[np.bool_],
) -> float:
    occupied_points = grid[occupancy]
    lower = np.min(grid, axis=0)
    upper = np.max(grid, axis=0)
    margins = np.concatenate(
        (
            np.min(occupied_points - lower, axis=0),
            np.min(upper - occupied_points, axis=0),
        )
    )
    return float(np.min(margins))


def morphology_diagnostics(
    grid: NDArray[np.float64],
    occupancy: NDArray[np.bool_],
    *,
    jump_distance: float,
    cell_length: float,
    m_max: int,
    radial_bin_width: float | None,
    radial_smoothing_width: float | None,
    inner_density_threshold: float,
    inner_sustain_bins: int,
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    occupancy = np.asarray(occupancy, dtype=np.bool_)
    occupied_points = grid[occupancy]
    if occupied_points.size == 0:
        raise ValueError("Cannot diagnose an empty lawn.")

    center = np.mean(occupied_points, axis=0)
    relative = grid - center
    radii = np.linalg.norm(relative, axis=1)
    angles = np.arctan2(relative[:, 1], relative[:, 0])
    width = cell_length if radial_bin_width is None else radial_bin_width
    smoothing_width = (
        2.0 * cell_length
        if radial_smoothing_width is None
        else radial_smoothing_width
    )
    centers, raw_density, counts, radial_indices = radial_profile(
        radii,
        occupancy,
        width,
    )
    density = smooth_radial_density(
        raw_density,
        counts,
        bin_width=width,
        smoothing_width=smoothing_width,
    )
    modes, mode_values = angular_modes(
        angles,
        radial_indices,
        density,
        occupancy,
        m_max,
    )
    maximum_index = int(np.argmax(mode_values))
    dominant_mode = int(modes[maximum_index])
    anisotropy = float(mode_values[maximum_index])
    sorted_values = np.sort(mode_values)
    second_mode_value = (
        float(sorted_values[-2]) if len(sorted_values) > 1 else 0.0
    )
    mode_sum = float(np.sum(mode_values))
    inner_radius = estimate_inner_radius(
        density,
        counts,
        width,
        threshold=inner_density_threshold,
        sustain_bins=inner_sustain_bins,
    )
    occupied_radii = radii[occupancy]
    covariance = np.cov(relative[occupancy].T, bias=True)
    eigenvalues = np.linalg.eigvalsh(covariance)
    inertia_denominator = float(np.sum(eigenvalues))
    inertia_anisotropy = (
        0.0
        if inertia_denominator <= np.finfo(float).eps
        else float((eigenvalues[-1] - eigenvalues[0]) / inertia_denominator)
    )

    metrics: dict[str, Any] = {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "radial_bin_width": float(width),
        "radial_smoothing_width": float(smoothing_width),
        "anisotropy_q": anisotropy,
        "dominant_mode": dominant_mode,
        "second_mode_q": second_mode_value,
        "mode_prominence": (
            None
            if second_mode_value <= np.finfo(float).eps
            else float(anisotropy / second_mode_value)
        ),
        "mode_concentration": (
            0.0 if mode_sum <= np.finfo(float).eps else float(anisotropy / mode_sum)
        ),
        "inner_radius": finite_or_none(inner_radius),
        "inner_radius_over_d": finite_or_none(inner_radius / jump_distance),
        "occupied_radius_mean": float(np.mean(occupied_radii)),
        "occupied_radius_std": float(np.std(occupied_radii)),
        "occupied_radius_q01": float(np.quantile(occupied_radii, 0.01)),
        "occupied_radius_q99": float(np.quantile(occupied_radii, 0.99)),
        "inertia_anisotropy": inertia_anisotropy,
        "boundary_margin": boundary_margin(grid, occupancy),
    }
    for mode, value in zip(modes, mode_values, strict=True):
        metrics[f"q_{int(mode)}"] = float(value)

    arrays: dict[str, NDArray[Any]] = {
        "center": np.asarray(center, dtype=np.float64),
        "radial_bin_centers": centers,
        "radial_density": density,
        "radial_density_raw": raw_density,
        "radial_counts": counts,
        "q_modes": modes,
        "q_values": mode_values,
    }
    return metrics, arrays


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_save_npy(path: Path, array: NDArray[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **arrays: NDArray[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def write_invocation(
    output_dir: Path,
    args: argparse.Namespace,
    regimes: Sequence[Regime],
    case_count: int,
) -> None:
    serializable_args = vars(args).copy()
    serializable_args["output_dir"] = str(args.output_dir)
    if serializable_args.get("regime") is not None:
        serializable_args["regime"] = [
            asdict(regime) for regime in serializable_args["regime"]
        ]
    invocation = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": int(case_count),
        "regimes": [asdict(regime) for regime in regimes],
        "arguments": serializable_args,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "grasshopper": gh.__version__,
            "git_revision": git_revision(),
        },
        "notes": {
            "sigma_num_definition": (
                "h * sqrt(4/3 - 8/pi^2), with h = K^(-1/2)"
            ),
            "sigma_eff_definition": "sqrt(sigma^2 + sigma_num^2)",
            "sigma_eff_caveat": (
                "The library samples the physical Gaussian directly rather "
                "than convolving it with the compact cosine kernel. "
                "sigma_eff is stored as a variance-based comparison coordinate, "
                "not as the literal implemented kernel width."
            ),
        },
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    atomic_write_json(output_dir / "invocations" / f"{timestamp}.json", invocation)
    atomic_write_json(output_dir / "latest_invocation.json", invocation)


def load_records(record_dir: Path) -> list[dict[str, Any]]:
    records = []
    if not record_dir.exists():
        return records
    for path in sorted(record_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    return records


def write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    fieldnames = sorted({key for record in records for key in record})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def best_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    group_fields = (
        "scan",
        "regime",
        "jump_distance",
        "occupied_count",
        "eta",
        "sigma",
    )
    for record in records:
        key = tuple(record[field] for field in group_fields)
        incumbent = grouped.get(key)
        if incumbent is None or record["continuum_probability"] > incumbent[
            "continuum_probability"
        ]:
            grouped[key] = record
    return sorted(
        grouped.values(),
        key=lambda record: (
            record["scan"],
            record["regime"],
            record["occupied_count"],
            record["eta"],
        ),
    )


def save_grid_once(
    system: gh.System,
    case: ExperimentCase,
    shape: LatticeShape,
    output_dir: Path,
    lattice: str,
) -> Path:
    grid_id = (
        f"{slug(lattice)}__K{case.occupied_count}"
        f"__{shape.width}x{shape.height}__M{len(system.grid)}"
    )
    path = output_dir / "grids" / f"{grid_id}.npy"
    if not path.exists():
        atomic_save_npy(path, np.asarray(system.grid, dtype=np.float64))
    return path


def collect_case(
    case: ExperimentCase,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    record_path = output_dir / "records" / f"{case.key}.json"
    if record_path.exists() and not args.overwrite:
        with record_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    started = time.perf_counter()
    system, shape = build_system(case, args)
    grid_path = save_grid_once(
        system,
        case,
        shape,
        output_dir,
        args.lattice,
    )
    init_kind = initialization_for_run(args.initialization, case.run_index)
    init_seed = deterministic_seed(args.seed, case.key, 0)
    initialize_system(
        system,
        kind=init_kind,
        seed=init_seed,
        jitter_fraction=args.jitter_fraction,
        annulus_radius=args.annulus_radius,
    )
    optimization = optimize_system(system, args, case)

    state = np.asarray(system.best_spins, dtype=np.bool_).copy()
    cell_length = case.occupied_count ** -0.5
    diagnostics, diagnostic_arrays = morphology_diagnostics(
        system.grid,
        state,
        jump_distance=case.jump_distance,
        cell_length=cell_length,
        m_max=args.m_max,
        radial_bin_width=args.radial_bin_width,
        radial_smoothing_width=args.radial_smoothing_width,
        inner_density_threshold=args.inner_density_threshold,
        inner_sustain_bins=args.inner_sustain_bins,
    )

    sigma_num = NUMERICAL_WIDTH_FACTOR * cell_length
    sigma_eff = math.sqrt(case.sigma**2 + sigma_num**2)
    state_path = output_dir / "states" / f"{case.key}.npz"
    atomic_save_npz(
        state_path,
        best_spins=state.astype(np.uint8),
        **diagnostic_arrays,
    )
    checkpoint_path: Path | None = None
    if args.save_checkpoints:
        checkpoint_path = output_dir / "checkpoints" / f"{case.key}.nc"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        system.save(checkpoint_path)

    elapsed = time.perf_counter() - started
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "case_key": case.key,
        "scan": case.scan,
        "regime": case.regime,
        "jump_distance": float(case.jump_distance),
        "eta": float(case.eta),
        "sigma": float(case.sigma),
        "sigma_over_d": float(case.sigma / case.jump_distance),
        "kernel_kind": system.kernel.kind,
        "gaussian_cutoff": (
            None
            if system.kernel.kind != "gaussian" or system.kernel.cutoff is None
            else float(system.kernel.cutoff)
        ),
        "occupied_count": int(case.occupied_count),
        "candidate_count": int(len(system.grid)),
        "candidate_factor": float(len(system.grid) / case.occupied_count),
        "lattice": args.lattice,
        "lattice_width": int(shape.width),
        "lattice_height": int(shape.height),
        "cell_length": float(cell_length),
        "sigma_num": float(sigma_num),
        "sigma_num_over_d": float(sigma_num / case.jump_distance),
        "sigma_eff": float(sigma_eff),
        "sigma_eff_over_d": float(sigma_eff / case.jump_distance),
        "sigma_eff_is_comparison_only": True,
        "run_index": int(case.run_index),
        "seed": int(init_seed),
        "initialization": init_kind,
        "optimizer": args.optimizer,
        "energy": float(system.best_energy),
        "continuum_probability": float(system.best_probability),
        "graph_probability": float(system.probability("best", normalization="graph")),
        "runtime_seconds": float(elapsed),
        "grid_path": relative_path(grid_path, output_dir),
        "state_path": relative_path(state_path, output_dir),
        "checkpoint_path": (
            None
            if checkpoint_path is None
            else relative_path(checkpoint_path, output_dir)
        ),
        **asdict(optimization),
        **diagnostics,
    }
    record["boundary_margin_over_h"] = float(
        record["boundary_margin"] / cell_length
    )
    record["boundary_margin_over_d"] = float(
        record["boundary_margin"] / case.jump_distance
    )
    atomic_write_json(record_path, record)
    return record


def print_case_header(index: int, total: int, case: ExperimentCase) -> None:
    print(
        f"[{index:04d}/{total:04d}] "
        f"{case.scan} {case.regime}: d={case.jump_distance:.6g}, "
        f"K={case.occupied_count}, eta={case.eta:.6g}, "
        f"run={case.run_index}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    regimes = args.regime if args.regime is not None else default_regimes()
    cases = build_cases(
        scan=args.scan,
        regimes=regimes,
        etas=args.etas,
        occupied_count=args.occupied_count,
        resolution_counts=args.resolution_counts,
        runs=args.runs,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_invocation(output_dir, args, regimes, len(cases))

    failures = 0
    for index, case in enumerate(cases, start=1):
        print_case_header(index, len(cases), case)
        record_path = output_dir / "records" / f"{case.key}.json"
        resumed = record_path.exists() and not args.overwrite
        try:
            record = collect_case(case, args, output_dir)
        except Exception as exc:
            failures += 1
            failure = {
                "schema_version": SCHEMA_VERSION,
                "case_key": case.key,
                "case": asdict(case),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(
                output_dir / "failures" / f"{case.key}.json",
                failure,
            )
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        if resumed:
            print("  existing record reused")
        print(
            "  "
            f"p={record['continuum_probability']:.8g}, "
            f"Q={record['anisotropy_q']:.5g}, "
            f"m*={record['dominant_mode']}, "
            f"Ri/d={record['inner_radius_over_d']}, "
            f"time={record['runtime_seconds']:.1f}s"
        )
        if record["boundary_margin_over_h"] < 3.0:
            print(
                "  WARNING: occupied sites approach the candidate-domain "
                "boundary; increase --candidate-factor.",
                file=sys.stderr,
            )

    records = load_records(output_dir / "records")
    write_csv(output_dir / "records.csv", records)
    write_csv(output_dir / "best_by_case.csv", best_records(records))
    print(
        f"Collected {len(records)} records in {output_dir}. "
        f"Failures in this invocation: {failures}."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
