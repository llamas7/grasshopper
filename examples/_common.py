"""Shared utilities for the runnable Grasshopper examples."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

cache_root = Path(tempfile.gettempdir()) / "grasshopper-example-cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import grasshopper as gh


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer.")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive finite number.")
    return parsed


def fraction(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("Expected a number in (0, 1].")
    return parsed


def add_optimization_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_steps: int = 2_000,
    default_iterations: int = 6,
    default_greedy_steps: int = 500,
) -> None:
    parser.add_argument("--steps", type=positive_int, default=default_steps)
    parser.add_argument(
        "--anneal-iterations",
        type=nonnegative_int,
        default=default_iterations,
    )
    parser.add_argument(
        "--greedy-steps",
        type=nonnegative_int,
        default=default_greedy_steps,
    )
    parser.add_argument("--starting-temp", type=positive_float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--show", action="store_true")


def paired_fibonacci_sphere(num_pairs: int) -> np.ndarray:
    """Return an antipodally ordered, approximately uniform spherical grid."""
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive.")
    indices = np.arange(num_pairs, dtype=float)
    z = (indices + 0.5) / num_pairs
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    azimuth = golden_angle * indices
    first_half = np.column_stack(
        (radius * np.cos(azimuth), radius * np.sin(azimuth), z)
    )
    return np.ascontiguousarray(np.vstack((first_half, -first_half)))


def cubic_grid(side: int) -> np.ndarray:
    """Return a centered cubic candidate grid with ``side**3`` sites."""
    if side <= 0:
        raise ValueError("side must be positive.")
    coordinates = np.linspace(-0.5, 0.5, side)
    mesh = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    return np.column_stack([axis.ravel() for axis in mesh])


def optimize(
    system: gh.System,
    *,
    steps: int,
    anneal_iterations: int,
    greedy_steps: int,
    starting_temp: float,
    seed: int,
) -> None:
    """Run a compact deterministic optimization suitable for examples."""
    if anneal_iterations:
        system.anneal(
            starting_temp=starting_temp,
            final_temp=max(starting_temp * 0.05, 1e-8),
            cooling_rate=0.65,
            min_acceptance=0.0,
            max_runtime_hours=0.1,
            max_iterations=anneal_iterations,
            steps=steps,
            reinit=True,
            seed=seed,
        )
    if greedy_steps:
        system.greedy(
            max_steps=greedy_steps,
            reinit=True,
        )


def report(system: gh.System, *, title: str) -> dict[str, float]:
    values = {
        "current_energy": float(system.current_energy),
        "best_energy": float(system.best_energy),
        "current_probability": float(system.current_probability),
        "best_probability": float(system.best_probability),
        "current_graph_probability": float(system.probability(normalization="graph")),
        "best_graph_probability": float(system.probability("best", normalization="graph")),
    }
    print(title)
    print(f"  sites: {len(system.grid)}")
    print(f"  stored spins: {system.num_spins}")
    print(f"  kernel: {system.kernel.kind}")
    print(f"  best energy: {values['best_energy']:.6g}")
    print(f"  best continuum probability: {values['best_probability']:.6g}")
    print(f"  best graph probability: {values['best_graph_probability']:.6g}")
    return values


def save_spherical_plot(
    system: gh.System,
    path: Path,
    *,
    title: str,
    show: bool,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = system.plot(plot_figure=False)
    figure.update_layout(title=title)
    figure.write_html(path)
    if show:
        figure.show()
    return path


def save_euclidean_plot(
    system: gh.System,
    path: Path,
    *,
    title: str,
    show: bool,
    state: str = "best",
    voronoi: bool = False,
    show_edges: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = system.plot(
        state=state,
        voronoi=voronoi,
        show_edges=show_edges,
        plot_figure=False,
    )
    figure.axes[0].set_title(title)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(figure)
    return path

