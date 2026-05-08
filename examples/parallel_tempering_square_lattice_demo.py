"""Demonstrate parallel tempering on a fixed-density square lattice.

Run from the repository root with:

    python examples/parallel_tempering_square_lattice_demo.py

The demo builds a Euclidean square-lattice grasshopper system, constructs its
neighborhood with jump_distance=0.60, runs replica-exchange Monte Carlo, and
writes Plotly HTML snapshots of the initial, cold-final, and best states.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
cache_dir = Path(tempfile.gettempdir()) / "grasshopper-demo-cache"
matplotlib_cache = cache_dir / "matplotlib"
xdg_cache = cache_dir / "xdg"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

import grasshopper as gh


DEFAULT_JUMP_DISTANCE = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel tempering demo on a square lattice."
    )
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--jump-distance", type=float, default=DEFAULT_JUMP_DISTANCE)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--sweep-steps", type=int, default=1_000)
    parser.add_argument("--cold-temp", type=float, default=0.05)
    parser.add_argument("--hot-temp", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/output/parallel_tempering_square_lattice"),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-iteration parallel tempering diagnostics.",
    )
    return parser.parse_args()


def write_state_plot(
    system: gh.System,
    path: Path,
    *,
    title: str,
    plot_current: bool,
) -> None:
    fig = system.plot(
        plot_current=plot_current,
        show_edges=True,
        plot_figure=False,
    )
    fig.update_layout(title=title)
    fig.write_html(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    system = gh.System.square_lattice(
        args.width,
        args.height,
        density=args.density,
        center=True,
    )
    system.compute_neighborhood(
        jump_distance=args.jump_distance,
        delta="two",
        distance_metric="euclidean",
        silent=True,
    )
    system.random_init(seed=args.seed)

    initial_energy = system.current_energy
    write_state_plot(
        system,
        args.output_dir / "initial_state.html",
        title="Initial square-lattice state",
        plot_current=True,
    )

    result = system.parallel_temper(
        num_replicas=args.replicas,
        iterations=args.iterations,
        sweep_steps=args.sweep_steps,
        exchange_interval=1,
        cold_temp=args.cold_temp,
        hot_temp=args.hot_temp,
        n_workers=1,
        reinit=True,
        seed=args.seed + 1,
        verbose=args.verbose,
    )

    write_state_plot(
        system,
        args.output_dir / "cold_final_state.html",
        title="Cold replica after parallel tempering",
        plot_current=True,
    )
    write_state_plot(
        system,
        args.output_dir / "best_state.html",
        title="Best state found by parallel tempering",
        plot_current=False,
    )

    mean_acceptance = float(np.mean(result.sweep_acceptance))
    print("Parallel tempering square-lattice demo")
    print(f"  lattice: {args.width} x {args.height}")
    print(f"  density: {args.density:.3f} ({system.num_spins} occupied sites)")
    print(f"  jump_distance: {args.jump_distance:.2f}")
    print(f"  temperatures: {np.array2string(result.temperatures, precision=3)}")
    print(f"  initial energy: {initial_energy:.6g}")
    print(f"  cold final energy: {result.cold_energy:.6g}")
    print(f"  best energy: {result.best_energy:.6g}")
    print(f"  mean sweep acceptance: {mean_acceptance:.3f}")
    print(
        "  exchange rates: "
        f"{np.array2string(np.asarray(result.exchange_rates), precision=3)}"
    )
    print(f"  plots written to: {args.output_dir}")


if __name__ == "__main__":
    main()
