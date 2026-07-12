#!/usr/bin/env python3
"""Demonstrate replica-exchange Monte Carlo on a square Grasshopper lattice."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from _common import (
    fraction,
    gh,
    positive_float,
    positive_int,
    report,
    save_euclidean_plot,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=positive_int, default=24)
    parser.add_argument("--height", type=positive_int, default=24)
    parser.add_argument("--density", type=fraction, default=0.5)
    parser.add_argument("--jump-distance", type=positive_float, default=0.35)
    parser.add_argument("--replicas", type=positive_int, default=6)
    parser.add_argument("--iterations", type=positive_int, default=12)
    parser.add_argument("--sweep-steps", type=positive_int, default=1_000)
    parser.add_argument("--cold-temp", type=positive_float, default=0.02)
    parser.add_argument("--hot-temp", type=positive_float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/output/parallel_tempering"),
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> gh.System:
    args = parse_args(argv)
    system = gh.System.square_lattice(
        args.width,
        args.height,
        density=args.density,
        center=True,
    )
    system.compute_neighborhood(
        jump_distance=args.jump_distance,
        kernel=gh.KernelSpec.cosine(),
        distance_metric="euclidean",
        silent=True,
    )
    system.random_init(seed=args.seed)
    initial_energy = float(system.current_energy)
    initial_output = save_euclidean_plot(
        system,
        args.output_dir / "initial_lawn.png",
        title="Initial lawn",
        show=False,
        plot_current=True,
        show_edges=True,
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
    final_output = save_euclidean_plot(
        system,
        args.output_dir / "cold_final_lawn.png",
        title="Cold replica after parallel tempering",
        show=False,
        plot_current=True,
        show_edges=True,
    )
    best_output = save_euclidean_plot(
        system,
        args.output_dir / "best_lawn.png",
        title="Best lawn found by parallel tempering",
        show=args.show,
        show_edges=True,
    )

    report(system, title="Parallel-tempering square-lattice example")
    print(f"  initial energy: {initial_energy:.6g}")
    print(
        "  temperatures: "
        f"{np.array2string(result.temperatures, precision=4)}"
    )
    print(
        "  exchange rates: "
        f"{np.array2string(np.asarray(result.exchange_rates), precision=3)}"
    )
    print(f"  initial plot: {initial_output}")
    print(f"  cold-final plot: {final_output}")
    print(f"  best plot: {best_output}")
    return system


if __name__ == "__main__":
    main()
