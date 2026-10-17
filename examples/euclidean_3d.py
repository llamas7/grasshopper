#!/usr/bin/env python3
"""Optimize a fixed-occupancy Grasshopper lawn on a cubic candidate grid."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from _common import (
    add_optimization_arguments,
    cubic_grid,
    fraction,
    gh,
    optimize,
    positive_float,
    positive_int,
    report,
    save_euclidean_plot,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=positive_int, default=10)
    parser.add_argument("--density", type=fraction, default=0.4)
    parser.add_argument("--jump-distance", type=positive_float, default=0.35)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/output/euclidean_3d"),
    )
    add_optimization_arguments(
        parser,
        default_steps=1_500,
        default_iterations=5,
        default_greedy_steps=300,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> gh.System:
    args = parse_args(argv)
    grid = cubic_grid(args.side)
    occupied_sites = max(1, int(round(args.density * len(grid))))
    system = gh.System.euclidean(grid, num_spins=occupied_sites)
    system.compute_neighborhood(
        jump_distance=args.jump_distance,
        kernel=gh.KernelSpec.cosine(),
        distance_metric="euclidean",
    )
    system.random_init(seed=args.seed)
    optimize(
        system,
        steps=args.steps,
        anneal_iterations=args.anneal_iterations,
        greedy_steps=args.greedy_steps,
        starting_temp=args.starting_temp,
        seed=args.seed + 1,
    )
    report(system, title="Three-dimensional Euclidean example")
    output = save_euclidean_plot(
        system,
        args.output_dir / "best_lawn.png",
        title="Best three-dimensional lawn",
        show=args.show,
    )
    print(f"  plot: {output}")
    return system


if __name__ == "__main__":
    main()

