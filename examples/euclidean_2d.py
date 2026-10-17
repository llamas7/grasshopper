#!/usr/bin/env python3
"""Optimize a fixed-occupancy Grasshopper lawn on a square lattice."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from _common import (
    add_optimization_arguments,
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
    parser.add_argument("--width", type=positive_int, default=32)
    parser.add_argument("--height", type=positive_int, default=32)
    parser.add_argument("--density", type=fraction, default=0.5)
    parser.add_argument("--jump-distance", type=positive_float, default=0.30)
    parser.add_argument("--voronoi", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/output/euclidean_2d"),
    )
    add_optimization_arguments(parser)
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
    report(system, title="Two-dimensional Euclidean example")
    output = save_euclidean_plot(
        system,
        args.output_dir / "best_lawn.png",
        title="Best two-dimensional lawn",
        show=args.show,
        voronoi=args.voronoi,
        show_edges=True,
    )
    print(f"  plot: {output}")
    return system


if __name__ == "__main__":
    main()

