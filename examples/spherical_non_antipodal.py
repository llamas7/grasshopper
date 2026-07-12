#!/usr/bin/env python3
"""Optimize a fixed-occupancy spherical lawn without antipodal constraints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from _common import (
    add_optimization_arguments,
    fraction,
    gh,
    optimize,
    paired_fibonacci_sphere,
    positive_int,
    report,
    save_spherical_plot,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=positive_int, default=300)
    parser.add_argument("--theta-over-pi", type=fraction, default=0.25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/output/spherical_non_antipodal"),
    )
    add_optimization_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> gh.System:
    args = parse_args(argv)
    system = gh.System.non_antipodal(paired_fibonacci_sphere(args.pairs))
    system.compute_neighborhood(
        jump_distance=args.theta_over_pi * np.pi,
        kernel=gh.KernelSpec.cosine(),
        silent=True,
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
    report(system, title="Spherical non-antipodal example")
    output = save_spherical_plot(
        system,
        args.output_dir / "best_lawn.html",
        title="Best non-antipodal lawn",
        show=args.show,
    )
    print(f"  plot: {output}")
    return system


if __name__ == "__main__":
    main()
