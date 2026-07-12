#!/usr/bin/env python3
"""Optimize an antipodal complementary lawn on the sphere."""

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
        default=Path("examples/output/spherical_complementary"),
    )
    add_optimization_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> gh.System:
    args = parse_args(argv)
    system = gh.System.complementary(paired_fibonacci_sphere(args.pairs))
    system.compute_neighborhood(
        jump_distance=args.theta_over_pi * np.pi,
        kernel=gh.KernelSpec.cosine(),
        silent=True,
    )
    system.hemisphere_init(seed=args.seed)
    optimize(
        system,
        steps=args.steps,
        anneal_iterations=args.anneal_iterations,
        greedy_steps=args.greedy_steps,
        starting_temp=args.starting_temp,
        seed=args.seed + 1,
    )
    report(system, title="Spherical antipodal complementary example")
    output = save_spherical_plot(
        system,
        args.output_dir / "best_lawn.html",
        title="Best complementary lawn",
        show=args.show,
    )
    print(f"  plot: {output}")
    return system


if __name__ == "__main__":
    main()
