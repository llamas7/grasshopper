#!/usr/bin/env python3
"""Compare compact-delta and Gaussian interactions on the same 2D lawn."""

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
    parser.add_argument("--eta", type=positive_float, default=0.20)
    parser.add_argument("--cutoff", type=positive_float, default=5.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/output/gaussian_kernel"),
    )
    add_optimization_arguments(parser)
    return parser.parse_args(argv)


def build_system(args: argparse.Namespace, kernel: gh.KernelSpec) -> gh.System:
    system = gh.System.square_lattice(
        args.width,
        args.height,
        density=args.density,
        center=True,
    )
    system.compute_neighborhood(
        jump_distance=args.jump_distance,
        kernel=kernel,
        distance_metric="euclidean",
        silent=True,
    )
    return system


def main(argv: Sequence[str] | None = None) -> tuple[gh.System, gh.System]:
    args = parse_args(argv)
    sigma = args.eta * args.jump_distance
    compact = build_system(args, gh.KernelSpec.cosine())
    gaussian = build_system(
        args,
        gh.KernelSpec.gaussian(sigma=sigma, cutoff=args.cutoff),
    )

    compact.random_init(seed=args.seed)
    gaussian.array_init(compact.current_spins.copy())
    optimize(
        compact,
        steps=args.steps,
        anneal_iterations=args.anneal_iterations,
        greedy_steps=args.greedy_steps,
        starting_temp=args.starting_temp,
        seed=args.seed + 1,
    )
    optimize(
        gaussian,
        steps=args.steps,
        anneal_iterations=args.anneal_iterations,
        greedy_steps=args.greedy_steps,
        starting_temp=args.starting_temp,
        seed=args.seed + 1,
    )

    print(f"Gaussian width: sigma={sigma:.6g}, sigma/d={args.eta:.6g}")
    report(compact, title="Compact cosine regularization")
    report(gaussian, title="Physical Gaussian interaction")
    compact_output = save_euclidean_plot(
        compact,
        args.output_dir / "compact_lawn.png",
        title="Compact cosine kernel",
        show=args.show,
        voronoi=True,
    )
    gaussian_output = save_euclidean_plot(
        gaussian,
        args.output_dir / "gaussian_lawn.png",
        title=rf"Gaussian kernel, $\sigma/d={args.eta:.3g}$",
        show=args.show,
        voronoi=True,
    )
    print(f"  compact plot: {compact_output}")
    print(f"  Gaussian plot: {gaussian_output}")
    return compact, gaussian


if __name__ == "__main__":
    main()
