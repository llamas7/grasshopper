#!/usr/bin/env python3
"""Search for achievable multi-angle anti-correlation scores (not certified optima)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from _common import add_optimization_arguments, gh, paired_fibonacci_sphere, positive_int


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angles-over-pi", type=float, nargs="+", required=True)
    parser.add_argument("--probabilities", type=float, nargs="+")
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument("--grid", type=Path, help="Antipodally ordered unit vectors, .npy or text")
    grid.add_argument("--pairs", type=positive_int, default=300, help="Demo resolution, not a convergence claim")
    parser.add_argument("--restarts", type=positive_int, default=4)
    parser.add_argument("--kernel", choices=["cosine", "cell_averaged_hat", "cell_averaged_cosine", "gaussian"], default="cosine")
    parser.add_argument("--sigma", type=float, help="Physical Gaussian angular width in radians")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output/spherical_multi_angle"))
    add_optimization_arguments(parser)
    args = parser.parse_args(argv)
    if args.kernel == "gaussian" and args.sigma is None:
        parser.error("--sigma is required for a Gaussian kernel")
    if args.kernel != "gaussian" and args.sigma is not None:
        parser.error("--sigma requires --kernel gaussian")
    return args


def main(argv=None):
    args = parse_args(argv)
    grid = (np.load(args.grid, allow_pickle=False) if args.grid.suffix == ".npy" else np.loadtxt(args.grid)) if args.grid else paired_fibonacci_sphere(args.pairs)
    kernel = (gh.KernelSpec.gaussian(args.sigma, cutoff=args.cutoff)
              if args.kernel == "gaussian" else gh.KernelSpec.from_value(args.kernel))
    base = gh.System.independent(grid).compute_anticorrelation_neighborhood(
        np.pi * np.asarray(args.angles_over_pi), probabilities=args.probabilities, kernel=kernel,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Refuse silent replacement of a previous experiment.
    if (args.output_dir / "summary.json").exists() or (args.output_dir / "grid.npy").exists():
        raise FileExistsError("Output directory already contains an experiment; choose a new --output-dir.")
    np.save(args.output_dir / "grid.npy", grid)
    seeds = np.random.SeedSequence(args.seed).spawn(args.restarts)
    runs = []
    winner = None
    for i, seed_sequence in enumerate(seeds):
        init_seed, anneal_seed = (int(x) for x in seed_sequence.generate_state(2))
        system = base.clone()
        if i == 0:
            system.hemisphere_init(seed=init_seed)
        else:
            system.random_init(seed=init_seed)
        if args.anneal_iterations:
            system.anneal(starting_temp=args.starting_temp,
                          final_temp=max(args.starting_temp * 0.05, 1e-8),
                          cooling_rate=0.65, min_acceptance=0.0,
                          max_iterations=args.anneal_iterations, steps=args.steps,
                          seed=anneal_seed, reinit=True)
        if args.greedy_steps:
            # Polish the best state visited, rather than a warmer terminal state.
            system.array_init(system.best_spins.copy())
            system.greedy(max_steps=args.greedy_steps, reinit=True)
        checkpoint = f"run_{i:03d}.nc"
        system.save(args.output_dir / checkpoint)
        runs.append(dict(index=i, initialization="hemisphere" if i == 0 else "random",
                         init_seed=init_seed, anneal_seed=anneal_seed, checkpoint=checkpoint,
                         current_probability=system.current_probability,
                         best_probability=system.best_probability,
                         current_angle_probabilities=system.angle_probabilities().tolist(),
                         best_angle_probabilities=system.angle_probabilities(system.best_spins).tolist()))
        if winner is None or system.best_probability > winner.best_probability:
            winner = system
        print(f"Run {i}: achievable score {system.best_probability:.10g}")
    best = max(runs, key=lambda r: r["best_probability"])
    summary = dict(objective="weighted_anticorrelation", status="achievable candidates, not certified optima",
                   angles=list(base.jump_distance), probabilities=list(base.sampling_probabilities),
                   kernel=asdict(kernel), seed=args.seed, grid_file="grid.npy",
                   grid_source=str(args.grid.resolve()) if args.grid else "paired_fibonacci_demo",
                   sites=len(grid), optimization=dict(steps=args.steps, anneal_iterations=args.anneal_iterations,
                   greedy_steps=args.greedy_steps, starting_temp=args.starting_temp),
                   runs=runs, winning_checkpoint=best["checkpoint"], best_probability=best["best_probability"])
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    if args.show:
        winner.plot(plot_figure=True)
    return summary


if __name__ == "__main__":
    main()
