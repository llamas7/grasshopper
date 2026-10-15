"""Normalized mixtures of independent-lawn anti-correlation experiments."""
from __future__ import annotations

import numpy as np

from .core import ConstraintModel, KernelSpec, Neighborhood
from .probability import _validate_spins, continuum_probability_prefactor


def validate_objective(system, angles, probabilities, kernel):
    if system.constraint != ConstraintModel.INDEPENDENT:
        raise ValueError("Anti-correlation objectives require System.independent.")
    angles = np.asarray(angles, dtype=float)
    if (angles.ndim != 1 or not angles.size or not np.all(np.isfinite(angles))
            or np.any(angles < 0) or np.any(angles >= np.pi)):
        raise ValueError("angles must be a nonempty finite sequence in [0, pi).")
    q = np.full(angles.size, 1 / angles.size) if probabilities is None else np.asarray(probabilities, dtype=float)
    if (q.shape != angles.shape or not np.all(np.isfinite(q)) or np.any(q < 0)
            or not np.isclose(q.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError("probabilities must match angles, be nonnegative and finite, and sum to one.")
    if callable(kernel):
        raise ValueError("Use a built-in compact or Gaussian kernel.")
    spec = KernelSpec.cosine() if kernel is None else KernelSpec.from_value(kernel)
    if spec.kind == "exact":
        raise ValueError("The exact diagnostic kernel is not supported; theta=0 is already exact.")
    grid = system.grid
    if (not np.all(np.isfinite(grid)) or
            not np.allclose(np.linalg.norm(grid, axis=1), 1, rtol=0, atol=1e-8) or
            not np.allclose(grid[:system.num_spins], -grid[system.num_spins:], rtol=0, atol=1e-8)):
        raise ValueError("Grid must contain unit vectors with antipodes in matching second-half order.")
    return tuple(angles), tuple(q), spec


def build_objective(system, angles, probabilities, kernel):
    # Merge one component at a time: do not retain an array of full graphs.
    n = system.num_spins
    positive = [dict() for _ in range(n)]
    negative = [dict() for _ in range(n)]
    weights = []
    for angle, q in zip(angles, probabilities):
        if angle == 0:
            weights.append(float(q))
            for i in range(n):
                positive[i][i] = positive[i].get(i, 0.0) + q
            continue
        component = type(system).independent(system.grid)
        component.compute_neighborhood(jump_distance=angle, kernel=kernel)
        weight = q * n / 2 * continuum_probability_prefactor(component)
        weights.append(float(weight))
        for i, row in enumerate(component.neighborhood):
            for target, indices, values in (
                (positive[i], row.positive_spins, row.positive_contributions),
                (negative[i], row.negative_spins, row.negative_contributions),
            ):
                for index, value in zip(indices, values):
                    target[int(index)] = target.get(int(index), 0.0) + weight * value
    rows = []
    for pos, neg in zip(positive, negative):
        pos = {i: v for i, v in pos.items() if v != 0}
        neg = {i: v for i, v in neg.items() if v != 0}
        rows.append(Neighborhood(list(pos), list(pos.values()), list(neg), list(neg.values()),
                                 float(sum(pos.values()) + sum(neg.values()))))
    return rows, tuple(weights)


def component_probabilities(system, spins):
    if spins is None:
        raise RuntimeError("Initialize a spin state first.")
    spins = _validate_spins(system, spins)
    result = []
    for angle in system.jump_distance:
        if angle == 0:
            result.append(float(np.mean(spins[:, 0] != spins[:, 1])))
        else:
            component = type(system).independent(system.grid)
            component.compute_neighborhood(jump_distance=angle, kernel=system.kernel)
            result.append(component.probability(spins))
    return np.asarray(result)
