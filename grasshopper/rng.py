from __future__ import annotations

import numpy as np


Seed = int | np.integer | None


def normalize_seed(seed: Seed) -> int | None:
    """Validate and normalize a NumPy RandomState-compatible seed."""
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer or None, got {type(seed)!r}.")
    normalized = int(seed)
    if normalized < 0 or normalized > np.iinfo(np.uint32).max:
        raise ValueError(
            "seed must be between 0 and "
            f"{np.iinfo(np.uint32).max}, got {seed!r}."
        )
    return normalized


def seed_numpy(seed: Seed):
    """Seed NumPy's legacy global RNG, returning state for later restoration."""
    normalized = normalize_seed(seed)
    if normalized is None:
        return None
    state = np.random.get_state()
    np.random.seed(normalized)
    return state


def restore_numpy_state(state) -> None:
    """Restore a state returned by seed_numpy()."""
    if state is not None:
        np.random.set_state(state)
