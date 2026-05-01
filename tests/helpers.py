from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np

import grasshopper as gh
from grasshopper.engines import get_engine


SPHERICAL_GRID = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)

EUCLIDEAN_GRID = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ],
    dtype=float,
)

TMP_ROOT = Path(__file__).resolve().parent / "_tmp"


def spherical_grid() -> np.ndarray:
    return SPHERICAL_GRID.copy()


def euclidean_grid() -> np.ndarray:
    return EUCLIDEAN_GRID.copy()


def exact_energy(system: gh.System) -> float:
    return float(get_engine(system.config).energy(system.current_system, system.neighborhood))


def assert_fixed_density_consistent(testcase, system: gh.System) -> None:
    testcase.assertIsNotNone(system.occupancy)
    testcase.assertIsNotNone(system.current_system)
    system.occupancy.validate(
        n_sites=int(system.current_system.shape[0]),
        expected_occupied=int(system.num_spins),
        spin_array=system.current_system,
    )
    testcase.assertEqual(
        int(system.occupancy.occupied.size),
        int(np.count_nonzero(system.current_system)),
    )


def make_test_dir(name: str) -> Path:
    path = TMP_ROOT / name
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_tmp_root() -> None:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT, ignore_errors=True)
