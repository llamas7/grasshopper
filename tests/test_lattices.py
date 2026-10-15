"""Geometry and boundary checks for compact finite lattice patches."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import grasshopper as gh


class FibonacciLatticeTests(unittest.TestCase):
    def test_disk_has_exact_count_and_uniform_radial_area(self):
        for count in (1, 2, 37, 1000):
            with self.subTest(count=count):
                points = gh.fibonacci_lattice(count, radius=2.0, origin=(3.0, -4.0))
                self.assertEqual(points.shape, (count, 2))
                self.assertEqual(points.dtype, np.float64)
                self.assertTrue(points.flags.c_contiguous)
                self.assertEqual(len(np.unique(points, axis=0)), count)
                squared_radii = np.sum((points - (3.0, -4.0)) ** 2, axis=1)
                self.assertTrue(np.all(np.diff(squared_radii) > 0))
                self.assertTrue(np.all(squared_radii < 4.0))
                # Every equal-area radial bin contains its share of the sites.
                for fraction in (0.25, 0.5, 0.75):
                    self.assertLessEqual(abs(np.count_nonzero(squared_radii < 4 * fraction)
                                             - fraction * count), 1)
                np.testing.assert_array_equal(
                    points, gh.fibonacci_lattice(count, radius=2.0, origin=(3.0, -4.0))
                )
        points = gh.fibonacci_lattice(1000)
        self.assertLess(np.linalg.norm(points.mean(axis=0)), 0.002)
        radii = np.linalg.norm(points, axis=1)
        np.testing.assert_allclose(np.pi * radii[-1] ** 2, 999.5 / 1000)
        directions = points / radii[:, None]
        angular_steps = np.sum(directions[1:] * directions[:-1], axis=1)
        np.testing.assert_allclose(angular_steps, np.cos(np.pi * (3 - np.sqrt(5))), atol=1e-12)

    def test_invalid_inputs(self):
        for count in (0, -1):
            with self.assertRaises(ValueError):
                gh.fibonacci_lattice(count)
        for count in (True, 3.5):
            with self.assertRaises(TypeError):
                gh.fibonacci_lattice(count)
        for radius in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                gh.fibonacci_lattice(10, radius=radius)
        with self.assertRaises(ValueError):
            gh.fibonacci_lattice(10, origin=(np.nan, 0))
        with self.assertRaises(ValueError):
            gh.System.fibonacci_lattice(10, num_spins=4, density=0.5)

    def test_physical_scale_optimization_and_save_load(self):
        for count in (64, 128):
            system = gh.System.fibonacci_lattice(count, num_spins=32, origin=(2, -1))
            radial_area = np.pi * np.sum((system.grid - (2, -1)) ** 2, axis=1)
            np.testing.assert_allclose(np.diff(radial_area), 1 / 32)
        system = gh.System.fibonacci_lattice(64, density=0.5)
        system.compute_neighborhood(0.3, silent=True)
        system.random_init(seed=123)
        system.greedy(max_steps=10, reinit=False)
        self.assertEqual(np.count_nonzero(system.current_system), 32)
        self.assertTrue(np.isfinite(system.best_probability))
        with TemporaryDirectory() as directory:
            path = system.save(Path(directory) / "fibonacci.nc")
            restored = gh.System.fibonacci_lattice(64)
            restored.load(path, rebuild_neighborhood=True)
            np.testing.assert_array_equal(restored.grid, system.grid)
            np.testing.assert_array_equal(restored.current_system, system.current_system)
            self.assertAlmostEqual(restored.best_probability, system.best_probability)


class CompactLatticeTests(unittest.TestCase):
    def test_exact_site_budget_spacing_and_connectivity(self):
        for builder, sites_per_cell, degree in (
            (gh.triangular_lattice, 1, 6),
            (gh.hexagonal_lattice, 2, 3),
        ):
            for count in (1, 2, 3, 4, 7, 16, 63, 256):
                with self.subTest(lattice=builder.__name__, count=count):
                    points = builder(count, 1, spacing=1.7)
                    self.assertEqual(points.shape, (count * sites_per_cell, 2))
                    self.assertEqual(len(np.unique(points, axis=0)), len(points))
                    if len(points) == 1:
                        continue
                    tree = cKDTree(points)
                    np.testing.assert_allclose(tree.query(points, k=2)[0][:, 1], 1.7)
                    pairs = tree.query_pairs(1.7 + 1e-8, output_type="ndarray")
                    adjacency = coo_matrix(
                        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                        shape=(len(points), len(points)),
                    )
                    self.assertEqual(connected_components(adjacency, directed=False)[0], 1)
                    neighbors = tree.query_ball_point(points, 1.7 + 1e-8, return_length=True) - 1
                    self.assertLessEqual(neighbors.max(), degree)

    def test_compact_patches_reduce_boundary_and_missing_links(self):
        for builder, degree in ((gh.triangular_lattice, 6), (gh.hexagonal_lattice, 3)):
            for size in (16, 32, 64):
                with self.subTest(lattice=builder.__name__, size=size):
                    compact = builder(size, size, spacing=1.0)
                    legacy = builder(size, size, spacing=1.0, compact=False)
                    counts = [
                        cKDTree(p).query_ball_point(p, 1.0 + 1e-8, return_length=True) - 1
                        for p in (compact, legacy)
                    ]
                    self.assertEqual(len(compact), len(legacy))
                    self.assertLess(np.count_nonzero(counts[0] < degree),
                                    np.count_nonzero(counts[1] < degree))
                    self.assertLess(np.sum(degree - counts[0]), np.sum(degree - counts[1]))
                    extents = np.ptp(compact, axis=0)
                    self.assertLess(extents.max() / extents.min(), 1.1)

    def test_center_origin_aliases_and_determinism(self):
        for builder, name in ((gh.triangular_lattice, "tri"), (gh.hexagonal_lattice, "hex")):
            points = builder(12, 8, spacing=2, center=True, origin=(3, -4))
            np.testing.assert_allclose((points.min(axis=0) + points.max(axis=0)) / 2, (3, -4))
            np.testing.assert_array_equal(
                points, gh.regular_lattice(name, 12, 8, spacing=2, center=True, origin=(3, -4))
            )
            np.testing.assert_array_equal(points, builder(8, 12, spacing=2, center=True, origin=(3, -4)))
        np.testing.assert_array_equal(gh.honeycomb_lattice(12, 8), gh.hexagonal_lattice(12, 8))
        honeycomb = gh.hexagonal_lattice(12, 8, spacing=1, center=True)
        np.testing.assert_allclose(cKDTree(honeycomb).query(-honeycomb)[0], 0, atol=1e-12)

    def test_system_uses_lattice_cell_area_for_physical_spacing(self):
        # Area per site is sqrt(3)/2 * a^2 (triangular), 3sqrt(3)/4 * a^2 (honeycomb).
        for constructor, area_factor in (
            (gh.System.triangular_lattice, np.sqrt(3) / 2),
            (gh.System.hexagonal_lattice, 3 * np.sqrt(3) / 4),
        ):
            for size in (12, 24):
                system = constructor(size, size, num_spins=50)
                distances = cKDTree(system.grid).query(system.grid, k=2)[0][:, 1]
                np.testing.assert_allclose(distances, np.sqrt(1 / (50 * area_factor)))
                self.assertEqual(system.num_spins, 50)
                system.compute_neighborhood(jump_distance=0.2, silent=True)
                system.random_init(seed=123)
                self.assertEqual(np.count_nonzero(system.current_system), 50)
                self.assertTrue(np.isfinite(system.current_probability))
