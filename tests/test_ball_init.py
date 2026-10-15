import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import grasshopper as gh
from tests.helpers import assert_fixed_density_consistent, exact_energy, spherical_grid


class BallInitializationTests(unittest.TestCase):
    def test_square_ball_includes_boundary_and_rescales_density(self):
        system = gh.System.square_lattice(9, 9, num_spins=20)
        system.compute_neighborhood(0.3, silent=True)
        system.random_init(seed=1)
        system.greedy(max_steps=1)
        grid = system.grid.copy()
        neighborhood = system.neighborhood
        result = system.ball_init(2)
        rows, cols = np.indices((9, 9))
        expected = ((rows - 4) ** 2 + (cols - 4) ** 2 <= 4).ravel()
        self.assertIs(result, system)
        self.assertEqual(system.num_spins, 13)
        np.testing.assert_array_equal(system.current_spins, expected)
        np.testing.assert_array_equal(system.best_spins, expected)
        np.testing.assert_allclose(system.grid, grid * np.sqrt(20 / 13))
        self.assertIsNot(system.neighborhood, neighborhood)
        self.assertEqual(system.jump_distance, 0.3)
        self.assertEqual(system.greedy_history.runs, 0)
        self.assertTrue(system.initialized)
        self.assertAlmostEqual(system.current_energy, exact_energy(system))
        self.assertAlmostEqual(system.best_energy, system.current_energy)
        self.assertTrue(np.isfinite(system.current_probability))
        assert_fixed_density_consistent(self, system)
        with TemporaryDirectory() as directory:
            path = system.save(Path(directory) / "ball.nc")
            restored = gh.System.square_lattice(9, 9, num_spins=13)
            restored.load(path, rebuild_neighborhood=True)
            np.testing.assert_allclose(restored.grid, system.grid)
            np.testing.assert_array_equal(restored.current_spins, expected)
            self.assertAlmostEqual(restored.current_probability, system.current_probability)
        system.greedy(max_steps=5)
        assert_fixed_density_consistent(self, system)

    def test_same_count_reuses_graph_and_ties_choose_first_site(self):
        system = gh.System.square_lattice(4, 4, num_spins=5)
        system.compute_neighborhood(0.3, silent=True)
        grid, graph = system.grid, system.neighborhood
        system.ball_init(1)
        # Four center candidates tie; row-major site 5 is selected.
        expected = np.zeros(16, dtype=bool)
        expected[[1, 4, 5, 6, 9]] = True
        np.testing.assert_array_equal(system.current_spins, expected)
        self.assertIs(system.grid, grid)
        self.assertIs(system.neighborhood, graph)

    def test_one_and_three_dimensions(self):
        for dimension, expected_count in ((1, 3), (3, 7)):
            with self.subTest(dimension=dimension):
                axes = np.meshgrid(*([np.arange(5)] * dimension), indexing="ij")
                grid = np.column_stack([axis.ravel() for axis in axes]).astype(float)
                system = gh.System.euclidean(grid, num_spins=2)
                system.compute_neighborhood(0.4, silent=True)
                system.ball_init(1)
                self.assertEqual(system.num_spins, expected_count)
                assert_fixed_density_consistent(self, system)

    def test_irregular_disk_uses_a_site_as_center_and_accepts_fractional_radius(self):
        system = gh.System.fibonacci_lattice(100, num_spins=40, origin=(3, -2))
        system.compute_neighborhood(0.3, silent=True)
        grid = system.grid.copy()
        center = (grid.min(axis=0) + grid.max(axis=0)) / 2
        center_index = np.argmin(np.linalg.norm(grid - center, axis=1))
        expected = np.linalg.norm(grid - grid[center_index], axis=1) <= 1.7 / np.sqrt(40)
        system.ball_init(1.7)
        np.testing.assert_array_equal(system.current_spins, expected)
        self.assertTrue(system.current_spins[center_index])
        assert_fixed_density_consistent(self, system)

    def test_zero_radius_and_invalid_inputs(self):
        system = gh.System.square_lattice(5, 5, num_spins=10)
        with self.assertRaisesRegex(RuntimeError, "Neighborhood"):
            system.ball_init(1)
        system.compute_neighborhood(0.3, silent=True)
        grid, graph = system.grid, system.neighborhood
        for radius in (-1, np.nan, np.inf, 100):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                system.ball_init(radius)
            self.assertIs(system.grid, grid)
            self.assertIs(system.neighborhood, graph)
            self.assertEqual(system.num_spins, 10)
        system.ball_init(0)
        self.assertEqual(system.num_spins, 1)
        self.assertTrue(system.current_spins[12])
        assert_fixed_density_consistent(self, system)
        sphere = gh.System.fixed_density(spherical_grid())
        with self.assertRaisesRegex(ValueError, "Euclidean"):
            sphere.ball_init(1)

    def test_failed_rebuild_preserves_previous_state(self):
        system = gh.System.square_lattice(7, 7, num_spins=20)
        system.compute_neighborhood(0.3, silent=True)
        system.random_init(seed=1)
        state = system.snapshot_state()
        grid, graph = system.grid, system.neighborhood
        with mock.patch("grasshopper.system.build_euclidean_neighborhood", side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                system.ball_init(1)
        self.assertIs(system.grid, grid)
        self.assertIs(system.neighborhood, graph)
        self.assertEqual(system.num_spins, 20)
        np.testing.assert_array_equal(system.current_spins, state.current_spins)
        self.assertEqual(system.current_energy, state.current_energy)

    def test_custom_kernel_is_retained_during_density_rebuild(self):
        from grasshopper.kernels import cosine_kernel

        system = gh.System.square_lattice(7, 7, num_spins=20)
        system.compute_neighborhood(0.3, kernel=cosine_kernel, distance_metric="manhattan")
        system.ball_init(1)
        self.assertIs(system.kernel, cosine_kernel)
        self.assertEqual(system.distance_metric, "manhattan")
        self.assertEqual(system.num_spins, 5)
        self.assertAlmostEqual(system.current_energy, exact_energy(system))
