from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from unittest import mock
import unittest

import numpy as np

import grasshopper as gh
import grasshopper.euclidean.build_neighbors as euclidean_neighbors
from grasshopper.kernels import (
    calculate_contributions,
    cosine_kernel,
    resolve_delta_kernel,
)

from tests.helpers import cleanup_tmp_root, euclidean_grid, make_test_dir, spherical_grid


class SystemApiTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        cleanup_tmp_root()

    def setUp(self) -> None:
        np.random.seed(12345)

    def test_named_constructors_set_expected_configs(self) -> None:
        system = gh.System.non_antipodal(spherical_grid())

        self.assertEqual(system.geometry, gh.Geometry.SPHERICAL)
        self.assertEqual(system.constraint, gh.ConstraintModel.FIXED_DENSITY)
        self.assertEqual(system.interaction_model, gh.InteractionModel.OCCUPANCY)
        self.assertEqual(system.update_rule, gh.UpdateRule.SWAP)
        self.assertEqual(system.config, gh.SystemConfig.non_antipodal())

        inverse = gh.System.inverse(spherical_grid())
        self.assertEqual(inverse.config, gh.SystemConfig.inverse())

    def test_regular_lattice_grid_builders(self) -> None:
        square = gh.square_lattice(3, 2, spacing=2.0)
        np.testing.assert_allclose(
            square,
            np.array(
                [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [4.0, 0.0],
                    [0.0, 2.0],
                    [2.0, 2.0],
                    [4.0, 2.0],
                ]
            ),
        )

        rectangular = gh.rectangular_lattice(2, 2, spacing=(2.0, 3.0))
        np.testing.assert_allclose(
            rectangular,
            np.array(
                [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 3.0],
                    [2.0, 3.0],
                ]
            ),
        )

        triangular = gh.triangular_lattice(2, 2, spacing=1.0)
        np.testing.assert_allclose(
            triangular,
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.5, np.sqrt(3.0) / 2.0],
                    [1.5, np.sqrt(3.0) / 2.0],
                ]
            ),
        )

        hexagonal = gh.hexagonal_lattice(1, 1, spacing=2.0)
        np.testing.assert_allclose(hexagonal, np.array([[0.0, 0.0], [0.0, 2.0]]))

    def test_regular_lattice_system_constructors(self) -> None:
        system = gh.System.square_lattice(4, 3, density=0.25)

        self.assertEqual(system.config, gh.SystemConfig.euclidean())
        self.assertEqual(system.grid.shape, (12, 2))
        self.assertEqual(system.num_spins, 3)
        self.assertEqual(system.system_shape, 12)

        honeycomb = gh.System.honeycomb_lattice(2, 2)
        self.assertEqual(honeycomb.grid.shape, (8, 2))
        self.assertEqual(honeycomb.num_spins, 4)

        triangular = gh.System.regular_lattice(
            "tri",
            3,
            2,
            num_spins=2,
            center=True,
        )
        self.assertEqual(triangular.config, gh.SystemConfig.euclidean())
        self.assertEqual(triangular.num_spins, 2)
        np.testing.assert_allclose(
            (np.min(triangular.grid, axis=0) + np.max(triangular.grid, axis=0)) / 2,
            np.zeros(2),
        )

    def test_regular_lattice_constructor_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown lattice"):
            gh.regular_lattice("kite", 2, 2)
        with self.assertRaisesRegex(ValueError, "Pass either num_spins or density"):
            gh.System.square_lattice(2, 2, num_spins=1, density=0.5)
        with self.assertRaisesRegex(ValueError, "density must be between 0 and 1"):
            gh.System.square_lattice(2, 2, density=1.0)
        with self.assertRaises(TypeError):
            gh.System.square_lattice(2, 2, spacing=1.0)
        with self.assertRaises(TypeError):
            gh.System.euclidean(euclidean_grid(), num_spins=2, normalize=False)
        with self.assertRaises(TypeError):
            gh.System(
                euclidean_grid(),
                config=gh.SystemConfig.euclidean(),
                num_spins=2,
                normalize_euclidean=False,
            )

    def test_kernel_drops_zero_weight_cutoff_neighbors(self) -> None:
        contributions, mask, spin_potential = calculate_contributions(
            jump_distance=0.30,
            cell_length=0.01,
            distances=np.array([0.31999999999999995, 0.32000000000000006]),
            delta_func=cosine_kernel,
        )

        self.assertEqual(contributions.size, 0)
        self.assertFalse(np.any(mask))
        self.assertEqual(spin_potential, 0.0)

    def test_manual_square_grid_matches_builtin_square_lattice_coordinates(self) -> None:
        grid_length = 100
        num_spins = 1000
        manual_grid = np.array(
            [
                [i / grid_length, j / grid_length]
                for i in range(grid_length)
                for j in range(grid_length)
            ],
            dtype=float,
        )
        builtin_system = gh.System.square_lattice(
            grid_length,
            grid_length,
            num_spins=num_spins,
        )
        manual_system = gh.System.euclidean(manual_grid, num_spins=num_spins)

        manual_order = np.lexsort((manual_system.grid[:, 1], manual_system.grid[:, 0]))
        builtin_order = np.lexsort((builtin_system.grid[:, 1], builtin_system.grid[:, 0]))
        np.testing.assert_allclose(
            manual_system.grid[manual_order],
            builtin_system.grid[builtin_order],
        )
        self.assertEqual(manual_system.num_spins, builtin_system.num_spins)

        with self.assertRaisesRegex(ValueError, "between 1 and N-1"):
            gh.System.euclidean(manual_grid, num_spins=grid_length * grid_length)

    def test_euclidean_constructor_normalizes_raw_square_grid_area(self) -> None:
        grid_length = 4
        raw_grid = np.array(
            [[i, j] for i in range(grid_length) for j in range(grid_length)],
            dtype=float,
        )

        raw_system = gh.System.euclidean(raw_grid, num_spins=4)
        builtin_system = gh.System.square_lattice(grid_length, grid_length, num_spins=4)
        raw_order = np.lexsort((raw_system.grid[:, 1], raw_system.grid[:, 0]))
        builtin_order = np.lexsort((builtin_system.grid[:, 1], builtin_system.grid[:, 0]))
        np.testing.assert_allclose(
            raw_system.grid[raw_order],
            builtin_system.grid[builtin_order],
        )
        self.assertAlmostEqual(
            euclidean_neighbors.estimate_bounding_box_volume(raw_system.grid),
            float(grid_length * grid_length) / raw_system.num_spins,
        )
        self.assertAlmostEqual(
            euclidean_neighbors.estimate_bounding_box_volume(raw_grid),
            float(grid_length * grid_length),
        )

    def test_euclidean_cell_length_uses_occupied_site_area(self) -> None:
        seen_cell_lengths = []

        def record_cell_length(distances, cell_length):
            seen_cell_lengths.append(float(cell_length))
            mask = np.zeros(distances.shape, dtype=bool)
            return np.ones(0, dtype=float), mask

        system = gh.System.square_lattice(4, 4, num_spins=4)
        system.compute_neighborhood(
            jump_distance=0.25,
            kernel=record_cell_length,
            silent=True,
        )

        self.assertTrue(seen_cell_lengths)
        np.testing.assert_allclose(
            seen_cell_lengths,
            np.full(len(seen_cell_lengths), 0.5),
        )

    def test_euclidean_cell_length_does_not_shrink_with_candidate_sites(self) -> None:
        def collect_cell_lengths(grid_length):
            seen_cell_lengths = []

            def record_cell_length(distances, cell_length):
                seen_cell_lengths.append(float(cell_length))
                return np.ones(0, dtype=float), np.zeros(distances.shape, dtype=bool)

            system = gh.System.square_lattice(grid_length, grid_length, num_spins=4)
            system.compute_neighborhood(
                jump_distance=0.25,
                kernel=record_cell_length,
                silent=True,
            )
            return seen_cell_lengths

        coarse_cell_lengths = collect_cell_lengths(4)
        fine_cell_lengths = collect_cell_lengths(8)

        self.assertTrue(coarse_cell_lengths)
        self.assertTrue(fine_cell_lengths)
        np.testing.assert_allclose(
            coarse_cell_lengths,
            np.full(len(coarse_cell_lengths), 0.5),
        )
        np.testing.assert_allclose(
            fine_cell_lengths,
            np.full(len(fine_cell_lengths), 0.5),
        )

    def test_manual_square_grid_matches_builtin_square_lattice_neighborhoods(self) -> None:
        grid_length = 8
        num_spins = 10
        jump_distance = 0.30
        manual_grid = np.array(
            [
                [i / grid_length, j / grid_length]
                for i in range(grid_length)
                for j in range(grid_length)
            ],
            dtype=float,
        )
        manual_system = gh.System.euclidean(manual_grid, num_spins=num_spins)
        builtin_system = gh.System.square_lattice(
            grid_length,
            grid_length,
            num_spins=num_spins,
        )

        manual_system.compute_neighborhood(jump_distance=jump_distance, silent=True)
        builtin_system.compute_neighborhood(jump_distance=jump_distance, silent=True)
        self.assertEqual(manual_system.kernel, builtin_system.kernel)

        manual_to_builtin = np.empty(len(manual_grid), dtype=np.int64)
        for manual_index, (x_coord, y_coord) in enumerate(manual_grid):
            x_index = int(round(x_coord * grid_length))
            y_index = int(round(y_coord * grid_length))
            manual_to_builtin[manual_index] = y_index * grid_length + x_index

        for manual_index in range(len(manual_grid)):
            manual_row = manual_system.neighborhood[manual_index]
            builtin_row = builtin_system.neighborhood[manual_to_builtin[manual_index]]
            mapped_neighbors = manual_to_builtin[manual_row.neighbor_spins]
            order = np.argsort(mapped_neighbors, kind="stable")

            np.testing.assert_array_equal(
                mapped_neighbors[order],
                builtin_row.neighbor_spins,
            )
            np.testing.assert_allclose(
                manual_row.contributions[order],
                builtin_row.contributions,
            )
            self.assertAlmostEqual(
                manual_row.spin_potential,
                builtin_row.spin_potential,
            )

    def test_regular_lattice_plotting(self) -> None:
        system = gh.System.square_lattice(2, 2)

        self.assertFalse(hasattr(system, "plot_lattice"))

        system.compute_neighborhood(jump_distance=1.0, silent=True)
        system.random_init(seed=1)
        spin_fig = system.plot(plot_current=True, show_edges=True, plot_figure=False)
        trace_names = {trace.name for trace in spin_fig.data}
        self.assertIn("A", trace_names)
        self.assertIn("B", trace_names)
        self.assertNotIn("voronoi A", trace_names)
        self.assertNotIn("voronoi B", trace_names)

    def test_euclidean_plot_voronoi_colors_cells_by_spin_state(self) -> None:
        system = gh.System.square_lattice(3, 3, num_spins=4)
        system.compute_neighborhood(jump_distance=1.0, silent=True)
        system.array_init(
            np.array(
                [
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    False,
                ],
                dtype=bool,
            )
        )

        fig = system.plot(plot_current=True, voronoi=True, plot_figure=False)
        trace_names = {trace.name for trace in fig.data}

        self.assertIn("A", trace_names)
        self.assertIn("B", trace_names)
        voronoi_a = next(trace for trace in fig.data if trace.name == "A")
        voronoi_b = next(trace for trace in fig.data if trace.name == "B")
        self.assertEqual(voronoi_a.fill, "toself")
        self.assertEqual(voronoi_b.fill, "toself")
        self.assertIn("state: A", voronoi_a.text[0])
        self.assertIn("state: B", voronoi_b.text[0])

    def test_voronoi_padding_uses_half_cell_length(self) -> None:
        system = gh.System.square_lattice(3, 3, num_spins=4)
        system.compute_neighborhood(jump_distance=1.5, silent=True)
        system.random_init(seed=1)
        expected_padding = ((1.0 / system.num_spins) ** (1.0 / system.dimension)) / 2.0

        with mock.patch("grasshopper.system.plot_spin_system") as plot_spin_system:
            system.plot(plot_current=True, voronoi=True, plot_figure=False)
            self.assertAlmostEqual(
                plot_spin_system.call_args.kwargs["voronoi_padding_distance"],
                expected_padding,
            )

        with mock.patch("grasshopper.system.plot_spin_system") as plot_spin_system:
            system.plot_neighbors(selected_site=4, plot_figure=False)
            self.assertAlmostEqual(
                plot_spin_system.call_args.kwargs["voronoi_padding_distance"],
                expected_padding,
            )

    def test_euclidean_plot_voronoi_requires_initialized_state(self) -> None:
        system = gh.System.square_lattice(2, 2)

        with self.assertRaisesRegex(RuntimeError, "initialized spin state"):
            system.plot(voronoi=True, plot_figure=False)

    def test_plot_voronoi_requires_2d_euclidean_system(self) -> None:
        system = gh.System.inverse(spherical_grid())
        system.compute_neighborhood(jump_distance=np.pi / 2, silent=True)
        system.hemisphere_init(seed=1)

        with self.assertRaisesRegex(ValueError, "2D Euclidean"):
            system.plot(voronoi=True, plot_figure=False)

    def test_plot_neighbors_selected_site_colors_cells(self) -> None:
        system = gh.System.square_lattice(3, 3, num_spins=4)
        system.compute_neighborhood(
            jump_distance=1.5,
            silent=True,
        )

        fig = system.plot_neighbors(
            selected_site=4,
            plot_figure=False,
        )
        trace_names = {trace.name for trace in fig.data}

        self.assertIn("voronoi cells", trace_names)
        self.assertIn("selected voronoi cell", trace_names)
        self.assertIn("neighbor contribution cells", trace_names)
        self.assertIn("selected site", trace_names)
        self.assertNotIn("lattice edges", trace_names)
        self.assertNotIn("selected-neighbor correlations", trace_names)
        selected_cell = next(
            trace for trace in fig.data if trace.name == "selected voronoi cell"
        )
        self.assertEqual(selected_cell.fill, "toself")
        neighbor_cell = next(
            trace for trace in fig.data if trace.name == "neighbor contribution cells"
        )
        self.assertIn("correlation", neighbor_cell.text[0])
        self.assertIn("rgba(", neighbor_cell.fillcolor)

    def test_plot_neighbors_skips_tiny_grids(self) -> None:
        grid = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
        system = gh.System.euclidean(grid, num_spins=1)
        system.compute_neighborhood(jump_distance=1.0, silent=True)

        fig = system.plot_neighbors(selected_site=0, plot_figure=False)
        trace_names = {trace.name for trace in fig.data}

        self.assertNotIn("voronoi cells", trace_names)
        self.assertIn("site", trace_names)
        self.assertIn("selected site", trace_names)

    def test_plot_neighbors_requires_neighborhood(self) -> None:
        system = gh.System.square_lattice(2, 2)

        with self.assertRaisesRegex(RuntimeError, "compute_neighborhood"):
            system.plot_neighbors(selected_site=0, plot_figure=False)

    def test_plot_neighbors_validates_index(self) -> None:
        system = gh.System.square_lattice(2, 2)

        with self.assertRaisesRegex(ValueError, "between 0 and 3"):
            system.plot_neighbors(selected_site=4, plot_figure=False)
        with self.assertRaises(TypeError):
            system.plot_neighbors(selected_site=True, plot_figure=False)

    def test_plot_neighbors_requires_2d_euclidean_system(self) -> None:
        system = gh.System.inverse(spherical_grid())
        with self.assertRaisesRegex(ValueError, "2D Euclidean"):
            system.plot_neighbors(selected_site=0, plot_figure=False)

    def test_explicit_config_constructor_sets_expected_config(self) -> None:
        system = gh.System(euclidean_grid(), config=gh.SystemConfig.euclidean(), num_spins=2)

        self.assertEqual(system.geometry, gh.Geometry.EUCLIDEAN)
        self.assertEqual(system.constraint, gh.ConstraintModel.FIXED_DENSITY)
        self.assertEqual(system.interaction_model, gh.InteractionModel.OCCUPANCY)
        self.assertEqual(system.update_rule, gh.UpdateRule.SWAP)
        self.assertEqual(system.system_shape, 4)

    def test_removed_legacy_constructor_shapes_raise_type_error(self) -> None:
        with self.assertRaises(TypeError):
            gh.System(spherical_grid(), system_type="non-antipodal")
        with self.assertRaises(TypeError):
            gh.System(spherical_grid(), geometry="spherical")

    def test_compute_neighborhood_requires_jump_distance(self) -> None:
        system = gh.System.inverse(spherical_grid())
        with self.assertRaisesRegex(ValueError, "jump_distance is required"):
            system.compute_neighborhood()

    def test_compute_neighborhood_storage_matches_config(self) -> None:
        inverse = gh.System.inverse(spherical_grid())
        inverse.compute_neighborhood(jump_distance=np.pi / 2, silent=True)
        self.assertIsInstance(inverse.neighborhood, list)

        non_antipodal = gh.System.non_antipodal(spherical_grid())
        non_antipodal.compute_neighborhood(jump_distance=np.pi / 2, silent=True)
        self.assertIsInstance(non_antipodal.neighborhood, gh.NonAntipodalNeighborhoodCSR)

        euclidean = gh.System.euclidean(euclidean_grid(), num_spins=2)
        euclidean.compute_neighborhood(jump_distance=1.0, silent=True)
        self.assertIsInstance(euclidean.neighborhood, gh.NonAntipodalNeighborhoodCSR)

    def test_euclidean_2d_l2_uses_ring_pruning(self) -> None:
        with mock.patch.object(
            euclidean_neighbors,
            "_ring_pruned_indices_2d",
            wraps=euclidean_neighbors._ring_pruned_indices_2d,
        ) as ring_prune, mock.patch.object(
            euclidean_neighbors,
            "_ball_pruned_indices",
            wraps=euclidean_neighbors._ball_pruned_indices,
        ) as ball_prune:
            rows = euclidean_neighbors.build_euclidean_neighborhood(
                euclidean_grid(),
                jump_distance=1.0,
                silent=True,
                distance_metric="euclidean",
            )

        self.assertEqual(ring_prune.call_count, len(euclidean_grid()))
        self.assertEqual(ball_prune.call_count, 0)
        self.assertEqual(len(rows), len(euclidean_grid()))

    def test_euclidean_3d_l2_uses_padded_ball_pruning(self) -> None:
        grid = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        with mock.patch.object(
            euclidean_neighbors,
            "_ring_pruned_indices_2d",
            wraps=euclidean_neighbors._ring_pruned_indices_2d,
        ) as ring_prune, mock.patch.object(
            euclidean_neighbors,
            "_ball_pruned_indices",
            wraps=euclidean_neighbors._ball_pruned_indices,
        ) as ball_prune:
            rows = euclidean_neighbors.build_euclidean_neighborhood(
                grid,
                jump_distance=1.0,
                silent=True,
                distance_metric="euclidean",
            )

        self.assertEqual(ring_prune.call_count, 0)
        self.assertEqual(ball_prune.call_count, len(grid))
        self.assertEqual(len(rows), len(grid))

    def test_euclidean_custom_metric_uses_brute_force_candidates(self) -> None:
        def custom_metric(points, spin):
            return np.linalg.norm(points - spin, axis=1)

        with mock.patch.object(
            euclidean_neighbors,
            "_ring_pruned_indices_2d",
            wraps=euclidean_neighbors._ring_pruned_indices_2d,
        ) as ring_prune, mock.patch.object(
            euclidean_neighbors,
            "_ball_pruned_indices",
            wraps=euclidean_neighbors._ball_pruned_indices,
        ) as ball_prune:
            rows = euclidean_neighbors.build_euclidean_neighborhood(
                euclidean_grid(),
                jump_distance=1.0,
                silent=True,
                distance_metric=custom_metric,
            )

        self.assertEqual(ring_prune.call_count, 0)
        self.assertEqual(ball_prune.call_count, 0)
        self.assertEqual(len(rows), len(euclidean_grid()))

    def test_unknown_kernel_raises_consistently(self) -> None:
        system = gh.System.inverse(spherical_grid())

        with self.assertRaisesRegex(ValueError, "Unknown kernel"):
            system.compute_neighborhood(jump_distance=np.pi / 2, kernel="twpo")
        with self.assertRaisesRegex(ValueError, "Unknown kernel"):
            resolve_delta_kernel("twpo")

    def test_compute_neighborhood_accepts_callable_kernel(self) -> None:
        def custom_delta(distances, cell_length):
            mask = distances <= cell_length
            return np.ones(np.count_nonzero(mask), dtype=float), mask

        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(
            jump_distance=1.0,
            kernel=custom_delta,
            silent=True,
        )

        attrs = system.to_dataset().attrs
        self.assertIs(system.kernel, custom_delta)
        self.assertTrue(bool(attrs["kernel_custom"]))
        self.assertEqual(attrs["kernel_kind"], "custom")
        self.assertIn("custom_delta", attrs["kernel_repr"])
        self.assertIsInstance(system.neighborhood, gh.NonAntipodalNeighborhoodCSR)

    def test_loaded_custom_kernel_requires_callable_to_rebuild(self) -> None:
        def custom_delta(distances, cell_length):
            mask = distances <= cell_length
            return np.ones(np.count_nonzero(mask), dtype=float), mask

        test_dir = make_test_dir(self.id().replace(".", "_"))
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(
            jump_distance=1.0,
            kernel=custom_delta,
            silent=True,
        )
        system.array_init(np.array([True, False, True, False], dtype=bool))
        save_path = system.save(test_dir)

        loaded = gh.System.euclidean(euclidean_grid(), num_spins=2)
        loaded.load(save_path)

        with self.assertRaisesRegex(ValueError, "custom kernel"):
            loaded.compute_neighborhood(silent=True)

        loaded.compute_neighborhood(kernel=custom_delta, silent=True)
        self.assertIs(loaded.kernel, custom_delta)
        self.assertIsInstance(loaded.neighborhood, gh.NonAntipodalNeighborhoodCSR)

    def test_seeded_random_init_is_reproducible_without_consuming_global_rng(self) -> None:
        np.random.seed(777)
        expected_next = np.random.random()
        np.random.seed(777)

        first = gh.System.euclidean(euclidean_grid(), num_spins=2)
        first.compute_neighborhood(jump_distance=1.0, silent=True)
        first.random_init(seed=42)
        actual_next = np.random.random()

        second = gh.System.euclidean(euclidean_grid(), num_spins=2)
        second.compute_neighborhood(jump_distance=1.0, silent=True)
        second.random_init(seed=42)

        self.assertEqual(actual_next, expected_next)
        self.assertTrue(np.array_equal(first.current_system, second.current_system))

    def test_seeded_hemisphere_init_is_reproducible_without_global_reseed(self) -> None:
        np.random.seed(888)
        expected_next = np.random.random()
        np.random.seed(888)

        first = gh.System.inverse(spherical_grid())
        first.compute_neighborhood(jump_distance=np.pi / 2, silent=True)
        first.hemisphere_init(seed=123)
        actual_next = np.random.random()

        second = gh.System.inverse(spherical_grid())
        second.compute_neighborhood(jump_distance=np.pi / 2, silent=True)
        second.hemisphere_init(seed=123)

        self.assertEqual(actual_next, expected_next)
        self.assertTrue(np.array_equal(first.current_system, second.current_system))

    def test_seed_validation_rejects_bool(self) -> None:
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(jump_distance=1.0, silent=True)

        with self.assertRaises(TypeError):
            system.random_init(seed=True)

    def test_neighborhood_rows_normalize_and_are_frozen(self) -> None:
        neighborhood = gh.Neighborhood(
            positive_spins=np.array([5, 2, 3]),
            positive_contributions=np.array([0.5, 0.2, 0.3]),
            negative_spins=np.array([4, 1]),
            negative_contributions=np.array([0.4, 0.1]),
            spin_potential=np.float32(1.25),
        )
        non_antipodal = gh.NonAntipodalNeighborhood(
            neighbor_spins=np.array([4, 1, 3]),
            contributions=np.array([0.4, 0.1, 0.3]),
            spin_potential=np.float32(2.5),
        )

        self.assertTrue(
            np.array_equal(neighborhood.positive_spins, np.array([2, 3, 5]))
        )
        self.assertTrue(
            np.array_equal(
                neighborhood.positive_contributions,
                np.array([0.2, 0.3, 0.5]),
            )
        )
        self.assertTrue(
            np.array_equal(neighborhood.negative_spins, np.array([1, 4]))
        )
        self.assertTrue(
            np.array_equal(
                neighborhood.negative_contributions,
                np.array([0.1, 0.4]),
            )
        )
        self.assertEqual(neighborhood.spin_potential, 1.25)
        self.assertTrue(
            np.array_equal(non_antipodal.neighbor_spins, np.array([1, 3, 4]))
        )
        self.assertTrue(
            np.array_equal(non_antipodal.contributions, np.array([0.1, 0.3, 0.4]))
        )
        self.assertEqual(non_antipodal.spin_potential, 2.5)

        with self.assertRaises(FrozenInstanceError):
            neighborhood.spin_potential = 0.0

    def test_save_directory_and_explicit_file_paths(self) -> None:
        test_dir = make_test_dir(self.id().replace(".", "_"))
        system = gh.System.inverse(spherical_grid())
        system.compute_neighborhood(jump_distance=np.pi / 2, silent=True)
        system.array_init(np.array([True, False, True], dtype=bool))

        folder_path = system.save(test_dir / "folder_target")
        file_path = system.save(test_dir / "explicit_state.nc")

        self.assertEqual(folder_path.name, "system_state.nc")
        self.assertTrue(folder_path.exists())
        self.assertEqual(file_path, test_dir / "explicit_state.nc")
        self.assertTrue(file_path.exists())

    def test_to_dataset_uses_current_persistence_schema(self) -> None:
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(jump_distance=1.0, silent=True)
        system.array_init(np.array([True, False, True, False], dtype=bool))

        attrs = system.to_dataset().attrs

        self.assertEqual(attrs["format_version"], "4.0")
        self.assertEqual(attrs["geometry"], "euclidean")
        self.assertEqual(attrs["constraint"], "fixed-density")
        self.assertEqual(attrs["interaction_model"], "occupancy")
        self.assertEqual(attrs["update_rule"], "swap")
        self.assertEqual(attrs["kernel_kind"], "cosine")
        self.assertFalse(bool(attrs["kernel_custom"]))
        self.assertEqual(attrs["distance_metric"], "euclidean")
        self.assertFalse(bool(attrs["distance_metric_custom"]))
        self.assertEqual(attrs["dimension"], 2)
        self.assertEqual(json.loads(attrs["grid_shape_json"]), [4, 2])
        self.assertEqual(attrs["grid_dtype"], str(system.grid.dtype))
        self.assertIn("jump_distance", attrs)
        self.assertNotIn("prob_prefactor", attrs)
        self.assertTrue(bool(attrs["neighborhood_built"]))
        self.assertEqual(attrs["neighborhood_storage"], "csr")
        self.assertEqual(attrs["neighborhood_rows"], 4)
        self.assertIn("neighborhood_edges", attrs)
        self.assertIn("anneal_data_json", attrs)
        self.assertIn("greedy_history_json", attrs)
        self.assertIn("parallel_tempering_history_json", attrs)
        self.assertNotIn("system_type", attrs)
        self.assertNotIn("antipodal", attrs)
        self.assertNotIn("constraint_model", attrs)
        self.assertNotIn("system_shape_json", attrs)

    def test_load_restores_distance_metric_metadata_for_rebuild(self) -> None:
        test_dir = make_test_dir(self.id().replace(".", "_"))
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(
            jump_distance=1.0,
            silent=True,
            distance_metric="manhattan",
        )
        system.array_init(np.array([True, False, True, False], dtype=bool))
        save_path = system.save(test_dir)

        loaded = gh.System.euclidean(euclidean_grid(), num_spins=2)
        loaded.load(save_path)

        self.assertEqual(loaded.distance_metric, "manhattan")
        self.assertIsNone(loaded.neighborhood)

        loaded.compute_neighborhood(silent=True)
        self.assertEqual(loaded.distance_metric, "manhattan")
        self.assertIsInstance(loaded.neighborhood, gh.NonAntipodalNeighborhoodCSR)


if __name__ == "__main__":
    unittest.main()
