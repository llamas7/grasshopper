from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from unittest import mock
import unittest

import matplotlib.pyplot as plt
import numpy as np

import grasshopper as gh
import grasshopper.euclidean.build_neighbors as euclidean_neighbors
from grasshopper.kernels import (
    calculate_contributions,
    cosine_kernel,
)

from tests.helpers import cleanup_tmp_root, euclidean_grid, make_test_dir, spherical_grid


class SystemApiTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        cleanup_tmp_root()

    def setUp(self) -> None:
        np.random.seed(12345)

    def tearDown(self) -> None:
        plt.close("all")

    @staticmethod
    def _antipodal_row_dict(row) -> tuple[dict[int, float], dict[int, float]]:
        positive = {
            int(index): float(value)
            for index, value in zip(row.positive_spins, row.positive_contributions)
        }
        negative = {
            int(index): float(value)
            for index, value in zip(row.negative_spins, row.negative_contributions)
        }
        return positive, negative

    @staticmethod
    def _fixed_density_row_dict(row) -> dict[int, float]:
        return {
            int(index): float(value)
            for index, value in zip(row.neighbor_spins, row.contributions)
        }

    def _assert_weighted_antipodal_rows_equal(
        self,
        multi,
        scalar_systems,
        weights,
    ) -> None:
        for row_index, multi_row in enumerate(multi.neighborhood):
            expected_positive: dict[int, float] = {}
            expected_negative: dict[int, float] = {}
            for scalar, weight in zip(scalar_systems, weights):
                positive, negative = self._antipodal_row_dict(
                    scalar.neighborhood[row_index]
                )
                for index, value in positive.items():
                    expected_positive[index] = expected_positive.get(index, 0.0) + (
                        weight * value
                    )
                for index, value in negative.items():
                    expected_negative[index] = expected_negative.get(index, 0.0) + (
                        weight * value
                    )

            actual_positive, actual_negative = self._antipodal_row_dict(multi_row)
            self.assertEqual(actual_positive.keys(), expected_positive.keys())
            self.assertEqual(actual_negative.keys(), expected_negative.keys())
            for index, expected in expected_positive.items():
                self.assertAlmostEqual(actual_positive[index], expected)
            for index, expected in expected_negative.items():
                self.assertAlmostEqual(actual_negative[index], expected)
            self.assertAlmostEqual(
                multi_row.total_weight,
                sum(expected_positive.values()) + sum(expected_negative.values()),
            )

    def _assert_weighted_fixed_density_rows_equal(
        self,
        multi,
        scalar_systems,
        weights,
    ) -> None:
        for row_index in range(len(multi.neighborhood)):
            expected_row: dict[int, float] = {}
            for scalar, weight in zip(scalar_systems, weights):
                row = self._fixed_density_row_dict(scalar.neighborhood[row_index])
                for index, value in row.items():
                    expected_row[index] = expected_row.get(index, 0.0) + (
                        weight * value
                    )

            multi_row = multi.neighborhood[row_index]
            actual_row = self._fixed_density_row_dict(multi_row)
            self.assertEqual(actual_row.keys(), expected_row.keys())
            for index, expected in expected_row.items():
                self.assertAlmostEqual(actual_row[index], expected)
            self.assertAlmostEqual(
                multi_row.total_weight,
                sum(expected_row.values()),
            )

    def test_named_constructors_set_expected_configs(self) -> None:
        system = gh.System.fixed_density(spherical_grid())

        self.assertIs(system.kind, gh.SystemKind.FIXED_DENSITY)
        self.assertTrue(system.kind.spherical)
        self.assertTrue(system.kind.fixed_density)
        self.assertFalse(system.antipodal)

        complementary = gh.System.complementary(spherical_grid())
        self.assertIs(complementary.kind, gh.SystemKind.COMPLEMENTARY)
        self.assertTrue(complementary.antipodal)

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

        triangular = gh.triangular_lattice(2, 2, spacing=1.0, compact=False)
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

        self.assertIs(system.kind, gh.SystemKind.EUCLIDEAN)
        self.assertEqual(system.grid.shape, (12, 2))
        self.assertEqual(system.num_spins, 3)
        self.assertEqual(system.system_shape, 12)

        honeycomb = gh.System.regular_lattice("honeycomb", 2, 2)
        self.assertEqual(honeycomb.grid.shape, (8, 2))
        self.assertEqual(honeycomb.num_spins, 4)

        triangular = gh.System.regular_lattice(
            "tri",
            3,
            2,
            num_spins=2,
            center=True,
        )
        self.assertIs(triangular.kind, gh.SystemKind.EUCLIDEAN)
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
                gh.SystemKind.EUCLIDEAN,
                num_spins=2,
                normalize_euclidean=False,
            )

    def test_kernel_drops_zero_weight_cutoff_neighbors(self) -> None:
        contributions, mask, total_weight = calculate_contributions(
            jump_distance=0.30,
            cell_length=0.01,
            distances=np.array([0.31999999999999995, 0.32000000000000006]),
            kernel_func=cosine_kernel,
        )

        self.assertEqual(contributions.size, 0)
        self.assertFalse(np.any(mask))
        self.assertEqual(total_weight, 0.0)

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

        manual_system.compute_neighborhood(jump_distance=jump_distance)
        builtin_system.compute_neighborhood(jump_distance=jump_distance)
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
                manual_row.total_weight,
                builtin_row.total_weight,
            )

    def test_regular_lattice_plotting(self) -> None:
        system = gh.System.square_lattice(2, 2)

        self.assertFalse(hasattr(system, "plot_lattice"))

        system.compute_neighborhood(jump_distance=1.0)
        system.random_init(seed=1)
        spin_fig = system.plot(state="current", show_edges=True, plot_figure=False)
        self.assertIsInstance(spin_fig, plt.Figure)
        labels = set(spin_fig.axes[0].get_legend_handles_labels()[1])
        self.assertIn("A", labels)
        self.assertIn("B", labels)
        self.assertIn("lattice edges", labels)
        self.assertEqual(len(spin_fig.axes[0].patches), 0)

    def test_euclidean_plot_uses_matplotlib_in_one_and_three_dimensions(self) -> None:
        one_dimensional = gh.System.euclidean(
            np.array([[0.0], [1.0], [2.0]]),
            num_spins=2,
        )
        one_dimensional_figure = one_dimensional.plot(plot_figure=False)

        self.assertIsInstance(one_dimensional_figure, plt.Figure)
        self.assertFalse(one_dimensional_figure.axes[0].get_yaxis().get_visible())

        three_dimensional = gh.System.euclidean(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            num_spins=2,
        )
        three_dimensional_figure = three_dimensional.plot(plot_figure=False)

        self.assertIsInstance(three_dimensional_figure, plt.Figure)
        self.assertEqual(three_dimensional_figure.axes[0].name, "3d")

    def test_euclidean_plot_voronoi_colors_cells_by_spin_state(self) -> None:
        system = gh.System.square_lattice(3, 3, num_spins=4)
        system.compute_neighborhood(jump_distance=1.0)
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

        fig = system.plot(state="current", voronoi=True, plot_figure=False)
        ax = fig.axes[0]
        patch_labels = {patch.get_label() for patch in ax.patches}

        self.assertIn("A", patch_labels)
        self.assertIn("B", patch_labels)
        self.assertEqual(len(ax.patches), len(system.grid))
        self.assertTrue(all(patch.get_gid().startswith("site=") for patch in ax.patches))
        self.assertEqual(ax.get_aspect(), 1.0)

    def test_voronoi_padding_uses_half_cell_length(self) -> None:
        system = gh.System.square_lattice(3, 3, num_spins=4)
        system.compute_neighborhood(jump_distance=1.5)
        system.random_init(seed=1)
        expected_padding = ((1.0 / system.num_spins) ** (1.0 / system.dimension)) / 2.0

        with mock.patch("grasshopper.system.plot_spin_system") as plot_spin_system:
            system.plot(state="current", voronoi=True, plot_figure=False)
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
        system = gh.System.complementary(spherical_grid())
        system.compute_neighborhood(jump_distance=np.pi / 2)
        system.hemisphere_init(seed=1)

        with self.assertRaisesRegex(ValueError, "2D Euclidean"):
            system.plot(voronoi=True, plot_figure=False)

    def test_plot_neighbors_selected_site_colors_cells(self) -> None:
        system = gh.System.square_lattice(3, 3, num_spins=4)
        system.compute_neighborhood(
            jump_distance=1.5,
        )

        fig = system.plot_neighbors(
            selected_site=4,
            plot_figure=False,
        )
        ax = fig.axes[0]
        labels = set(ax.get_legend_handles_labels()[1])
        patch_labels = {patch.get_label() for patch in ax.patches}

        self.assertIn("voronoi cells", patch_labels)
        self.assertIn("selected voronoi cell", patch_labels)
        self.assertIn("neighbor contribution cells", patch_labels)
        self.assertIn("selected site", labels)
        self.assertNotIn("lattice edges", labels)
        selected_cell = next(
            patch
            for patch in ax.patches
            if patch.get_label() == "selected voronoi cell"
        )
        self.assertAlmostEqual(selected_cell.get_alpha(), 0.40)
        neighbor_cells = [
            patch
            for patch in ax.patches
            if "contribution=" in patch.get_gid()
        ]
        self.assertTrue(neighbor_cells)
        self.assertTrue(
            any(
                patch.get_label() == "neighbor contribution cells"
                for patch in neighbor_cells
            )
        )

    def test_plot_neighbors_skips_tiny_grids(self) -> None:
        grid = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
        system = gh.System.euclidean(grid, num_spins=1)
        system.compute_neighborhood(jump_distance=1.0)

        fig = system.plot_neighbors(selected_site=0, plot_figure=False)
        ax = fig.axes[0]
        labels = set(ax.get_legend_handles_labels()[1])

        self.assertEqual(len(ax.patches), 0)
        self.assertIn("site", labels)
        self.assertIn("selected site", labels)

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
        system = gh.System.complementary(spherical_grid())
        with self.assertRaisesRegex(ValueError, "2D Euclidean"):
            system.plot_neighbors(selected_site=0, plot_figure=False)

    def test_explicit_config_constructor_sets_expected_config(self) -> None:
        system = gh.System(euclidean_grid(), "euclidean", num_spins=2)

        self.assertIs(system.kind, gh.SystemKind.EUCLIDEAN)
        self.assertEqual(system.system_shape, 4)

    def test_removed_legacy_constructor_shapes_raise_type_error(self) -> None:
        with self.assertRaises(TypeError):
            gh.System(spherical_grid(), system_type="fixed-density")
        with self.assertRaises(TypeError):
            gh.System(spherical_grid(), geometry="spherical")

    def test_compute_neighborhood_requires_jump_distance(self) -> None:
        system = gh.System.complementary(spherical_grid())
        with self.assertRaisesRegex(ValueError, "jump_distance is required"):
            system.compute_neighborhood()

    def test_compute_neighborhood_storage_matches_config(self) -> None:
        complementary = gh.System.complementary(spherical_grid())
        complementary.compute_neighborhood(jump_distance=np.pi / 2)
        self.assertIsInstance(complementary.neighborhood, list)

        fixed_density = gh.System.fixed_density(spherical_grid())
        fixed_density.compute_neighborhood(jump_distance=np.pi / 2)
        self.assertIsInstance(fixed_density.neighborhood, gh.FixedDensityNeighborhoodCSR)

        euclidean = gh.System.euclidean(euclidean_grid(), num_spins=2)
        euclidean.compute_neighborhood(jump_distance=1.0)
        self.assertIsInstance(euclidean.neighborhood, gh.FixedDensityNeighborhoodCSR)

    def test_setup_methods_return_the_same_system_for_chaining(self) -> None:
        complementary = gh.System.complementary(spherical_grid())
        self.assertIs(
            complementary.compute_neighborhood(jump_distance=np.pi / 2),
            complementary,
        )
        self.assertIs(complementary.random_init(seed=1), complementary)

        hemisphere = gh.System.complementary(spherical_grid())
        hemisphere.compute_neighborhood(jump_distance=np.pi / 2)
        self.assertIs(hemisphere.hemisphere_init(seed=2), hemisphere)

        triangle = gh.System.complementary(spherical_grid())
        triangle.compute_neighborhood(jump_distance=np.pi / 2)
        self.assertIs(triangle.triangle_cogwheel_init(num_cogs=3, cog_height=0.1), triangle)

        cosine = gh.System.complementary(spherical_grid())
        cosine.compute_neighborhood(jump_distance=np.pi / 2)
        self.assertIs(cosine.cosine_cogwheel_init(num_cogs=3, cog_height=0.1), cosine)

        explicit = gh.System.complementary(spherical_grid())
        explicit.compute_neighborhood(jump_distance=np.pi / 2)
        self.assertIs(
            explicit.array_init(np.array([True, False, True], dtype=bool)),
            explicit,
        )

    def test_state_summary_tracks_lifecycle_and_metadata(self) -> None:
        system = gh.System.complementary(spherical_grid())

        initial = system.state_summary()
        self.assertEqual(initial["kind"], "complementary")
        self.assertEqual(initial["site_count"], 6)
        self.assertEqual(initial["num_spins"], 3)
        self.assertEqual(initial["dimension"], 3)
        self.assertEqual(initial["system_shape"], 3)
        self.assertFalse(initial["neighborhood_built"])
        self.assertFalse(initial["initialized"])
        self.assertFalse(initial["has_current_spins"])
        self.assertFalse(initial["has_best_spins"])
        self.assertFalse(initial["ready"])
        self.assertIsNone(initial["jump_distance"])
        self.assertIsNone(initial["jump_weights"])
        self.assertEqual(initial["kernel"], "cosine")
        self.assertIsNone(initial["distance_metric"])

        system.compute_neighborhood(
            jump_distance=[0.48 * np.pi, 0.52 * np.pi],
            jump_weights=[1.0, 0.75],
        )
        built = system.state_summary()
        self.assertTrue(built["neighborhood_built"])
        self.assertFalse(built["initialized"])
        self.assertFalse(built["ready"])
        self.assertEqual(built["jump_distance"], (0.48 * np.pi, 0.52 * np.pi))
        self.assertEqual(built["jump_weights"], (1.0, 0.75))
        self.assertEqual(built["distance_metric"], "spherical")

        system.array_init(np.array([True, False, True], dtype=bool))
        ready = system.state_summary()
        self.assertTrue(ready["initialized"])
        self.assertTrue(ready["has_current_spins"])
        self.assertTrue(ready["has_best_spins"])
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["current_energy"], system.current_energy)
        self.assertEqual(ready["best_energy"], system.best_energy)
        self.assertEqual(ready["temperature"], system.temperature)

    def test_scalar_jump_distance_metadata_remains_backward_compatible(self) -> None:
        system = gh.System.complementary(spherical_grid())
        system.compute_neighborhood(jump_distance=np.pi / 2)

        attrs = system.to_dataset().attrs

        self.assertIsInstance(system.jump_distance, float)
        self.assertEqual(system.jump_weights, (1.0,))
        self.assertIn("jump_distance", attrs)
        self.assertNotIn("jump_distances_json", attrs)
        self.assertNotIn("jump_weights_json", attrs)

    def test_weighted_multi_shell_complementary_matches_manual_scalar_sum(self) -> None:
        distances = (0.48 * np.pi, 0.52 * np.pi)
        weights = (1.0, 0.75)
        multi = gh.System.complementary(spherical_grid())
        multi.compute_neighborhood(
            jump_distance=distances,
            jump_weights=weights,
        )
        scalar_systems = []
        for distance in distances:
            scalar = gh.System.complementary(spherical_grid())
            scalar.compute_neighborhood(jump_distance=distance)
            scalar_systems.append(scalar)

        self.assertEqual(multi.jump_distance, distances)
        self.assertEqual(multi.jump_weights, weights)
        self._assert_weighted_antipodal_rows_equal(multi, scalar_systems, weights)

    def test_weighted_multi_shell_independent_matches_manual_scalar_sum(self) -> None:
        distances = (0.48 * np.pi, 0.52 * np.pi)
        weights = (1.0, 0.75)
        multi = gh.System.independent(spherical_grid())
        multi.compute_neighborhood(
            jump_distance=distances,
            jump_weights=weights,
        )
        scalar_systems = []
        for distance in distances:
            scalar = gh.System.independent(spherical_grid())
            scalar.compute_neighborhood(jump_distance=distance)
            scalar_systems.append(scalar)

        self._assert_weighted_antipodal_rows_equal(multi, scalar_systems, weights)

    def test_weighted_multi_shell_fixed_density_matches_manual_scalar_sum(self) -> None:
        distances = (0.48 * np.pi, 0.52 * np.pi)
        weights = (1.0, 0.75)
        multi = gh.System.fixed_density(spherical_grid())
        multi.compute_neighborhood(
            jump_distance=distances,
            jump_weights=weights,
        )
        scalar_systems = []
        for distance in distances:
            scalar = gh.System.fixed_density(spherical_grid())
            scalar.compute_neighborhood(jump_distance=distance)
            scalar_systems.append(scalar)

        self._assert_weighted_fixed_density_rows_equal(
            multi,
            scalar_systems,
            weights,
        )

    def test_multi_shell_validation_errors_are_clear(self) -> None:
        system = gh.System.complementary(spherical_grid())
        with self.assertRaisesRegex(ValueError, "non-empty 1D sequence"):
            system.compute_neighborhood(jump_distance=[])
        with self.assertRaisesRegex(ValueError, "finite"):
            system.compute_neighborhood(
                jump_distance=[np.pi / 2, np.nan],
            )
        with self.assertRaisesRegex(ValueError, "one finite value per jump distance"):
            system.compute_neighborhood(
                jump_distance=[np.pi / 3, np.pi / 2],
                jump_weights=[1.0],
            )
        with self.assertRaisesRegex(ValueError, "jump_weights values must be finite"):
            system.compute_neighborhood(
                jump_distance=[np.pi / 3, np.pi / 2],
                jump_weights=[1.0, np.inf],
            )

        euclidean = gh.System.euclidean(euclidean_grid(), num_spins=2)
        with self.assertRaisesRegex(ValueError, "spherical systems"):
            euclidean.compute_neighborhood(jump_distance=[1.0, 1.5])

    def test_multi_shell_save_load_restores_rebuild_metadata(self) -> None:
        test_dir = make_test_dir(self.id().replace(".", "_"))
        distances = (0.48 * np.pi, 0.52 * np.pi)
        weights = (1.0, 0.75)
        system = gh.System.fixed_density(spherical_grid())
        system.compute_neighborhood(
            jump_distance=distances,
            jump_weights=weights,
        )
        system.array_init(np.array([True, False, True, False, True, False]))
        save_path = system.save(test_dir)

        attrs = system.to_dataset().attrs
        self.assertNotIn("jump_distance", attrs)
        self.assertEqual(json.loads(attrs["jump_distances_json"]), list(distances))
        self.assertEqual(json.loads(attrs["jump_weights_json"]), list(weights))

        loaded = gh.System.fixed_density(spherical_grid())
        loaded.load(save_path)

        self.assertEqual(loaded.jump_distance, distances)
        self.assertEqual(loaded.jump_weights, weights)
        self.assertIsNone(loaded.neighborhood)
        loaded.compute_neighborhood()
        self._assert_weighted_fixed_density_rows_equal(
            loaded,
            [system],
            [1.0],
        )

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
                distance_metric=custom_metric,
            )

        self.assertEqual(ring_prune.call_count, 0)
        self.assertEqual(ball_prune.call_count, 0)
        self.assertEqual(len(rows), len(euclidean_grid()))

    def test_unknown_kernel_raises_consistently(self) -> None:
        system = gh.System.complementary(spherical_grid())

        with self.assertRaisesRegex(ValueError, "Unknown kernel"):
            system.compute_neighborhood(jump_distance=np.pi / 2, kernel="twpo")
        with self.assertRaisesRegex(ValueError, "Unknown kernel"):
            gh.KernelSpec("twpo")

    def test_removed_public_api_is_unavailable(self) -> None:
        system = gh.System.complementary(spherical_grid())

        self.assertFalse(hasattr(gh.System, "inverse"))
        self.assertFalse(hasattr(gh, "SystemConfig"))
        self.assertFalse(hasattr(gh.SystemKind, "INVERSE"))
        self.assertFalse(hasattr(gh, "DeltaType"))
        self.assertFalse(hasattr(gh.KernelSpec, "two"))
        self.assertFalse(hasattr(gh.KernelSpec, "one_star"))
        self.assertFalse(hasattr(gh.KernelSpec, "two_star"))
        self.assertFalse(hasattr(system, "delta"))
        self.assertFalse(hasattr(system, "prob_prefactor"))
        with self.assertRaises(TypeError):
            system.compute_neighborhood(jump_distance=np.pi / 2, delta="cosine")

    def test_compute_neighborhood_accepts_callable_kernel(self) -> None:
        def custom_delta(distances, cell_length):
            mask = distances <= cell_length
            return np.ones(np.count_nonzero(mask), dtype=float), mask

        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(
            jump_distance=1.0,
            kernel=custom_delta,
        )

        attrs = system.to_dataset().attrs
        self.assertIs(system.kernel, custom_delta)
        self.assertTrue(bool(attrs["kernel_custom"]))
        self.assertEqual(attrs["kernel_kind"], "custom")
        self.assertIn("custom_delta", attrs["kernel_repr"])
        self.assertIsInstance(system.neighborhood, gh.FixedDensityNeighborhoodCSR)

    def test_loaded_custom_kernel_requires_callable_to_rebuild(self) -> None:
        def custom_delta(distances, cell_length):
            mask = distances <= cell_length
            return np.ones(np.count_nonzero(mask), dtype=float), mask

        test_dir = make_test_dir(self.id().replace(".", "_"))
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(
            jump_distance=1.0,
            kernel=custom_delta,
        )
        system.array_init(np.array([True, False, True, False], dtype=bool))
        save_path = system.save(test_dir)

        loaded = gh.System.euclidean(euclidean_grid(), num_spins=2)
        loaded.load(save_path)

        with self.assertRaisesRegex(ValueError, "custom kernel"):
            loaded.compute_neighborhood()

        loaded.compute_neighborhood(kernel=custom_delta)
        self.assertIs(loaded.kernel, custom_delta)
        self.assertIsInstance(loaded.neighborhood, gh.FixedDensityNeighborhoodCSR)

    def test_seeded_random_init_is_reproducible_without_consuming_global_rng(self) -> None:
        np.random.seed(777)
        expected_next = np.random.random()
        np.random.seed(777)

        first = gh.System.euclidean(euclidean_grid(), num_spins=2)
        first.compute_neighborhood(jump_distance=1.0)
        first.random_init(seed=42)
        actual_next = np.random.random()

        second = gh.System.euclidean(euclidean_grid(), num_spins=2)
        second.compute_neighborhood(jump_distance=1.0)
        second.random_init(seed=42)

        self.assertEqual(actual_next, expected_next)
        self.assertTrue(np.array_equal(first.current_spins, second.current_spins))

    def test_seeded_hemisphere_init_is_reproducible_without_global_reseed(self) -> None:
        np.random.seed(888)
        expected_next = np.random.random()
        np.random.seed(888)

        first = gh.System.complementary(spherical_grid())
        first.compute_neighborhood(jump_distance=np.pi / 2)
        first.hemisphere_init(seed=123)
        actual_next = np.random.random()

        second = gh.System.complementary(spherical_grid())
        second.compute_neighborhood(jump_distance=np.pi / 2)
        second.hemisphere_init(seed=123)

        self.assertEqual(actual_next, expected_next)
        self.assertTrue(np.array_equal(first.current_spins, second.current_spins))

    def test_seed_validation_rejects_bool(self) -> None:
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(jump_distance=1.0)

        with self.assertRaises(TypeError):
            system.random_init(seed=True)

    def test_neighborhood_rows_normalize_and_are_frozen(self) -> None:
        neighborhood = gh.Neighborhood(
            positive_spins=np.array([5, 2, 3]),
            positive_contributions=np.array([0.5, 0.2, 0.3]),
            negative_spins=np.array([4, 1]),
            negative_contributions=np.array([0.4, 0.1]),
            total_weight=np.float32(1.25),
        )
        fixed_density = gh.FixedDensityNeighborhood(
            neighbor_spins=np.array([4, 1, 3]),
            contributions=np.array([0.4, 0.1, 0.3]),
            total_weight=np.float32(2.5),
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
        self.assertEqual(neighborhood.total_weight, 1.25)
        self.assertTrue(
            np.array_equal(fixed_density.neighbor_spins, np.array([1, 3, 4]))
        )
        self.assertTrue(
            np.array_equal(fixed_density.contributions, np.array([0.1, 0.3, 0.4]))
        )
        self.assertEqual(fixed_density.total_weight, 2.5)

        with self.assertRaises(FrozenInstanceError):
            neighborhood.total_weight = 0.0
        removed_name = "spin_" + "potential"
        self.assertFalse(hasattr(neighborhood, removed_name))
        with self.assertRaises(TypeError):
            gh.FixedDensityNeighborhood(
                neighbor_spins=np.array([], dtype=int),
                contributions=np.array([], dtype=float),
                **{removed_name: 0.0},
            )

    def test_save_directory_and_explicit_file_paths(self) -> None:
        test_dir = make_test_dir(self.id().replace(".", "_"))
        system = gh.System.complementary(spherical_grid())
        system.compute_neighborhood(jump_distance=np.pi / 2)
        system.array_init(np.array([True, False, True], dtype=bool))

        folder_path = system.save(test_dir / "folder_target")
        file_path = system.save(test_dir / "explicit_state.nc")

        self.assertEqual(folder_path.name, "system_state.nc")
        self.assertTrue(folder_path.exists())
        self.assertEqual(file_path, test_dir / "explicit_state.nc")
        self.assertTrue(file_path.exists())

    def test_to_dataset_uses_current_persistence_schema(self) -> None:
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(jump_distance=1.0)
        system.array_init(np.array([True, False, True, False], dtype=bool))

        attrs = system.to_dataset().attrs

        self.assertEqual(attrs["format_version"], "7.0")
        self.assertEqual(attrs["kind"], "euclidean")
        self.assertEqual(attrs["kernel_kind"], "cosine")
        self.assertFalse(bool(attrs["kernel_custom"]))
        self.assertEqual(attrs["distance_metric"], "euclidean")
        self.assertFalse(bool(attrs["distance_metric_custom"]))
        self.assertEqual(attrs["dimension"], 2)
        self.assertEqual(json.loads(attrs["grid_shape_json"]), [4, 2])
        self.assertEqual(attrs["grid_dtype"], str(system.grid.dtype))
        self.assertIn("jump_distance", attrs)
        self.assertNotIn("prob_prefactor", attrs)
        self.assertAlmostEqual(
            attrs["current_probability"], system.current_probability
        )
        self.assertAlmostEqual(attrs["best_probability"], system.best_probability)
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
            distance_metric="manhattan",
        )
        system.array_init(np.array([True, False, True, False], dtype=bool))
        save_path = system.save(test_dir)

        loaded = gh.System.euclidean(euclidean_grid(), num_spins=2)
        loaded.load(save_path)

        self.assertEqual(loaded.distance_metric, "manhattan")
        self.assertIsNone(loaded.neighborhood)

        loaded.compute_neighborhood()
        self.assertEqual(loaded.distance_metric, "manhattan")
        self.assertIsInstance(loaded.neighborhood, gh.FixedDensityNeighborhoodCSR)

    def test_load_can_rebuild_builtin_neighborhood_metadata(self) -> None:
        test_dir = make_test_dir(self.id().replace(".", "_"))
        state = np.array([True, False, True, False], dtype=bool)
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(
            jump_distance=1.0,
            kernel=gh.KernelSpec.gaussian(0.2, cutoff=5.0),
        )
        system.array_init(state)
        save_path = system.save(test_dir)

        loaded_without_rebuild = gh.System.euclidean(euclidean_grid(), num_spins=2)
        loaded_without_rebuild.load(save_path)
        self.assertIsNone(loaded_without_rebuild.neighborhood)
        self.assertFalse(loaded_without_rebuild.state_summary()["ready"])

        loaded = gh.System.euclidean(euclidean_grid(), num_spins=2)
        loaded.load(save_path, rebuild_neighborhood=True)

        self.assertIsInstance(loaded.neighborhood, gh.FixedDensityNeighborhoodCSR)
        self.assertTrue(loaded.state_summary()["ready"])
        self.assertEqual(loaded.kernel, gh.KernelSpec.gaussian(0.2, cutoff=5.0))
        self.assertAlmostEqual(loaded.current_energy, system.current_energy)
        self.assertAlmostEqual(loaded.best_energy, system.best_energy)

    def test_load_rebuild_rejects_custom_kernel_metadata(self) -> None:
        def custom_kernel(distances, cell_length):
            mask = distances <= cell_length
            return np.ones(np.count_nonzero(mask), dtype=float), mask

        test_dir = make_test_dir(self.id().replace(".", "_"))
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(
            jump_distance=1.0,
            kernel=custom_kernel,
        )
        system.array_init(np.array([True, False, True, False], dtype=bool))
        save_path = system.save(test_dir)

        loaded = gh.System.euclidean(euclidean_grid(), num_spins=2)
        with self.assertRaisesRegex(ValueError, "custom kernel"):
            loaded.load(save_path, rebuild_neighborhood=True)

        self.assertIsNone(loaded.neighborhood)
        self.assertTrue(loaded.initialized)
        self.assertEqual(loaded.state_summary()["kernel"], "custom")

    def test_load_rebuild_rejects_custom_shell_measure_metadata(self) -> None:
        test_dir = make_test_dir(self.id().replace(".", "_"))
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(
            jump_distance=1.0,
            distance_metric="manhattan",
            shell_measure=lambda radius: 8.0 * radius,
            radial_bounds=(0.0, 4.0),
        )
        system.array_init(np.array([True, False, True, False], dtype=bool))
        save_path = system.save(test_dir)

        loaded = gh.System.euclidean(euclidean_grid(), num_spins=2)
        with self.assertRaisesRegex(ValueError, "custom shell_measure"):
            loaded.load(save_path, rebuild_neighborhood=True)

        self.assertIsNone(loaded.neighborhood)
        self.assertTrue(loaded.initialized)

    def test_state_summary_after_load_reports_loaded_but_not_ready(self) -> None:
        test_dir = make_test_dir(self.id().replace(".", "_"))
        system = gh.System.complementary(spherical_grid())
        system.compute_neighborhood(jump_distance=np.pi / 2)
        system.array_init(np.array([True, False, True], dtype=bool))
        save_path = system.save(test_dir)

        loaded = gh.System.complementary(spherical_grid())
        loaded.load(save_path)
        summary = loaded.state_summary()

        self.assertTrue(summary["initialized"])
        self.assertTrue(summary["has_current_spins"])
        self.assertTrue(summary["has_best_spins"])
        self.assertFalse(summary["neighborhood_built"])
        self.assertFalse(summary["ready"])
        self.assertEqual(summary["jump_distance"], np.pi / 2)
        self.assertEqual(summary["jump_weights"], (1.0,))
        self.assertEqual(summary["kernel"], "cosine")
        self.assertEqual(summary["distance_metric"], "spherical")


if __name__ == "__main__":
    unittest.main()
