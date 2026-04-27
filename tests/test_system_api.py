from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

import numpy as np

import refactoredpc as rf

from tests.helpers import cleanup_tmp_root, euclidean_grid, make_test_dir, spherical_grid


class SystemApiTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        cleanup_tmp_root()

    def setUp(self) -> None:
        np.random.seed(12345)

    def test_named_constructors_set_expected_configs(self) -> None:
        system = rf.System.non_antipodal(spherical_grid())

        self.assertEqual(system.geometry, rf.Geometry.SPHERICAL)
        self.assertEqual(system.constraint, rf.ConstraintModel.FIXED_DENSITY)
        self.assertEqual(system.interaction_model, rf.InteractionModel.OCCUPANCY)
        self.assertEqual(system.update_rule, rf.UpdateRule.SWAP)
        self.assertEqual(system.config, rf.SystemConfig.non_antipodal())

        inverse = rf.System.inverse(spherical_grid())
        self.assertEqual(inverse.config, rf.SystemConfig.inverse())

    def test_explicit_config_constructor_sets_expected_config(self) -> None:
        system = rf.System(euclidean_grid(), config=rf.SystemConfig.euclidean(), num_spins=2)

        self.assertEqual(system.geometry, rf.Geometry.EUCLIDEAN)
        self.assertEqual(system.constraint, rf.ConstraintModel.FIXED_DENSITY)
        self.assertEqual(system.interaction_model, rf.InteractionModel.OCCUPANCY)
        self.assertEqual(system.update_rule, rf.UpdateRule.SWAP)
        self.assertEqual(system.system_shape, 4)

    def test_removed_legacy_constructor_shapes_raise_type_error(self) -> None:
        with self.assertRaises(TypeError):
            rf.System(spherical_grid(), system_type="non-antipodal")
        with self.assertRaises(TypeError):
            rf.System(spherical_grid(), geometry="spherical")

    def test_compute_neighborhood_requires_jump_distance(self) -> None:
        system = rf.System.inverse(spherical_grid())
        with self.assertRaisesRegex(ValueError, "jump_distance is required"):
            system.compute_neighborhood()

    def test_compute_neighborhood_storage_matches_config(self) -> None:
        inverse = rf.System.inverse(spherical_grid())
        inverse.compute_neighborhood(jump_distance=np.pi / 2, silent=True)
        self.assertIsInstance(inverse.neighborhood, list)

        non_antipodal = rf.System.non_antipodal(spherical_grid())
        non_antipodal.compute_neighborhood(jump_distance=np.pi / 2, silent=True)
        self.assertIsInstance(non_antipodal.neighborhood, rf.NonAntipodalNeighborhoodCSR)

        euclidean = rf.System.euclidean(euclidean_grid(), num_spins=2)
        euclidean.compute_neighborhood(jump_distance=1.0, silent=True)
        self.assertIsInstance(euclidean.neighborhood, rf.NonAntipodalNeighborhoodCSR)

    def test_neighborhood_rows_normalize_and_are_frozen(self) -> None:
        neighborhood = rf.Neighborhood(
            positive_spins=np.array([5, 2, 3]),
            positive_contributions=np.array([0.5, 0.2, 0.3]),
            negative_spins=np.array([4, 1]),
            negative_contributions=np.array([0.4, 0.1]),
            spin_potential=np.float32(1.25),
        )
        non_antipodal = rf.NonAntipodalNeighborhood(
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
        system = rf.System.inverse(spherical_grid())
        system.compute_neighborhood(jump_distance=np.pi / 2, silent=True)
        system.array_init(np.array([True, False, True], dtype=bool))

        folder_path = system.save(test_dir / "folder_target")
        file_path = system.save(test_dir / "explicit_state.nc")

        self.assertEqual(folder_path.name, "system_state.nc")
        self.assertTrue(folder_path.exists())
        self.assertEqual(file_path, test_dir / "explicit_state.nc")
        self.assertTrue(file_path.exists())

    def test_to_dataset_uses_current_persistence_schema(self) -> None:
        system = rf.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(jump_distance=1.0, silent=True)
        system.array_init(np.array([True, False, True, False], dtype=bool))

        attrs = system.to_dataset().attrs

        self.assertEqual(attrs["format_version"], "3.0")
        self.assertEqual(attrs["geometry"], "euclidean")
        self.assertEqual(attrs["constraint"], "fixed-density")
        self.assertEqual(attrs["interaction_model"], "occupancy")
        self.assertEqual(attrs["update_rule"], "swap")
        self.assertIn("anneal_data_json", attrs)
        self.assertIn("greedy_history_json", attrs)
        self.assertIn("parallel_tempering_history_json", attrs)
        self.assertNotIn("system_type", attrs)
        self.assertNotIn("antipodal", attrs)
        self.assertNotIn("constraint_model", attrs)
        self.assertNotIn("system_shape_json", attrs)


if __name__ == "__main__":
    unittest.main()
