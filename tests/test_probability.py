from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

import numpy as np
import xarray as xr
from scipy.integrate import quad

import grasshopper as gh
from grasshopper.kernels import (
    cell_averaged_cosine_kernel,
    cell_averaged_hat_kernel,
    cosine_kernel,
    gaussian_kernel,
)
from grasshopper.probability import (
    continuum_probability_prefactor,
    euclidean_shell_measure,
    spherical_shell_measure,
)

from tests.helpers import cleanup_tmp_root, euclidean_grid, make_test_dir, spherical_grid


class ProbabilityTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        cleanup_tmp_root()

    def test_kernel_spec_factories_are_frozen_and_validated(self) -> None:
        self.assertEqual(gh.KernelSpec.cosine().kind, "cosine")
        self.assertEqual(
            gh.KernelSpec.cell_averaged_hat().kind,
            "cell_averaged_hat",
        )
        self.assertEqual(
            gh.KernelSpec.cell_averaged_cosine().kind,
            "cell_averaged_cosine",
        )
        self.assertEqual(gh.KernelSpec.exact().kind, "exact")
        gaussian = gh.KernelSpec.gaussian(0.2)
        self.assertEqual(gaussian.sigma, 0.2)
        self.assertEqual(gaussian.cutoff, 5.0)
        with self.assertRaises(FrozenInstanceError):
            gaussian.sigma = 0.3
        with self.assertRaises(ValueError):
            gh.KernelSpec.gaussian(0.0)
        with self.assertRaises(ValueError):
            gh.KernelSpec.gaussian(0.2, cutoff=0.0)

    def test_legacy_kernel_names_are_removed(self) -> None:
        for legacy_name in (
            "two",
            "one_star",
            "one*",
            "two_star",
            "two*",
            "delta",
            "dirac",
            "exact_delta",
            "real_delta",
        ):
            with self.assertRaisesRegex(ValueError, "Unknown kernel"):
                gh.KernelSpec(legacy_name)

    def test_compact_kernels_have_unit_mass(self) -> None:
        for kernel, support in (
            (cosine_kernel, 2.0),
            (cell_averaged_hat_kernel, 1.5),
            (cell_averaged_cosine_kernel, 2.5),
        ):
            def integrand(value):
                weights, _ = kernel(np.array([abs(value)]), 1.0)
                return 0.0 if weights.size == 0 else float(weights[0])

            mass, _ = quad(integrand, -support, support, epsabs=1e-11)
            self.assertAlmostEqual(mass, 1.0, places=9)

    def test_gaussian_kernel_stores_h_times_normalized_density(self) -> None:
        sigma = 0.3
        cell_length = 0.2

        def integrand(value):
            weights, _ = gaussian_kernel(
                np.array([abs(value)]),
                cell_length,
                sigma=sigma,
                cutoff=None,
            )
            return float(weights[0])

        mass, _ = quad(integrand, -np.inf, np.inf, epsabs=1e-11)
        self.assertAlmostEqual(mass, cell_length, places=10)

    def test_continuum_prefactors_match_dissertation_formulas(self) -> None:
        sphere = gh.System.complementary(spherical_grid())
        theta = np.pi / 3.0
        sphere.compute_neighborhood(theta, kernel=gh.KernelSpec.cosine())
        expected_sphere = 4.0 / (
            np.sin(theta)
            * len(sphere.grid) ** 1.5
            * np.sqrt(np.pi)
        )
        self.assertAlmostEqual(
            continuum_probability_prefactor(sphere),
            expected_sphere,
        )

        euclidean = gh.System.euclidean(euclidean_grid(), num_spins=2)
        euclidean.compute_neighborhood(1.0, kernel=gh.KernelSpec.cosine())
        expected_euclidean = 1.0 / (np.pi * 2.0**1.5)
        self.assertAlmostEqual(
            continuum_probability_prefactor(euclidean),
            expected_euclidean,
        )

    def test_multi_shell_continuum_prefactor_uses_weighted_radial_mass(self) -> None:
        distances = (np.pi / 3.0, np.pi / 2.0)
        weights = (1.0, 0.5)
        sphere = gh.System.complementary(spherical_grid())
        sphere.compute_neighborhood(
            distances,
            jump_weights=weights,
            kernel=gh.KernelSpec.cosine(),
        )

        cell_measure = 4.0 * np.pi / len(sphere.grid)
        starting_measure = sphere.num_spins * cell_measure
        cell_length = np.sqrt(cell_measure)
        radial_mass = sum(
            weight * spherical_shell_measure(distance)
            for distance, weight in zip(distances, weights)
        )
        expected = 2.0 * cell_measure**2 / (
            starting_measure * cell_length * radial_mass
        )

        self.assertAlmostEqual(continuum_probability_prefactor(sphere), expected)

    def test_probability_api_supports_current_best_and_arbitrary_states(self) -> None:
        systems_and_states = [
            (
                gh.System.complementary(spherical_grid()),
                np.array([True, False, True], dtype=bool),
            ),
            (
                gh.System.independent(spherical_grid()),
                np.array(
                    [[True, False], [False, True], [True, True]],
                    dtype=bool,
                ),
            ),
            (
                gh.System.fixed_density(spherical_grid()),
                np.array([True, False, True, False, True, False], dtype=bool),
            ),
            (
                gh.System.euclidean(euclidean_grid(), num_spins=2),
                np.array([True, False, True, False], dtype=bool),
            ),
        ]
        for system, state in systems_and_states:
            distance = np.pi / 2 if system.kind.spherical else 1.0
            system.compute_neighborhood(distance)
            system.array_init(state)
            self.assertAlmostEqual(
                system.current_probability,
                system.current_energy
                * continuum_probability_prefactor(system, state),
            )
            self.assertAlmostEqual(
                system.best_probability,
                system.probability(state),
            )
            self.assertTrue(np.isfinite(system.current_graph_probability))
            self.assertTrue(np.isfinite(system.best_graph_probability))

    def test_antipodal_graph_probability_is_half_at_right_angle(self) -> None:
        complementary = gh.System.complementary(spherical_grid())
        complementary.compute_neighborhood(np.pi / 2)
        for state in (
            np.array([True, False, True], dtype=bool),
            np.array([False, False, False], dtype=bool),
        ):
            self.assertAlmostEqual(
                complementary.probability(state, normalization="graph"),
                0.5,
            )

        independent = gh.System.independent(spherical_grid())
        independent.compute_neighborhood(np.pi / 2)
        state = np.array(
            [[True, False], [False, False], [True, True]],
            dtype=bool,
        )
        self.assertAlmostEqual(
            independent.probability(state, normalization="graph"),
            0.5,
        )

    def test_gaussian_normalization_and_delta_limit(self) -> None:
        distance = 1.0
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        sharp = gh.System.euclidean(euclidean_grid(), num_spins=2)
        sharp.compute_neighborhood(distance, kernel=gh.KernelSpec.cosine())
        sharp_prefactor = continuum_probability_prefactor(sharp)

        differences = []
        for sigma in (0.2, 0.1, 0.05):
            system.compute_neighborhood(
                distance,
                kernel=gh.KernelSpec.gaussian(sigma, cutoff=None),
            )
            differences.append(
                abs(continuum_probability_prefactor(system) - sharp_prefactor)
            )
        self.assertLess(differences[-1], differences[0])

        sigma = 0.2
        truncated = gh.System.euclidean(euclidean_grid(), num_spins=2)
        truncated.compute_neighborhood(
            distance,
            kernel=gh.KernelSpec.gaussian(sigma, cutoff=5.0),
        )
        lower = max(0.0, distance - 5.0 * sigma)
        upper = distance + 5.0 * sigma
        radial, _ = quad(
            lambda radius: euclidean_shell_measure(radius, 2)
            * np.exp(-0.5 * ((radius - distance) / sigma) ** 2)
            / (np.sqrt(2.0 * np.pi) * sigma),
            lower,
            upper,
        )
        expected = 2.0 * (0.5**2) / (1.0 * np.sqrt(0.5) * radial)
        self.assertAlmostEqual(
            continuum_probability_prefactor(truncated),
            expected,
        )

    def test_sparse_and_all_pairs_gaussians_agree_up_to_truncated_tail(self) -> None:
        sparse = gh.System.fixed_density(spherical_grid())
        full = gh.System.fixed_density(spherical_grid())
        sparse.compute_neighborhood(
            np.pi / 2,
            kernel=gh.KernelSpec.gaussian(0.3, cutoff=5.0),
        )
        full.compute_neighborhood(
            np.pi / 2,
            kernel=gh.KernelSpec.gaussian(0.3, cutoff=None),
        )
        np.testing.assert_allclose(
            sparse.neighborhood.total_weight,
            full.neighborhood.total_weight,
            atol=3e-6,
            rtol=0,
        )
        self.assertLessEqual(
            sparse.neighborhood.neighbor_spins.size,
            full.neighborhood.neighbor_spins.size,
        )

    def test_custom_metric_requires_shell_measure_for_continuum_probability(self) -> None:
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(
            1.0,
            distance_metric="manhattan",
        )
        state = np.array([True, False, True, False], dtype=bool)
        with self.assertRaisesRegex(ValueError, "shell_measure"):
            system.probability(state)
        self.assertTrue(np.isfinite(system.probability(state, normalization="graph")))

        system.compute_neighborhood(
            1.0,
            distance_metric="manhattan",
            shell_measure=lambda radius: 8.0 * radius,
            radial_bounds=(0.0, np.inf),
        )
        self.assertTrue(np.isfinite(system.probability(state)))

    def test_exact_and_callable_kernels_only_support_graph_probability(self) -> None:
        exact = gh.System.euclidean(euclidean_grid(), num_spins=2)
        exact.compute_neighborhood(1.0, kernel=gh.KernelSpec.exact())
        state = np.array([True, False, True, False], dtype=bool)
        with self.assertRaisesRegex(ValueError, "exact diagnostic"):
            exact.probability(state)

        def signed_kernel(errors, cell_length):
            mask = errors <= cell_length
            return -np.ones(np.count_nonzero(mask)), mask

        custom = gh.System.euclidean(euclidean_grid(), num_spins=2)
        custom.compute_neighborhood(1.0, kernel=signed_kernel)
        with self.assertRaisesRegex(ValueError, "callable kernels"):
            custom.probability(state)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            custom.probability(state, normalization="graph")

    def test_empty_lawn_zero_graph_weight_and_invalid_distances(self) -> None:
        system = gh.System.fixed_density(spherical_grid())
        system.compute_neighborhood(np.pi / 3)
        with self.assertRaisesRegex(ZeroDivisionError, "zero occupied"):
            system.probability(np.zeros(6, dtype=bool))

        exact = gh.System.euclidean(euclidean_grid(), num_spins=2)
        exact.compute_neighborhood(0.123, kernel=gh.KernelSpec.exact())
        with self.assertRaisesRegex(ZeroDivisionError, "zero available"):
            exact.probability(
                np.array([True, False, True, False], dtype=bool),
                normalization="graph",
            )

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            gh.System.euclidean(euclidean_grid(), num_spins=2).compute_neighborhood(
                -0.1
            )
        with self.assertRaisesRegex(ValueError, r"\[0, pi\]"):
            gh.System.complementary(spherical_grid()).compute_neighborhood(np.pi + 0.1)

    def test_sharp_endpoints_reject_continuum_but_gaussian_is_defined(self) -> None:
        state = np.array([True, False, True], dtype=bool)
        sharp = gh.System.complementary(spherical_grid())
        sharp.compute_neighborhood(0.0)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            sharp.probability(state)

        gaussian = gh.System.complementary(spherical_grid())
        gaussian.compute_neighborhood(
            0.0,
            kernel=gh.KernelSpec.gaussian(0.2, cutoff=None),
        )
        self.assertTrue(np.isfinite(gaussian.probability(state)))

    def test_version_five_gaussian_roundtrip_and_version_three_rejection(self) -> None:
        test_dir = make_test_dir(self.id().replace(".", "_"))
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(
            1.0,
            kernel=gh.KernelSpec.gaussian(0.2, cutoff=None),
        )
        system.array_init(np.array([True, False, True, False], dtype=bool))
        path = system.save(test_dir / "v5.nc")

        loaded = gh.System.euclidean(euclidean_grid(), num_spins=2)
        loaded.load(path)
        self.assertEqual(loaded.kernel, gh.KernelSpec.gaussian(0.2, cutoff=None))
        loaded.compute_neighborhood()
        self.assertAlmostEqual(loaded.current_energy, system.current_energy)

        legacy = system.to_dataset()
        legacy.attrs["format_version"] = "3.0"
        legacy.attrs["delta"] = "two"
        legacy.attrs["delta_custom"] = False
        for key in list(legacy.attrs):
            if key.startswith("kernel_") or key.startswith("gaussian_"):
                del legacy.attrs[key]
        legacy_path = test_dir / "v3.nc"
        legacy.to_netcdf(legacy_path)

        rejected = gh.System.euclidean(euclidean_grid(), num_spins=2)
        with self.assertRaisesRegex(ValueError, "Unsupported saved-state"):
            rejected.load(legacy_path)

    def test_version_four_inverse_metadata_migrates_to_complementary(self) -> None:
        test_dir = make_test_dir(self.id().replace(".", "_"))
        system = gh.System.complementary(spherical_grid())
        system.compute_neighborhood(np.pi / 2)
        system.array_init(np.array([True, False, True], dtype=bool))

        legacy = system.to_dataset()
        del legacy.attrs["kind"]
        legacy.attrs.update(
            format_version="4.0",
            geometry="spherical",
            constraint="antipodal",
            interaction_model="inverse",
            update_rule="flip",
        )
        legacy_path = test_dir / "v4_complementary.nc"
        legacy.to_netcdf(legacy_path)

        loaded = gh.System.complementary(spherical_grid())
        loaded.load(legacy_path, rebuild_neighborhood=True)

        self.assertIs(loaded.kind, gh.SystemKind.COMPLEMENTARY)
        self.assertAlmostEqual(loaded.current_energy, system.current_energy)

    def test_probabilities_survive_save_load_and_resave_without_neighborhood(self) -> None:
        test_dir = make_test_dir(self.id().replace(".", "_"))
        constructors = (
            lambda: gh.System.complementary(spherical_grid()),
            lambda: gh.System.independent(spherical_grid()),
            lambda: gh.System.fixed_density(spherical_grid()),
            lambda: gh.System.euclidean(euclidean_grid(), num_spins=2),
        )
        keys = ("current_probability", "best_probability",
                "current_graph_probability", "best_graph_probability")
        for index, construct in enumerate(constructors):
            with self.subTest(family=index):
                system = construct()
                system.compute_neighborhood(1.0, silent=True)
                system.random_init(seed=123)
                system.greedy(reinit=False)
                expected = {key: getattr(system, key) for key in keys}
                path = system.save(test_dir / f"family_{index}.nc")
                with xr.open_dataset(path) as ds:
                    for key, value in expected.items():
                        self.assertAlmostEqual(ds.attrs[key], value)

                loaded = construct()
                loaded.load(path)
                self.assertIsNone(loaded.neighborhood)
                for key, value in expected.items():
                    self.assertAlmostEqual(getattr(loaded, key), value)
                self.assertAlmostEqual(loaded.probability(), expected["current_probability"])
                self.assertAlmostEqual(loaded.probability(normalization="graph"),
                                       expected["current_graph_probability"])

                resaved = loaded.save(test_dir / f"resaved_{index}.nc")
                with xr.open_dataset(resaved) as ds:
                    for key, value in expected.items():
                        self.assertAlmostEqual(ds.attrs[key], value)
                clone = loaded.clone()
                clone.current_energy = 0.0
                with self.assertRaisesRegex(RuntimeError, "Neighborhood"):
                    _ = clone.current_graph_probability
                self.assertAlmostEqual(loaded.current_graph_probability,
                                       expected["current_graph_probability"])

                loaded.compute_neighborhood(silent=True)
                for key, value in expected.items():
                    self.assertAlmostEqual(getattr(loaded, key), value)

    def test_save_records_why_continuum_probability_is_unavailable(self) -> None:
        test_dir = make_test_dir(self.id().replace(".", "_"))
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(1.0, kernel=gh.KernelSpec.exact(), silent=True)
        system.array_init(np.array([True, False, True, False], dtype=bool))
        expected = system.current_graph_probability
        path = system.save(test_dir / "exact.nc")
        with xr.open_dataset(path) as ds:
            self.assertNotIn("current_probability", ds.attrs)
            self.assertIn("exact diagnostic", ds.attrs["current_probability_unavailable"])
            self.assertAlmostEqual(ds.attrs["current_graph_probability"], expected)
        loaded = gh.System.euclidean(euclidean_grid(), num_spins=2)
        loaded.load(path)
        self.assertAlmostEqual(loaded.current_graph_probability, expected)

    def test_load_rejects_different_particle_count_and_malformed_spins(self) -> None:
        test_dir = make_test_dir(self.id().replace(".", "_"))
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(1.0, silent=True)
        system.random_init(seed=123)
        path = system.save(test_dir / "state.nc")
        different_density = gh.System.euclidean(euclidean_grid(), num_spins=1)
        with self.assertRaisesRegex(ValueError, "expects num_spins=2"):
            different_density.load(path)

        malformed = system.to_dataset()
        malformed["current_system"] = (("wrong_site",), [True, False, True])
        malformed.to_netcdf(test_dir / "malformed.nc")
        with self.assertRaisesRegex(ValueError, "current_system has shape"):
            system.load(test_dir / "malformed.nc")

    def test_older_checkpoint_without_probabilities_can_be_rebuilt(self) -> None:
        test_dir = make_test_dir(self.id().replace(".", "_"))
        system = gh.System.complementary(spherical_grid())
        system.compute_neighborhood(1.0, silent=True)
        system.random_init(seed=123)
        ds = system.to_dataset()
        for key in list(ds.attrs):
            if "probability" in key:
                del ds.attrs[key]
        path = test_dir / "legacy.nc"
        ds.to_netcdf(path)
        loaded = gh.System.complementary(spherical_grid())
        loaded.load(path, rebuild_neighborhood=True)
        self.assertAlmostEqual(loaded.current_probability, system.current_probability)
        self.assertAlmostEqual(loaded.current_graph_probability, system.current_graph_probability)

    def test_custom_shell_measure_must_be_resupplied_after_load(self) -> None:
        test_dir = make_test_dir(self.id().replace(".", "_"))
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(
            1.0,
            distance_metric="manhattan",
            shell_measure=lambda radius: 8.0 * radius,
            radial_bounds=(0.0, 4.0),
        )
        system.array_init(np.array([True, False, True, False], dtype=bool))
        path = system.save(test_dir / "custom_shell.nc")

        loaded = gh.System.euclidean(euclidean_grid(), num_spins=2)
        loaded.load(path)
        with self.assertRaisesRegex(ValueError, "custom shell_measure"):
            loaded.compute_neighborhood()
        loaded.compute_neighborhood(shell_measure=lambda radius: 8.0 * radius)
        self.assertEqual(loaded.radial_bounds, (0.0, 4.0))
        self.assertTrue(np.isfinite(loaded.current_probability))

    def test_spherical_shell_measure_matches_geodesic_circumference(self) -> None:
        self.assertAlmostEqual(spherical_shell_measure(np.pi / 2), 2.0 * np.pi)
        self.assertEqual(spherical_shell_measure(-0.1), 0.0)


if __name__ == "__main__":
    unittest.main()
