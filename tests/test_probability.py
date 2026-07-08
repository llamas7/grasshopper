from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest
import warnings

import numpy as np
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

    def test_legacy_kernel_names_remain_compatible(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(gh.KernelSpec.two(), gh.KernelSpec.cosine())
            self.assertEqual(
                gh.KernelSpec.one_star(),
                gh.KernelSpec.cell_averaged_hat(),
            )
            self.assertEqual(
                gh.KernelSpec.two_star(),
                gh.KernelSpec.cell_averaged_cosine(),
            )
        self.assertEqual(len(caught), 3)
        self.assertTrue(all(item.category is DeprecationWarning for item in caught))
        self.assertEqual(gh.KernelSpec("two").kind, "cosine")
        self.assertEqual(
            gh.KernelSpec("one_star").kind,
            "cell_averaged_hat",
        )
        self.assertEqual(
            gh.KernelSpec("two_star").kind,
            "cell_averaged_cosine",
        )

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
        sphere = gh.System.inverse(spherical_grid())
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

    def test_probability_api_supports_current_best_and_arbitrary_states(self) -> None:
        systems_and_states = [
            (
                gh.System.inverse(spherical_grid()),
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
                gh.System.non_antipodal(spherical_grid()),
                np.array([True, False, True, False, True, False], dtype=bool),
            ),
            (
                gh.System.euclidean(euclidean_grid(), num_spins=2),
                np.array([True, False, True, False], dtype=bool),
            ),
        ]
        for system, state in systems_and_states:
            distance = np.pi / 2 if system.geometry == gh.Geometry.SPHERICAL else 1.0
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
        inverse = gh.System.inverse(spherical_grid())
        inverse.compute_neighborhood(np.pi / 2)
        for state in (
            np.array([True, False, True], dtype=bool),
            np.array([False, False, False], dtype=bool),
        ):
            self.assertAlmostEqual(
                inverse.probability(state, normalization="graph"),
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
        sparse = gh.System.non_antipodal(spherical_grid())
        full = gh.System.non_antipodal(spherical_grid())
        sparse.compute_neighborhood(
            np.pi / 2,
            kernel=gh.KernelSpec.gaussian(0.3, cutoff=5.0),
        )
        full.compute_neighborhood(
            np.pi / 2,
            kernel=gh.KernelSpec.gaussian(0.3, cutoff=None),
        )
        np.testing.assert_allclose(
            sparse.neighborhood.spin_potential,
            full.neighborhood.spin_potential,
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
        system = gh.System.non_antipodal(spherical_grid())
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
            gh.System.inverse(spherical_grid()).compute_neighborhood(np.pi + 0.1)

    def test_sharp_endpoints_reject_continuum_but_gaussian_is_defined(self) -> None:
        state = np.array([True, False, True], dtype=bool)
        sharp = gh.System.inverse(spherical_grid())
        sharp.compute_neighborhood(0.0)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            sharp.probability(state)

        gaussian = gh.System.inverse(spherical_grid())
        gaussian.compute_neighborhood(
            0.0,
            kernel=gh.KernelSpec.gaussian(0.2, cutoff=None),
        )
        self.assertTrue(np.isfinite(gaussian.probability(state)))

    def test_version_four_gaussian_roundtrip_and_version_three_migration(self) -> None:
        test_dir = make_test_dir(self.id().replace(".", "_"))
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(
            1.0,
            kernel=gh.KernelSpec.gaussian(0.2, cutoff=None),
        )
        system.array_init(np.array([True, False, True, False], dtype=bool))
        path = system.save(test_dir / "v4.nc")

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

        migrated = gh.System.euclidean(euclidean_grid(), num_spins=2)
        migrated.load(legacy_path)
        self.assertEqual(migrated.kernel, gh.KernelSpec.cosine())
        migrated.compute_neighborhood()
        self.assertEqual(migrated.kernel.kind, "cosine")

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

    def test_gpu_compact_kernels_match_cpu_when_cupy_is_available(self) -> None:
        import grasshopper.build_neighbors_gpu as gpu

        if gpu.cp is None:
            self.skipTest("CuPy is not installed.")
        errors = np.linspace(0.0, 2.5, 101)
        for cpu_kernel, gpu_kernel in (
            (cosine_kernel, gpu.cosine_kernel_gpu),
            (cell_averaged_hat_kernel, gpu.cell_averaged_hat_kernel_gpu),
            (
                cell_averaged_cosine_kernel,
                gpu.cell_averaged_cosine_kernel_gpu,
            ),
        ):
            cpu_weights, cpu_mask = cpu_kernel(errors, 1.0)
            gpu_weights, gpu_mask = gpu_kernel(gpu.cp.asarray(errors), 1.0)
            np.testing.assert_array_equal(gpu.cp.asnumpy(gpu_mask), cpu_mask)
            np.testing.assert_allclose(
                gpu.cp.asnumpy(gpu_weights)[cpu_mask],
                cpu_weights,
                atol=1e-12,
                rtol=1e-12,
            )

    def test_deprecated_delta_and_prefactor_aliases_warn(self) -> None:
        system = gh.System.euclidean(euclidean_grid(), num_spins=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            system.compute_neighborhood(1.0, delta="two")
            _ = system.delta
            prefactor = system.prob_prefactor
        self.assertEqual(len(caught), 3)
        self.assertTrue(all(item.category is DeprecationWarning for item in caught))
        self.assertAlmostEqual(
            prefactor,
            continuum_probability_prefactor(system),
        )

    def test_spherical_shell_measure_matches_geodesic_circumference(self) -> None:
        self.assertAlmostEqual(spherical_shell_measure(np.pi / 2), 2.0 * np.pi)
        self.assertEqual(spherical_shell_measure(-0.1), 0.0)


if __name__ == "__main__":
    unittest.main()
