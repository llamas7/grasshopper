from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np

from tests.helpers import cleanup_tmp_root, make_test_dir


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import euclidean_2d
import euclidean_3d
import gaussian_kernel
import parallel_tempering
import spherical_complementary
import spherical_independent
import spherical_fixed_density
from _common import paired_fibonacci_sphere


class ExampleSmokeTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        cleanup_tmp_root()

    def example_dir(self, name: str) -> Path:
        return make_test_dir(f"{self.id().replace('.', '_')}_{name}")

    def assert_finite_summary(self, system) -> None:
        for value in (
            system.current_energy,
            system.best_energy,
            system.current_probability,
            system.best_probability,
            system.current_graph_probability,
            system.best_graph_probability,
        ):
            self.assertTrue(np.isfinite(value))

    def run_example(self, module, arguments):
        with redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return module.main(arguments)

    def test_fibonacci_grid_is_unit_length_and_antipodally_ordered(self) -> None:
        grid = paired_fibonacci_sphere(12)

        np.testing.assert_allclose(np.linalg.norm(grid, axis=1), 1.0)
        np.testing.assert_allclose(grid[12:], -grid[:12])

    def test_spherical_examples(self) -> None:
        cases = (
            (
                spherical_complementary,
                "best_lawn.html",
                [6],
            ),
            (
                spherical_independent,
                "best_lawns.html",
                [6, 2],
            ),
            (
                spherical_fixed_density,
                "best_lawn.html",
                [12],
            ),
        )
        for module, filename, expected_shape in cases:
            with self.subTest(module=module.__name__):
                output_dir = self.example_dir(module.__name__)
                system = self.run_example(
                    module,
                    [
                        "--pairs",
                        "6",
                        "--steps",
                        "5",
                        "--anneal-iterations",
                        "1",
                        "--greedy-steps",
                        "2",
                        "--output-dir",
                        str(output_dir),
                    ]
                )
                self.assertEqual(list(system.current_spins.shape), expected_shape)
                self.assertTrue((output_dir / filename).exists())
                self.assert_finite_summary(system)
                if module is spherical_fixed_density:
                    self.assertEqual(
                        np.count_nonzero(system.current_spins),
                        len(system.grid) // 2,
                    )

    def test_euclidean_examples(self) -> None:
        output_2d = self.example_dir("euclidean_2d")
        system_2d = self.run_example(
            euclidean_2d,
            [
                "--width",
                "4",
                "--height",
                "4",
                "--steps",
                "5",
                "--anneal-iterations",
                "1",
                "--greedy-steps",
                "2",
                "--output-dir",
                str(output_2d),
            ]
        )
        self.assertTrue((output_2d / "best_lawn.png").exists())
        self.assertEqual(np.count_nonzero(system_2d.current_spins), system_2d.num_spins)
        self.assert_finite_summary(system_2d)

        output_3d = self.example_dir("euclidean_3d")
        system_3d = self.run_example(
            euclidean_3d,
            [
                "--side",
                "3",
                "--steps",
                "5",
                "--anneal-iterations",
                "1",
                "--greedy-steps",
                "2",
                "--output-dir",
                str(output_3d),
            ]
        )
        self.assertTrue((output_3d / "best_lawn.png").exists())
        self.assertEqual(np.count_nonzero(system_3d.current_spins), system_3d.num_spins)
        self.assert_finite_summary(system_3d)

    def test_gaussian_example(self) -> None:
        output_dir = self.example_dir("gaussian")
        compact, gaussian = self.run_example(
            gaussian_kernel,
            [
                "--width",
                "4",
                "--height",
                "4",
                "--steps",
                "5",
                "--anneal-iterations",
                "1",
                "--greedy-steps",
                "2",
                "--output-dir",
                str(output_dir),
            ]
        )
        self.assertTrue((output_dir / "compact_lawn.png").exists())
        self.assertTrue((output_dir / "gaussian_lawn.png").exists())
        self.assertEqual(gaussian.kernel.kind, "gaussian")
        self.assert_finite_summary(compact)
        self.assert_finite_summary(gaussian)

    def test_parallel_tempering_example(self) -> None:
        output_dir = self.example_dir("parallel_tempering")
        system = self.run_example(
            parallel_tempering,
            [
                "--width",
                "4",
                "--height",
                "4",
                "--replicas",
                "2",
                "--iterations",
                "1",
                "--sweep-steps",
                "5",
                "--output-dir",
                str(output_dir),
            ]
        )
        for filename in ("initial_lawn.png", "cold_final_lawn.png", "best_lawn.png"):
            self.assertTrue((output_dir / filename).exists())
        self.assertEqual(np.count_nonzero(system.current_spins), system.num_spins)
        self.assert_finite_summary(system)

    def test_seeded_example_is_deterministic(self) -> None:
        common_args = [
            "--width",
            "4",
            "--height",
            "4",
            "--steps",
            "5",
            "--anneal-iterations",
            "1",
            "--greedy-steps",
            "2",
            "--seed",
            "123",
        ]
        first = self.run_example(
            euclidean_2d,
            common_args + ["--output-dir", str(self.example_dir("first"))],
        )
        second = self.run_example(
            euclidean_2d,
            common_args + ["--output-dir", str(self.example_dir("second"))],
        )
        np.testing.assert_array_equal(first.best_spins, second.best_spins)
        self.assertEqual(first.best_energy, second.best_energy)


if __name__ == "__main__":
    unittest.main()
