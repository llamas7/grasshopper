from __future__ import annotations

import math
import unittest

import numpy as np

from examples.collect_gaussian_figure_data import (
    NUMERICAL_WIDTH_FACTOR,
    Regime,
    annulus_state,
    build_cases,
    disk_state,
    lattice_shape,
    morphology_diagnostics,
)


class GaussianFigureDataTests(unittest.TestCase):
    def setUp(self) -> None:
        coordinates = np.linspace(-1.0, 1.0, 41)
        xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
        self.grid = np.column_stack((xx.ravel(), yy.ravel()))

    def test_case_matrix_contains_gaussian_and_resolution_scans(self) -> None:
        cases = build_cases(
            scan="all",
            regimes=[Regime("test", 0.4)],
            etas=[0.0, 0.1],
            occupied_count=100,
            resolution_counts=[64, 100],
            runs=2,
        )
        self.assertEqual(len(cases), 8)
        self.assertEqual(
            {case.scan for case in cases},
            {"gaussian", "resolution"},
        )
        gaussian = [case for case in cases if case.scan == "gaussian"]
        self.assertAlmostEqual(gaussian[-1].sigma, 0.04)

    def test_lattice_shape_respects_requested_candidate_factor(self) -> None:
        shape = lattice_shape(100, 4.0, "square")
        self.assertGreaterEqual(shape.candidate_count, 400)
        self.assertGreater(shape.candidate_count, 100)
        honeycomb = lattice_shape(100, 4.0, "honeycomb")
        self.assertEqual(honeycomb.candidate_count % 2, 0)

    def test_disk_is_nearly_isotropic_and_annulus_has_inner_radius(self) -> None:
        occupied_count = 317
        disk = disk_state(self.grid, occupied_count)
        disk_metrics, _ = morphology_diagnostics(
            self.grid,
            disk,
            jump_distance=0.4,
            cell_length=0.05,
            m_max=12,
            radial_bin_width=0.05,
            radial_smoothing_width=0.10,
            inner_density_threshold=0.5,
            inner_sustain_bins=2,
        )
        self.assertLess(disk_metrics["anisotropy_q"], 0.05)
        self.assertEqual(disk_metrics["inner_radius"], 0.0)

        annulus = annulus_state(self.grid, occupied_count, target_radius=0.65)
        annulus_metrics, _ = morphology_diagnostics(
            self.grid,
            annulus,
            jump_distance=0.4,
            cell_length=0.05,
            m_max=12,
            radial_bin_width=0.05,
            radial_smoothing_width=0.10,
            inner_density_threshold=0.5,
            inner_sustain_bins=2,
        )
        self.assertIsNotNone(annulus_metrics["inner_radius"])
        self.assertGreater(annulus_metrics["inner_radius"], 0.2)

    def test_numerical_width_factor_matches_dissertation_formula(self) -> None:
        expected = math.sqrt(4.0 / 3.0 - 8.0 / math.pi**2)
        self.assertAlmostEqual(NUMERICAL_WIDTH_FACTOR, expected)
        self.assertAlmostEqual(expected, 0.723, places=3)


if __name__ == "__main__":
    unittest.main()
