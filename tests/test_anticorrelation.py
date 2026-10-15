from __future__ import annotations

import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import xarray as xr

import grasshopper as gh
from grasshopper.engines import get_engine
from grasshopper.spinflip_numba import compute_spinflip_numba_energy_grid
from tests.helpers import spherical_grid


class AnticorrelationTests(unittest.TestCase):
    def make_system(self, **kwargs):
        return gh.System.independent(spherical_grid()).compute_anticorrelation_neighborhood(
            kwargs.pop("angles", [0, .48*np.pi, .52*np.pi]), **kwargs
        )

    def test_component_sum_and_single_angle(self):
        for kernel in (gh.KernelSpec.cosine(), gh.KernelSpec.cell_averaged_hat(),
                       gh.KernelSpec.cell_averaged_cosine(), gh.KernelSpec.gaussian(.2)):
            for angles, q in (([0, .48*np.pi, .52*np.pi], [.2, .3, .5]),
                              ([.4, .4, 0], [0, .7, .3]), ([.4], [1]), ([0], [1])):
                with self.subTest(kernel=kernel, angles=angles):
                    s = self.make_system(angles=angles, probabilities=q, kernel=kernel).random_init(seed=3)
                    expected = []
                    for a in angles:
                        if a == 0:
                            expected.append(np.mean(s.current_spins[:, 0] != s.current_spins[:, 1]))
                        else:
                            raw = gh.System.independent(s.grid).compute_neighborhood(a, kernel=kernel)
                            expected.append(raw.probability(s.current_spins))
                    np.testing.assert_allclose(s.angle_probabilities(), expected, atol=1e-12)
                    self.assertAlmostEqual(s.current_probability, np.dot(q, expected))
                    self.assertAlmostEqual(s.current_energy, s.num_spins/2*np.dot(q, expected))
        s = self.make_system().random_init(seed=8)
        self.assertAlmostEqual(s.current_probability, np.mean(s.angle_probabilities()))

    def test_zero_exact_and_exhaustive_move_updates(self):
        s = self.make_system(probabilities=[.3, .2, .5])
        engine = get_engine(s.kind)
        for bits in itertools.product((False, True), repeat=6):
            state = np.array(bits).reshape(3, 2)
            s.array_init(state)
            energy = engine.energy(state, s.neighborhood)
            delta = engine.grid_gen(state, s.neighborhood)
            accelerated = compute_spinflip_numba_energy_grid(state, s.neighborhood)
            if accelerated is not None:
                np.testing.assert_allclose(accelerated[0], delta, atol=1e-12)
                self.assertAlmostEqual(accelerated[1], energy)
            for i, j in itertools.product(range(3), range(2)):
                trial = state.copy()
                trial[i, j] = ~trial[i, j]
                self.assertAlmostEqual(delta[i, j], engine.energy(trial, s.neighborhood)-energy)
                s.array_init(state.copy())
                updated = delta.copy()
                engine.update(s, (i, j), updated)
                np.testing.assert_array_equal(s.current_spins, trial)
                np.testing.assert_allclose(updated, engine.grid_gen(trial, s.neighborhood), atol=1e-12)
        z = self.make_system(angles=[0])
        for opposite, expected in ((False, 0), (True, 1)):
            state = np.zeros((3, 2), dtype=bool)
            state[:, 1] = opposite
            z.array_init(state)
            self.assertEqual(z.current_probability, expected)
            self.assertEqual(z.angle_probabilities()[0], expected)

    def test_optimizers_clone_and_rebuild(self):
        s = self.make_system().random_init(seed=9)
        engine = get_engine(s.kind)
        s.anneal(starting_temp=1, final_temp=.1, steps=100, max_iterations=3, seed=10)
        s.greedy(max_steps=30)
        s.parallel_temper(num_replicas=3, iterations=3, sweep_steps=30,
                          cold_temp=.1, hot_temp=1, n_workers=1, seed=11)
        self.assertAlmostEqual(s.current_energy, engine.energy(s.current_spins, s.neighborhood))
        self.assertAlmostEqual(s.best_energy, engine.energy(s.best_spins, s.neighborhood))
        c = s.clone()
        self.assertEqual(c.sampling_probabilities, s.sampling_probabilities)
        self.assertIs(c.neighborhood, s.neighborhood)
        c.compute_anticorrelation_neighborhood([0])
        self.assertAlmostEqual(c.current_probability, c.angle_probabilities()[0])
        self.assertEqual(len(s.sampling_probabilities), 3)
        c.compute_neighborhood(np.pi/2)
        self.assertIsNone(c.sampling_probabilities)
        with self.assertRaises(RuntimeError):
            c.angle_probabilities()

    def test_persistence_and_legacy(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            path = Path(td)/"state.nc"
            s = self.make_system().random_init(seed=15)
            s.save(path)
            loaded = gh.System.independent(s.grid)
            loaded.load(path)
            self.assertIsNone(loaded.neighborhood)
            self.assertEqual(loaded.sampling_probabilities, s.sampling_probabilities)
            self.assertAlmostEqual(loaded.best_probability, s.best_probability)
            np.testing.assert_allclose(loaded.angle_probabilities(), s.angle_probabilities())
            loaded.save(Path(td)/"again.nc")
            loaded.load(path, rebuild_neighborhood=True)
            self.assertAlmostEqual(loaded.current_probability, s.current_probability)
            np.testing.assert_allclose(loaded.angle_probabilities(loaded.best_spins), s.angle_probabilities(s.best_spins))
            raw = gh.System.independent(s.grid).compute_neighborhood(.4).random_init(seed=15)
            for version in ("4.0", "5.0"):
                ds = raw.to_dataset()
                ds.attrs["format_version"] = version
                ds.to_netcdf(path)
                loaded.load(path, rebuild_neighborhood=True)
                self.assertIsNone(loaded.sampling_probabilities)
                self.assertAlmostEqual(loaded.current_probability, raw.current_probability)

    def test_validation(self):
        for angles, q in (([], None), ([np.pi], None), ([-.1], None), ([np.nan], None),
                          ([0, .4], [1]), ([0, .4], [-1, 2]), ([0, .4], [.2, .2]),
                          ([0, .4], [np.nan, 1])):
            with self.assertRaises(ValueError):
                self.make_system(angles=angles, probabilities=q)
        for kernel in (gh.KernelSpec.exact(), lambda x, h: x):
            with self.assertRaises(ValueError):
                self.make_system(kernel=kernel)
        with self.assertRaises(ValueError):
            gh.System.complementary(spherical_grid()).compute_anticorrelation_neighborhood([0])

    def test_runner(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            output = Path(td)/"experiment"
            subprocess.run([sys.executable, "examples/spherical_multi_angle.py", "--angles-over-pi",
                            "0", str(1/9), "--probabilities", ".2", ".8", "--pairs", "16",
                            "--steps", "20", "--anneal-iterations", "1", "--greedy-steps", "5",
                            "--output-dir", str(output)], check=True, capture_output=True, text=True)
            summary = json.loads((output/"summary.json").read_text())
            self.assertEqual(len(summary["runs"]), 4)
            self.assertEqual(summary["runs"][0]["initialization"], "hemisphere")
            self.assertEqual(summary["best_probability"], max(r["best_probability"] for r in summary["runs"]))
            grid = np.load(output/"grid.npy")
            loaded = gh.System.independent(grid)
            loaded.load(output/summary["winning_checkpoint"], rebuild_neighborhood=True)
            self.assertAlmostEqual(loaded.best_probability, summary["best_probability"])
            self.assertAlmostEqual(loaded.best_probability,
                                   np.dot(summary["probabilities"], loaded.angle_probabilities(loaded.best_spins)))
