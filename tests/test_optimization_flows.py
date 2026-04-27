from __future__ import annotations

from unittest import mock
import unittest

import numpy as np

import refactoredpc as rf
import refactoredpc.annealing as annealing_module
from refactoredpc.engines import get_engine

from tests.helpers import (
    assert_fixed_density_consistent,
    cleanup_tmp_root,
    euclidean_grid,
    exact_energy,
    make_test_dir,
    spherical_grid,
)


class OptimizationFlowTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        cleanup_tmp_root()

    def setUp(self) -> None:
        np.random.seed(12345)

    def make_inverse_system(self) -> rf.System:
        system = rf.System.inverse(spherical_grid())
        system.compute_neighborhood(jump_distance=np.pi / 2, silent=True)
        system.array_init(np.array([True, False, True], dtype=bool))
        return system

    def make_independent_system(self) -> rf.System:
        system = rf.System.independent(spherical_grid())
        system.compute_neighborhood(jump_distance=np.pi / 2, silent=True)
        system.array_init(
            np.array(
                [
                    [True, False],
                    [False, True],
                    [True, True],
                ],
                dtype=bool,
            )
        )
        return system

    def make_non_antipodal_system(self) -> rf.System:
        system = rf.System.non_antipodal(spherical_grid())
        system.compute_neighborhood(jump_distance=np.pi / 2, silent=True)
        system.array_init(
            np.array([True, False, True, False, True, False], dtype=bool)
        )
        return system

    def make_euclidean_system(self) -> rf.System:
        system = rf.System.euclidean(euclidean_grid(), num_spins=2)
        system.compute_neighborhood(jump_distance=1.0, silent=True)
        system.array_init(np.array([True, False, True, False], dtype=bool))
        return system

    def test_thermalize_updates_runtime_temperature(self) -> None:
        system = self.make_inverse_system()
        system.temperature = 1.0

        with mock.patch("refactoredpc.annealing.metropolis_sweep", return_value=0.0):
            final_temperature = system.thermalize(
                target_acceptance=0.5,
                tolerance=0.0,
                steps_per_temp=1,
                max_adjustments=1,
            )

        self.assertNotEqual(final_temperature, 1.0)
        self.assertEqual(system.temperature, final_temperature)

    def test_metropolis_sweep_uses_spinflip_numba_hook_for_inverse(self) -> None:
        system = self.make_inverse_system()
        engine = get_engine(system.config)
        energy_array = engine.grid_gen(system.current_system, system.neighborhood)

        with mock.patch(
            "refactoredpc.annealing.run_fixed_density_numba_sweep",
            return_value=None,
        ), mock.patch(
            "refactoredpc.annealing.run_spinflip_numba_sweep",
            return_value=0.25,
        ) as accelerated:
            acceptance = annealing_module.metropolis_sweep(
                system=system,
                energy_array=energy_array,
                T=1.0,
                n_steps=4,
                engine=engine,
                track_best=False,
            )

        self.assertEqual(acceptance, 0.25)
        accelerated.assert_called_once()

    def test_anneal_appends_record_and_keeps_exact_energy(self) -> None:
        system = self.make_inverse_system()

        system.anneal(
            starting_temp=1.0,
            final_temp=0.5,
            cooling_rate=0.5,
            max_iterations=1,
            steps=3,
            reinit=True,
        )

        self.assertEqual(system.anneal_data.iterations, 1)
        self.assertEqual(len(system.anneal_data.records), 1)
        self.assertAlmostEqual(system.current_energy, exact_energy(system))
        self.assertEqual(system.temperature, 0.5)

    def test_greedy_runs_for_all_supported_engines(self) -> None:
        systems = [
            self.make_inverse_system(),
            self.make_independent_system(),
            self.make_non_antipodal_system(),
            self.make_euclidean_system(),
        ]

        for system in systems:
            result = system.greedy(max_steps=3, reinit=True)
            self.assertIsInstance(result, rf.GreedyResult)
            self.assertEqual(system.greedy_history.runs, 1)
            self.assertEqual(len(system.greedy_history.records), 1)
            self.assertEqual(system.greedy_history.records[0].steps, result.steps)
            self.assertAlmostEqual(system.current_energy, exact_energy(system))
            if not system.antipodal:
                assert_fixed_density_consistent(self, system)

    def test_parallel_temper_serial_keeps_exact_energy_for_euclidean(self) -> None:
        system = self.make_euclidean_system()

        result = system.parallel_temper(
            num_replicas=2,
            iterations=2,
            sweep_steps=2,
            exchange_interval=1,
            n_workers=1,
            reinit=True,
            seed=123,
        )

        self.assertIsInstance(result, rf.ParallelTemperingResult)
        self.assertEqual(result.iterations, 2)
        self.assertEqual(len(result.final_energies), 2)
        self.assertEqual(system.anneal_data.iterations, 0)
        self.assertEqual(system.parallel_tempering_history.runs, 1)
        self.assertEqual(len(system.parallel_tempering_history.records), 1)
        self.assertEqual(
            system.parallel_tempering_history.records[0].iterations,
            result.iterations,
        )
        self.assertEqual(
            len(system.parallel_tempering_history.records[0].cold_energy_history),
            result.iterations,
        )
        self.assertAlmostEqual(system.current_energy, exact_energy(system))
        assert_fixed_density_consistent(self, system)

    def test_save_load_roundtrip_restores_runtime_state_for_euclidean(self) -> None:
        test_dir = make_test_dir(self.id().replace(".", "_"))
        system = self.make_euclidean_system()
        system.temperature = 3.5
        system.greedy(max_steps=1, reinit=True)
        system.parallel_temper(
            num_replicas=2,
            iterations=1,
            sweep_steps=1,
            exchange_interval=1,
            n_workers=1,
            reinit=True,
            seed=321,
        )
        save_path = system.save(test_dir)

        loaded = rf.System.euclidean(euclidean_grid(), num_spins=2)
        loaded.load(save_path)

        self.assertTrue(np.array_equal(system.current_system, loaded.current_system))
        self.assertTrue(np.array_equal(system.best_system, loaded.best_system))
        self.assertEqual(system.current_energy, loaded.current_energy)
        self.assertEqual(system.best_energy, loaded.best_energy)
        self.assertEqual(loaded.temperature, 3.5)
        self.assertIsNone(loaded.neighborhood)
        self.assertEqual(loaded.greedy_history.runs, 1)
        self.assertEqual(len(loaded.greedy_history.records), 1)
        self.assertEqual(loaded.parallel_tempering_history.runs, 1)
        self.assertEqual(len(loaded.parallel_tempering_history.records), 1)
        assert_fixed_density_consistent(self, loaded)


if __name__ == "__main__":
    unittest.main()
