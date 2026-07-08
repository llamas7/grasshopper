from __future__ import annotations

import contextlib
import io
import unittest

from examples.collect_gaussian import (
    choose_adaptive_work,
    compatible_records,
    eta_grid,
    initial_eta_grid,
    parse_args,
    summarize_etas,
)


def make_args(*extra: str):
    return parse_args(
        [
            "transitions",
            "--jump-distance",
            "1.0",
            "--grid-length",
            "4",
            "--num-spins",
            "4",
            "--eta-min",
            "0.0",
            "--eta-max",
            "1.0",
            "--eta-step",
            "0.25",
            "--initial-eta-points",
            "3",
            "--max-eta-points",
            "8",
            "--runs",
            "2",
            "--max-runs-per-eta",
            "4",
            *extra,
        ]
    )


def record(
    eta: float,
    run_index: int,
    label: str,
    probability: float,
    *,
    jump_distance: float = 1.0,
    grid_length: int = 4,
    num_spins: int = 4,
) -> dict:
    return {
        "eta": eta,
        "run_index": run_index,
        "morphology_label": label,
        "probability": probability,
        "jump_distance": jump_distance,
        "grid_length": grid_length,
        "num_spins": num_spins,
    }


def completed_anchors(labels: tuple[str, str, str] = ("disk", "disk", "annulus")):
    rows = []
    for eta, label in zip((0.0, 0.5, 1.0), labels):
        rows.extend(
            [
                record(eta, 0, label, 1.0),
                record(eta, 1, label, 1.1),
            ]
        )
    return rows


class AdaptiveGaussianSchedulerTests(unittest.TestCase):
    def test_initial_anchors_are_completed_before_adaptation(self) -> None:
        args = make_args()
        first = choose_adaptive_work(args, [])
        self.assertEqual((first.eta, first.run_index, first.reason), (0.0, 0, "initial_anchor"))

        second = choose_adaptive_work(args, [record(0.0, 0, "disk", 1.0)])
        self.assertEqual((second.eta, second.run_index), (0.0, 1))

    def test_explicit_eta_is_always_an_initial_anchor(self) -> None:
        args = make_args("--eta", "0.3")
        self.assertEqual(initial_eta_grid(args), [0.0, 0.3, 0.5, 1.0])

    def test_best_probability_defines_eta_label(self) -> None:
        summary = summarize_etas(
            [
                record(0.5, 0, "disk", 2.0),
                record(0.5, 1, "annulus", 3.0),
            ]
        )[0.5]
        self.assertEqual(summary.best["morphology_label"], "annulus")
        self.assertTrue(summary.ambiguous)

    def test_ambiguous_eta_receives_an_extra_run_first(self) -> None:
        args = make_args()
        rows = completed_anchors()
        rows[1] = record(0.0, 1, "annulus", 1.1)
        decision = choose_adaptive_work(args, rows)
        self.assertEqual(
            (decision.eta, decision.run_index, decision.reason),
            (0.0, 2, "ambiguous_replicate"),
        )

    def test_completed_cached_anchors_resume_with_adaptive_work(self) -> None:
        decision = choose_adaptive_work(make_args(), completed_anchors())
        self.assertEqual(
            (decision.eta, decision.run_index, decision.reason),
            (0.75, 0, "boundary_refinement"),
        )

    def test_boundary_interval_is_bisected(self) -> None:
        decision = choose_adaptive_work(make_args(), completed_anchors())
        self.assertEqual(
            (decision.eta, decision.reason),
            (0.75, "boundary_refinement"),
        )

    def test_periodic_exploration_preempts_boundary_refinement(self) -> None:
        decision = choose_adaptive_work(
            make_args("--exploration-width", "0.4"),
            completed_anchors(),
            boundary_since_exploration=4,
        )
        self.assertEqual(
            (decision.eta, decision.reason),
            (0.25, "periodic_exploration"),
        )

    def test_ties_choose_the_lower_interval_deterministically(self) -> None:
        args = make_args("--exploration-width", "0.2")
        decision = choose_adaptive_work(
            args,
            completed_anchors(("disk", "disk", "disk")),
        )
        self.assertEqual((decision.eta, decision.reason), (0.25, "interval_exploration"))

    def test_scheduler_stops_when_intervals_are_resolved(self) -> None:
        args = make_args("--exploration-width", "0.5")
        decision = choose_adaptive_work(
            args,
            completed_anchors(("disk", "disk", "disk")),
        )
        self.assertIsNone(decision)

    def test_distinct_eta_budget_blocks_new_points(self) -> None:
        args = make_args("--max-eta-points", "3")
        self.assertIsNone(choose_adaptive_work(args, completed_anchors()))

    def test_compatible_records_exclude_other_scans(self) -> None:
        args = make_args()
        rows = [
            record(0.0, 0, "disk", 1.0),
            record(0.0, 1, "disk", 1.0, grid_length=5),
        ]
        self.assertEqual(compatible_records(rows, args), rows[:1])

    def test_uniform_eta_grid_retains_step_behavior(self) -> None:
        args = make_args("--uniform")
        self.assertEqual(eta_grid(args), [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_cli_rejects_invalid_adaptive_limits(self) -> None:
        invalid_cases = [
            ("--initial-eta-points", "1"),
            ("--max-eta-points", "2"),
            ("--max-runs-per-eta", "1"),
            ("--eta-tolerance", "0"),
        ]
        for option, value in invalid_cases:
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    make_args(option, value)


if __name__ == "__main__":
    unittest.main()
