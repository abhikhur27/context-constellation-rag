import argparse
import unittest

import numpy as np

from main import (
    Chunk,
    EmbeddingEngine,
    build_expected_source_metrics,
    build_quality_gate,
    unit_interval,
)


def evidence_row(rank: int, source: str) -> dict:
    return {
        "rank": rank,
        "chunk": Chunk(
            chunk_id=f"{source}::c1",
            source=source,
            text="fixture",
            start=0,
            end=7,
        ),
    }


class ExpectedSourceMetricTests(unittest.TestCase):
    def test_metrics_capture_recall_and_rank(self) -> None:
        metrics = build_expected_source_metrics(
            expected_sources=["rollout", "memory", "missing"],
            evidence=[
                evidence_row(1, "rollout_posture.md"),
                evidence_row(2, "governance_notes.md"),
                evidence_row(4, "memory_signals.md"),
            ],
        )

        self.assertEqual(metrics["matched_count"], 2)
        self.assertEqual(metrics["recall_at_k"], 0.6667)
        self.assertEqual(metrics["mean_reciprocal_rank"], 0.4167)
        self.assertEqual([item["best_rank"] for item in metrics["details"]], [1, 4, None])

    def test_no_expectations_are_not_scored(self) -> None:
        metrics = build_expected_source_metrics(expected_sources=[], evidence=[])

        self.assertIsNone(metrics["recall_at_k"])
        self.assertIsNone(metrics["mean_reciprocal_rank"])

    def test_invalid_expected_source_pattern_has_clear_failure(self) -> None:
        with self.assertRaisesRegex(SystemExit, "Invalid expected source regex"):
            build_expected_source_metrics(expected_sources=["["], evidence=[])


class QualityGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = {
            "expected_source_recall": 0.9,
            "expected_source_mrr": 0.75,
            "variant_stability_rate": 0.8,
            "flagged_query_rate": 0.1,
        }

    def test_gate_passes_at_configured_thresholds(self) -> None:
        gate = build_quality_gate(
            summary=self.summary,
            min_expected_source_recall=0.9,
            min_expected_source_mrr=0.7,
            min_variant_stability_rate=0.8,
            max_flagged_query_rate=0.1,
        )

        self.assertTrue(gate["configured"])
        self.assertTrue(gate["passed"])
        self.assertTrue(all(check["passed"] for check in gate["checks"]))

    def test_gate_reports_each_failed_metric(self) -> None:
        gate = build_quality_gate(
            summary=self.summary,
            min_expected_source_recall=1.0,
            max_flagged_query_rate=0.0,
        )

        self.assertFalse(gate["passed"])
        self.assertEqual(
            [check["metric"] for check in gate["checks"] if not check["passed"]],
            ["expected_source_recall", "flagged_query_rate"],
        )


class OfflineEmbeddingTests(unittest.TestCase):
    def test_hashing_mode_is_deterministic_and_normalized(self) -> None:
        engine = EmbeddingEngine("hashing")
        first = engine.encode(["rollout risk", "memory signal"])
        second = engine.encode(["rollout risk", "memory signal"])

        self.assertEqual(engine.mode, "hashing")
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), np.ones(2), atol=1e-6)

    def test_unit_interval_rejects_out_of_range_threshold(self) -> None:
        self.assertEqual(unit_interval("0.75"), 0.75)
        with self.assertRaises(argparse.ArgumentTypeError):
            unit_interval("1.1")


if __name__ == "__main__":
    unittest.main()
