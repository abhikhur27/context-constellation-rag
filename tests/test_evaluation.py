import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from main import (
    Chunk,
    EmbeddingEngine,
    build_conflict_source_metrics,
    build_expected_source_metrics,
    build_forbidden_source_metrics,
    build_quality_gate,
    build_retrieval_text,
    build_stale_source_penalties,
    mmr_select,
    read_corpus,
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


class RetrievalContractTests(unittest.TestCase):
    def test_corpus_source_ids_are_portable_posix_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            nested = Path(temp_dir) / "operations"
            nested.mkdir()
            (nested / "decision.md").write_text("Current decision.", encoding="utf-8")

            docs = read_corpus(Path(temp_dir))

        self.assertEqual(docs[0][0], "operations/decision.md")

    def test_retrieval_text_includes_searchable_source_metadata(self) -> None:
        text = build_retrieval_text(
            Chunk(
                chunk_id="support/eu_checkout_escalations.md::c1",
                source="support/eu_checkout_escalations.md",
                text="Customer reports increased.",
                start=0,
                end=27,
            )
        )

        self.assertIn("support eu checkout escalations", text)
        self.assertNotIn("escalations md", text)
        self.assertIn("Customer reports increased.", text)

    def test_current_intent_penalizes_archived_evidence_only(self) -> None:
        chunks = [
            Chunk("current::c1", "operations/current.md", "Active decision.", 0, 16),
            Chunk("old::c1", "archive/old.md", "Superseded launch decision.", 0, 28),
        ]

        self.assertEqual(
            build_stale_source_penalties("What is the current launch decision?", chunks),
            {1: 0.45},
        )
        self.assertEqual(build_stale_source_penalties("Compare both launch decisions", chunks), {})

    def test_forbidden_source_metrics_capture_distractor_rank(self) -> None:
        metrics = build_forbidden_source_metrics(
            forbidden_sources=["archive", "unrelated"],
            evidence=[
                evidence_row(1, "operations/current_decision.md"),
                evidence_row(3, "archive/checkout_launch_draft.md"),
            ],
        )

        self.assertEqual(metrics["hit_count"], 1)
        self.assertEqual(metrics["hit_rate"], 0.5)
        self.assertEqual(metrics["details"][0]["best_rank"], 3)

    def test_conflict_group_requires_distinct_sources_for_each_side(self) -> None:
        covered = build_conflict_source_metrics(
            conflict_source_groups=[["engineering", "support"]],
            evidence=[
                evidence_row(1, "engineering/load_test.md"),
                evidence_row(2, "support/customer_escalations.md"),
            ],
        )
        missing = build_conflict_source_metrics(
            conflict_source_groups=[["engineering", "support"]],
            evidence=[evidence_row(1, "engineering/load_test.md")],
        )

        self.assertEqual(covered["recall"], 1.0)
        self.assertTrue(covered["details"][0]["covered"])
        self.assertEqual(missing["recall"], 0.0)

    def test_mmr_uses_hybrid_relevance_when_provided(self) -> None:
        embeddings = np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32)
        selected = mmr_select(
            [0, 1],
            query_vec=np.asarray([1.0, 0.0], dtype=np.float32),
            doc_embeddings=embeddings,
            top_k=1,
            relevance_scores={0: 0.2, 1: 0.9},
        )

        self.assertEqual(selected, [1])


class QualityGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = {
            "expected_source_recall": 0.9,
            "expected_source_mrr": 0.75,
            "variant_stability_rate": 0.8,
            "flagged_query_rate": 0.1,
            "forbidden_source_hit_rate": 0.0,
            "conflict_source_recall": 1.0,
        }

    def test_gate_passes_at_configured_thresholds(self) -> None:
        gate = build_quality_gate(
            summary=self.summary,
            min_expected_source_recall=0.9,
            min_expected_source_mrr=0.7,
            min_variant_stability_rate=0.8,
            max_flagged_query_rate=0.1,
            max_forbidden_source_hit_rate=0.0,
            min_conflict_source_recall=1.0,
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

    def test_gate_enforces_distractor_and_conflict_contracts(self) -> None:
        summary = {
            **self.summary,
            "forbidden_source_hit_rate": 0.25,
            "conflict_source_recall": 0.5,
        }

        gate = build_quality_gate(
            summary=summary,
            max_forbidden_source_hit_rate=0.2,
            min_conflict_source_recall=0.75,
        )

        self.assertFalse(gate["passed"])
        self.assertEqual(
            [check["metric"] for check in gate["checks"] if not check["passed"]],
            ["forbidden_source_hit_rate", "conflict_source_recall"],
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
