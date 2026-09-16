import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from main import (
    Chunk,
    EmbeddingEngine,
    build_anchor_support_scores,
    build_answerability_metrics,
    build_conflict_source_metrics,
    build_expected_source_metrics,
    build_forbidden_source_metrics,
    build_grounding_decision,
    build_non_evidence_penalties,
    build_quality_gate,
    build_retrieval_text,
    build_scope_mismatch_penalties,
    build_source_scope_text,
    build_stale_source_penalties,
    expand_query_for_retrieval,
    mmr_select,
    read_corpus,
    load_query_suite,
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
    def test_anchor_support_uses_distinct_query_ranked_sources(self) -> None:
        chunks = [
            Chunk("decision::c1", "operations/decision.md", "cutover hold recovery lag", 0, 25),
            Chunk("decision::c2", "operations/decision.md", "cutover decision", 26, 42),
            Chunk("recovery::c1", "recovery/drill.md", "recovery restore validation", 0, 27),
            Chunk("budget::c1", "finance/budget.md", "budget allocation spending", 0, 26),
        ]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])

        scores, anchors = build_anchor_support_scores(
            lexical_matrix=matrix,
            ranked_candidates=[0, 1, 2, 3],
            chunks=chunks,
        )

        self.assertEqual(anchors, [0, 2])
        self.assertEqual(scores[0], 1.0)
        self.assertEqual(scores[2], 1.0)
        self.assertGreater(scores[1], scores[3])

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
            {1: 1.25},
        )
        self.assertEqual(build_stale_source_penalties("Compare both launch decisions", chunks), {})
        self.assertEqual(build_stale_source_penalties("Why did the launch fail?", chunks), {1: 0.30})

    def test_source_scope_uses_path_and_heading_without_body_noise(self) -> None:
        chunk = Chunk(
            "search::c1",
            "search/search_latency_review.md",
            "# Search Latency Review — 2026-06-18 This body mentions checkout invoices.",
            0,
            78,
        )

        self.assertEqual(
            build_source_scope_text(chunk),
            "search search latency review. Search Latency Review",
        )

        ascii_heading = Chunk(
            "recovery::c1",
            "recovery/restore_drill.md",
            "# Restore Drill Record - 2026-08-12 This body mentions unrelated invoices.",
            0,
            77,
        )
        self.assertEqual(
            build_source_scope_text(ascii_heading),
            "recovery restore drill. Restore Drill Record",
        )

    def test_scope_disclaimer_is_demoted_unless_query_names_its_subject(self) -> None:
        chunks = [
            Chunk(
                "search::c1",
                "search/search_latency_review.md",
                "This review concerns search ranking only. It contains no checkout evidence.",
                0,
                72,
            )
        ]

        self.assertEqual(
            build_scope_mismatch_penalties(
                "Why is checkout rollout paused even though latency checks passed?",
                chunks,
            ),
            {0: 0.75},
        )
        self.assertEqual(
            build_scope_mismatch_penalties("What does the search latency review conclude?", chunks),
            {},
        )

    def test_offline_query_expansion_bridges_gate_paraphrases(self) -> None:
        expanded = expand_query_for_retrieval(
            "Which teams must sign off and what proof do they need to restart?"
        )

        self.assertIn("approval", expanded)
        self.assertIn("evidence", expanded)
        self.assertIn("resume", expanded)
        self.assertIn("requires", expanded)
        self.assertIn(
            "approval",
            expand_query_for_retrieval("What approvals are still required?"),
        )
        generic_query = "What evidence is blocking the launch?"
        self.assertEqual(expand_query_for_retrieval(generic_query), generic_query)

    def test_non_evidence_penalty_requires_intent_and_subject_overlap(self) -> None:
        chunks = [
            Chunk(
                "load::c1",
                "engineering/load_test.md",
                "This test did not validate invoice tax correctness. "
                "Healthy load must not be treated as approval of VAT calculations.",
                0,
                119,
            )
        ]

        self.assertEqual(
            build_non_evidence_penalties(
                "Which evidence confirms the invoice tax defect?",
                chunks,
            ),
            {0: 0.60},
        )
        self.assertEqual(
            build_non_evidence_penalties(
                "Was service latency the root cause?",
                chunks,
            ),
            {},
        )

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

    def test_mmr_prefers_distinct_sources_before_repeating_chunks(self) -> None:
        embeddings = np.asarray(
            [[1.0, 0.0], [0.99, 0.01], [0.8, 0.2]],
            dtype=np.float32,
        )
        selected = mmr_select(
            [0, 1, 2],
            query_vec=np.asarray([1.0, 0.0], dtype=np.float32),
            doc_embeddings=embeddings,
            top_k=2,
            relevance_scores={0: 1.0, 1: 0.99, 2: 0.8},
            source_ids={0: "decision.md", 1: "decision.md", 2: "support.md"},
        )

        self.assertEqual(selected, [0, 2])

    def test_grounding_decision_abstains_when_subject_details_are_missing(self) -> None:
        rows = [
            evidence_row(1, "finance/migration_budget.md"),
            evidence_row(2, "operations/cutover_hold.md"),
        ]
        rows[0]["chunk"].text = "The database migration budget remains approved."
        rows[1]["chunk"].text = "The database cutover remains paused."

        supported = build_grounding_decision(
            "Why is the database cutover paused?",
            rows,
        )
        unsupported = build_grounding_decision(
            "What customer compensation amount and notification deadline were approved after the migration delay?",
            rows,
        )

        self.assertEqual(supported["outcome"], "answer")
        self.assertEqual(unsupported["outcome"], "abstain")
        self.assertLess(unsupported["best_query_coverage"], 0.40)

    def test_query_suite_validates_expected_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            suite_path = Path(temp_dir) / "queries.json"
            suite_path.write_text(
                '[{"query":"Known?","expected_outcome":"answer"},'
                '{"query":"Unknown?","expected_outcome":"abstain"}]',
                encoding="utf-8",
            )
            suite = load_query_suite(suite_path)
            self.assertEqual(
                [item["expected_outcome"] for item in suite],
                ["answer", "abstain"],
            )

            suite_path.write_text(
                '[{"query":"Maybe?","expected_outcome":"guess"}]',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "expected_outcome"):
                load_query_suite(suite_path)


class AnswerabilityMetricTests(unittest.TestCase):
    def test_metrics_keep_abstention_recall_visible(self) -> None:
        metrics = build_answerability_metrics(
            [
                {"label": "a", "expected_outcome": "answer", "actual_outcome": "answer"},
                {"label": "b", "expected_outcome": "abstain", "actual_outcome": "abstain"},
                {"label": "c", "expected_outcome": "abstain", "actual_outcome": "answer"},
            ]
        )

        self.assertEqual(metrics["accuracy"], 0.6667)
        self.assertEqual(metrics["abstention_recall"], 0.5)
        self.assertEqual(metrics["answer_recall"], 1.0)


class QualityGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = {
            "expected_source_recall": 0.9,
            "expected_source_mrr": 0.75,
            "variant_stability_rate": 0.8,
            "flagged_query_rate": 0.1,
            "forbidden_source_hit_rate": 0.0,
            "conflict_source_recall": 1.0,
            "answerability_accuracy": 1.0,
            "abstention_recall": 1.0,
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
            min_answerability_accuracy=1.0,
            min_abstention_recall=1.0,
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

    def test_gate_enforces_answerability_and_abstention_contracts(self) -> None:
        summary = {
            **self.summary,
            "answerability_accuracy": 0.75,
            "abstention_recall": 0.5,
        }

        gate = build_quality_gate(
            summary=summary,
            min_answerability_accuracy=1.0,
            min_abstention_recall=1.0,
        )

        self.assertFalse(gate["passed"])
        self.assertEqual(
            [check["metric"] for check in gate["checks"] if not check["passed"]],
            ["answerability_accuracy", "abstention_recall"],
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
