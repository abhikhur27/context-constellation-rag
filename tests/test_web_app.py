import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import web_app
from main import Chunk


class WebApiTests(unittest.TestCase):
    def test_ask_exposes_grounding_decision(self) -> None:
        result = {
            "answer": "Insufficient evidence.",
            "answer_mode": "abstained",
            "grounding": {
                "outcome": "abstain",
                "label": "insufficient-evidence",
                "best_query_coverage": 0.2,
            },
            "evidence_posture": {
                "coverage_label": "moderate",
                "tension_label": "cross-supported",
            },
            "evidence": [
                {
                    "citation": "C1",
                    "constellation": "K0 (recovery)",
                    "chunk": Chunk(
                        chunk_id="recovery.md::c1",
                        source="recovery.md",
                        text="Closest available recovery note.",
                        start=0,
                        end=32,
                    ),
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(web_app, "DEFAULT_INDEX", Path(temp_dir)),
                patch.object(web_app, "query_index", return_value=result),
            ):
                response = web_app.app.test_client().post(
                    "/api/ask",
                    json={"query": "What unsupported detail exists?", "llm": "off"},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["answer_mode"], "abstained")
        self.assertEqual(payload["grounding"]["outcome"], "abstain")
        self.assertEqual(payload["evidence"][0]["source"], "recovery.md")


if __name__ == "__main__":
    unittest.main()
