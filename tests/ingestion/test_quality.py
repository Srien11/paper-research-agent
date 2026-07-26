from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.ingestion.quality import assess_ingestion
from paper_research_agent.ingestion.runner import run_corpus_ingestion
from ingestion.helpers import fake_parse, write_manifests


class IngestionQualityTests(unittest.TestCase):
    def test_valid_build_passes_integrity_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_dir = self._build_fixture(Path(directory))

            assessment = assess_ingestion(
                build_dir,
                minimum_chars_per_paper=1,
                minimum_sections_per_paper=0,
            )

            self.assertEqual(assessment.status, "pass")
            self.assertTrue(assessment.gates["零跨记录完整性错误"])
            self.assertEqual(assessment.integrity_errors, {})

    def test_tampered_artifact_fails_hash_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_dir = self._build_fixture(Path(directory))
            pages_path = build_dir / "pages.jsonl"
            pages_path.write_text(
                pages_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )

            assessment = assess_ingestion(
                build_dir,
                minimum_chars_per_paper=1,
                minimum_sections_per_paper=0,
            )

            self.assertEqual(assessment.status, "fail")
            self.assertEqual(
                assessment.integrity_errors["artifact_hash_mismatch"],
                1,
            )

    def test_known_warning_is_distinct_from_pending_manual_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_dir = self._build_fixture(Path(directory))

            assessment = assess_ingestion(
                build_dir,
                minimum_chars_per_paper=1,
                minimum_sections_per_paper=0,
                known_warnings=("人工复核后确认不阻塞检索。",),
            )

            self.assertEqual(assessment.status, "pass_with_warnings")
            self.assertEqual(
                assessment.known_warnings,
                ["人工复核后确认不阻塞检索。"],
            )

    @staticmethod
    def _build_fixture(root: Path) -> Path:
        corpus_dir = root / "corpus"
        corpus_dir.mkdir()
        source = corpus_dir / "paper.pdf"
        source.write_bytes(b"%PDF synthetic fixture")
        write_manifests(corpus_dir, source)
        with patch(
            "paper_research_agent.ingestion.runner.parse_pdf_asset",
            side_effect=fake_parse,
        ):
            result = run_corpus_ingestion(corpus_dir, root / "output")
        return result.output_dir


if __name__ == "__main__":
    unittest.main()
