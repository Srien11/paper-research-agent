from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.corpus import CorpusValidationError, validate_corpus_files


class CorpusValidationTests(unittest.TestCase):
    def test_valid_fixture_returns_deterministic_report(self) -> None:
        fixtures = PROJECT_ROOT / "tests" / "fixtures"

        report = validate_corpus_files(
            [fixtures / "core.jsonl", fixtures / "challenge.jsonl"],
            require_local_pdfs=False,
        )

        self.assertEqual(report.corpus_version, "test-v1")
        self.assertEqual(report.paper_count, 2)
        self.assertEqual(report.core_count, 1)
        self.assertEqual(report.challenge_count, 1)
        self.assertEqual(report.total_pages, 30)
        self.assertEqual(report.canonical_key_count, 2)

    def test_duplicate_canonical_key_is_rejected(self) -> None:
        fixtures = PROJECT_ROOT / "tests" / "fixtures"
        core_record = (fixtures / "core.jsonl").read_text(encoding="utf-8")
        duplicate = core_record.replace('"corpus_id":"C001"', '"corpus_id":"C002"')

        with tempfile.TemporaryDirectory() as directory:
            duplicate_path = Path(directory) / "duplicates.jsonl"
            duplicate_path.write_text(core_record + duplicate, encoding="utf-8")

            with self.assertRaisesRegex(CorpusValidationError, "duplicate canonical_key"):
                validate_corpus_files(
                    [duplicate_path],
                    require_local_pdfs=False,
                )

    def test_missing_pdf_is_rejected_when_gate_is_enabled(self) -> None:
        fixtures = PROJECT_ROOT / "tests" / "fixtures"

        with self.assertRaisesRegex(CorpusValidationError, "missing local PDFs"):
            validate_corpus_files(
                [fixtures / "core.jsonl"],
                require_local_pdfs=True,
            )


if __name__ == "__main__":
    unittest.main()

