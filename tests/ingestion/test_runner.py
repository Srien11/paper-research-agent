from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.ingestion.runner import (
    IngestionRunError,
    run_corpus_ingestion,
)
from ingestion.helpers import fake_parse, write_manifests


class IngestionRunnerTests(unittest.TestCase):
    def test_run_is_deterministic_and_writes_no_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_dir = root / "corpus"
            output_root = root / "output"
            corpus_dir.mkdir()
            source = corpus_dir / "paper.pdf"
            source.write_bytes(b"%PDF synthetic fixture")
            write_manifests(corpus_dir, source)

            with patch(
                "paper_research_agent.ingestion.runner.parse_pdf_asset",
                side_effect=fake_parse,
            ):
                first = run_corpus_ingestion(corpus_dir, output_root)
                first_hashes = first.manifest.artifact_sha256
                second = run_corpus_ingestion(corpus_dir, output_root)

            self.assertEqual(first.output_dir, second.output_dir)
            self.assertEqual(first_hashes, second.manifest.artifact_sha256)
            assets_text = (first.output_dir / "assets.jsonl").read_text(encoding="utf-8")
            pages_text = (first.output_dir / "pages.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(str(source), assets_text)
            self.assertEqual(len(pages_text.splitlines()), 1)
            self.assertIn("\\u2028", pages_text)
            self.assertEqual(first.manifest.asset_count, 1)
            self.assertEqual(first.manifest.parsed_page_count, 1)

    def test_hash_mismatch_stops_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_dir = root / "corpus"
            corpus_dir.mkdir()
            source = corpus_dir / "paper.pdf"
            source.write_bytes(b"%PDF synthetic fixture")
            write_manifests(corpus_dir, source, sha256_override="0" * 64)

            with self.assertRaisesRegex(IngestionRunError, "SHA-256 不一致"):
                run_corpus_ingestion(corpus_dir, root / "output")

if __name__ == "__main__":
    unittest.main()
