from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.ingestion.identity import make_page_id, sha256_text
from paper_research_agent.ingestion.models import (
    DocumentElement,
    PageRecord,
)
from paper_research_agent.ingestion.parser import (
    PARSER_NAME,
    PARSER_VERSION,
    ParsedDocument,
)
from paper_research_agent.ingestion.runner import (
    IngestionRunError,
    run_corpus_ingestion,
)


class IngestionRunnerTests(unittest.TestCase):
    def test_run_is_deterministic_and_writes_no_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_dir = root / "corpus"
            output_root = root / "output"
            corpus_dir.mkdir()
            source = corpus_dir / "paper.pdf"
            source.write_bytes(b"%PDF synthetic fixture")
            self._write_manifests(corpus_dir, source)

            with patch(
                "paper_research_agent.ingestion.runner.parse_pdf_asset",
                side_effect=self._fake_parse,
            ):
                first = run_corpus_ingestion(corpus_dir, output_root)
                first_hashes = first.manifest.artifact_sha256
                second = run_corpus_ingestion(corpus_dir, output_root)

            self.assertEqual(first.output_dir, second.output_dir)
            self.assertEqual(first_hashes, second.manifest.artifact_sha256)
            assets_text = (first.output_dir / "assets.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(str(source), assets_text)
            self.assertEqual(first.manifest.asset_count, 1)
            self.assertEqual(first.manifest.parsed_page_count, 1)

    def test_hash_mismatch_stops_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_dir = root / "corpus"
            corpus_dir.mkdir()
            source = corpus_dir / "paper.pdf"
            source.write_bytes(b"%PDF synthetic fixture")
            self._write_manifests(corpus_dir, source, sha256_override="0" * 64)

            with self.assertRaisesRegex(IngestionRunError, "SHA-256 不一致"):
                run_corpus_ingestion(corpus_dir, root / "output")

    @staticmethod
    def _fake_parse(path: Path, asset) -> ParsedDocument:
        page_id = make_page_id(asset.source_sha256, 1)
        text = "Evidence"
        text_hash = sha256_text(text)
        page = PageRecord(
            page_id=page_id,
            asset_id=asset.asset_id,
            corpus_id=asset.corpus_id,
            page_number=1,
            status="parsed",
            raw_text=text,
            normalized_text=text,
            raw_text_sha256=text_hash,
            normalized_text_sha256=text_hash,
            width_points=100,
            height_points=100,
            source_sha256=asset.source_sha256,
            parser_name=PARSER_NAME,
            parser_version=PARSER_VERSION,
        )
        element = DocumentElement(
            element_id=f"element-{asset.corpus_id}",
            asset_id=asset.asset_id,
            page_id=page_id,
            corpus_id=asset.corpus_id,
            page_number=1,
            element_type="paragraph",
            reading_order=0,
            raw_text=text,
            normalized_text=text,
            normalized_text_sha256=text_hash,
            source_sha256=asset.source_sha256,
            parser_name=PARSER_NAME,
            parser_version=PARSER_VERSION,
        )
        return ParsedDocument(pages=(page,), sections=(), elements=(element,))

    @staticmethod
    def _write_manifests(
        corpus_dir: Path,
        source: Path,
        *,
        sha256_override: str | None = None,
    ) -> None:
        source_sha256 = sha256_override or hashlib.sha256(source.read_bytes()).hexdigest()
        record = {
            "corpus_id": "C001",
            "corpus_version": "corpus-v1",
            "dataset_split": "core",
            "canonical_key": "test:paper",
            "title": "Synthetic paper",
            "year": 2026,
            "authors": ["Test Author"],
            "official_url": "https://example.test/paper",
            "fulltext_url": "https://example.test/paper.pdf",
            "selection_status": "frozen",
            "content_status": "downloaded_and_parse_verified",
            "storage_class": "internal_research_only",
            "local_pdf_path": str(source),
            "download_sha256": source_sha256,
            "download_bytes": source.stat().st_size,
            "pdf_pages": 1,
            "parse_quality_status": "machine_parse_pass",
        }
        (corpus_dir / "core_frozen.jsonl").write_text(
            f"{json.dumps(record)}\n",
            encoding="utf-8",
        )
        (corpus_dir / "challenge_frozen.jsonl").write_text("", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

