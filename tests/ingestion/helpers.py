from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paper_research_agent.ingestion.identity import make_page_id, sha256_text
from paper_research_agent.ingestion.models import DocumentElement, PageRecord
from paper_research_agent.ingestion.parser import (
    PARSER_NAME,
    PARSER_VERSION,
    ParsedDocument,
)


def fake_parse(path: Path, asset) -> ParsedDocument:
    page_id = make_page_id(asset.source_sha256, 1)
    text = "Evidence\u2028continued"
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


def write_manifests(
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

