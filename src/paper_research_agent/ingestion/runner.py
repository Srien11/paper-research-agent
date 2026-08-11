"""运行全语料解析，并生成可复现的本地产物与清单。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from paper_research_agent.corpus import load_frozen_papers
from paper_research_agent.ingestion.identity import (
    make_asset_id,
    make_build_id,
    sha256_text,
)
from paper_research_agent.ingestion.models import DocumentAsset, IngestionManifest
from paper_research_agent.ingestion.parser import (
    PARSER_NAME,
    PARSER_VERSION,
    ParsedDocument,
    parse_pdf_asset,
    parser_config,
)
from paper_research_agent.models import FrozenPaper


class IngestionRunError(RuntimeError):
    """全语料解析无法安全继续。"""


@dataclass(frozen=True)
class IngestionRunResult:
    output_dir: Path
    manifest: IngestionManifest


def run_corpus_ingestion(
    corpus_dir: Path,
    output_root: Path,
) -> IngestionRunResult:
    """校验源文件，顺序解析全部论文，并写出确定性产物。"""

    papers = load_frozen_papers(
        [
            corpus_dir / "core_frozen.jsonl",
            corpus_dir / "challenge_frozen.jsonl",
        ]
    )
    papers = sorted(papers, key=lambda paper: paper.corpus_id)
    _validate_source_assets(papers)

    versions = {paper.corpus_version for paper in papers}
    if len(versions) != 1:
        raise IngestionRunError(f"语料版本不唯一: {sorted(versions)}")
    corpus_version = next(iter(versions))
    config = parser_config()
    config_json = _canonical_json(config)
    config_sha256 = sha256_text(config_json)
    build_id = make_build_id(
        corpus_version,
        PARSER_NAME,
        PARSER_VERSION,
        config_sha256,
        (paper.download_sha256 for paper in papers),
    )
    output_dir = output_root / corpus_version / build_id

    assets: list[DocumentAsset] = []
    parsed_documents: list[ParsedDocument] = []
    for paper in papers:
        asset = _asset_from_paper(paper)
        assets.append(asset)
        parsed_documents.append(parse_pdf_asset(paper.local_pdf_path, asset))

    pages = sorted(
        (page for document in parsed_documents for page in document.pages),
        key=lambda page: (page.corpus_id, page.page_number),
    )
    sections = sorted(
        (section for document in parsed_documents for section in document.sections),
        key=lambda section: (section.corpus_id, section.ordinal),
    )
    elements = sorted(
        (element for document in parsed_documents for element in document.elements),
        key=lambda element: (
            element.corpus_id,
            element.page_number,
            element.reading_order,
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "parser_config.json": _write_json(output_dir / "parser_config.json", config),
        "assets.jsonl": _write_jsonl(output_dir / "assets.jsonl", assets),
        "pages.jsonl": _write_jsonl(output_dir / "pages.jsonl", pages),
        "sections.jsonl": _write_jsonl(output_dir / "sections.jsonl", sections),
        "elements.jsonl": _write_jsonl(output_dir / "elements.jsonl", elements),
    }
    artifact_sha256 = {
        name: _sha256_file(path)
        for name, path in sorted(artifact_paths.items())
    }
    manifest = IngestionManifest(
        build_id=build_id,
        corpus_version=corpus_version,
        parser_name=PARSER_NAME,
        parser_version=PARSER_VERSION,
        parser_config_sha256=config_sha256,
        asset_count=len(assets),
        expected_page_count=sum(asset.expected_page_count for asset in assets),
        parsed_page_count=sum(page.status == "parsed" for page in pages),
        empty_page_count=sum(page.status == "empty" for page in pages),
        failed_page_count=sum(page.status == "failed" for page in pages),
        quarantined_page_count=sum(page.status == "quarantined" for page in pages),
        section_count=len(sections),
        element_count=len(elements),
        artifact_sha256=artifact_sha256,
    )
    _write_json(output_dir / "ingestion_manifest.json", manifest)
    return IngestionRunResult(output_dir=output_dir, manifest=manifest)


def _asset_from_paper(paper: FrozenPaper) -> DocumentAsset:
    return DocumentAsset(
        asset_id=make_asset_id(paper.download_sha256),
        corpus_id=paper.corpus_id,
        corpus_version=paper.corpus_version,
        source_sha256=paper.download_sha256,
        source_bytes=paper.download_bytes,
        expected_page_count=paper.pdf_pages,
        storage_class=paper.storage_class,
    )


def _validate_source_assets(papers: list[FrozenPaper]) -> None:
    errors: list[str] = []
    for paper in papers:
        path = paper.local_pdf_path
        if not path.is_file():
            errors.append(f"{paper.corpus_id}: PDF 不存在")
            continue
        actual_bytes = path.stat().st_size
        if actual_bytes != paper.download_bytes:
            errors.append(
                f"{paper.corpus_id}: 文件大小不一致 "
                f"{actual_bytes} != {paper.download_bytes}"
            )
            continue
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != paper.download_sha256:
            errors.append(f"{paper.corpus_id}: SHA-256 不一致")
    if errors:
        raise IngestionRunError("\n".join(errors))


def _write_jsonl(path: Path, records: Sequence[BaseModel]) -> Path:
    content = "".join(
        f"{_canonical_json(record.model_dump(mode='json'))}\n"
        for record in records
    )
    _atomic_write(path, content.encode("utf-8"))
    return path


def _write_json(path: Path, value: BaseModel | dict[str, Any]) -> Path:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    content = f"{_canonical_json(payload)}\n"
    _atomic_write(path, content.encode("utf-8"))
    return path


def _atomic_write(path: Path, content: bytes) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def _canonical_json(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return serialized.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
