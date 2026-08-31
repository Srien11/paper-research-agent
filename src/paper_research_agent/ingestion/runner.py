"""运行全语料解析，并生成可复现的本地产物与清单。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from paper_research_agent.corpus import load_frozen_papers
from paper_research_agent.ingestion.identity import (
    make_asset_id,
    make_build_id,
    sha256_text,
)
from paper_research_agent.ingestion.models import (
    DocumentAsset,
    DocumentElement,
    IngestionManifest,
    PageRecord,
    SectionRecord,
)
from paper_research_agent.ingestion.ocr import create_default_ocr_backend
from paper_research_agent.ingestion.parser import (
    PARSER_NAME,
    PARSER_VERSION,
    ParsedDocument,
    parse_pdf_asset,
    parser_config,
)
from paper_research_agent.models import FrozenPaper

RecordT = TypeVar("RecordT", bound=BaseModel)


class IngestionRunError(RuntimeError):
    """全语料解析无法安全继续。"""


@dataclass(frozen=True)
class IngestionRunResult:
    output_dir: Path
    manifest: IngestionManifest


def run_incremental_corpus_ingestion(
    corpus_dir: Path,
    output_root: Path,
    *,
    parent_output_dir: Path,
) -> IngestionRunResult:
    """只解析新增源文件，并把旧产物原样合并到新的不可变构建。"""

    parent_manifest_path = parent_output_dir / "ingestion_manifest.json"
    if not parent_manifest_path.is_file():
        raise IngestionRunError(f"父构建缺少解析清单: {parent_manifest_path}")
    parent_manifest = IngestionManifest.model_validate_json(
        parent_manifest_path.read_text(encoding="utf-8")
    )
    papers = _load_and_validate_papers(corpus_dir)
    old_assets = _read_jsonl(parent_output_dir / "assets.jsonl", DocumentAsset)
    old_pages = _read_jsonl(parent_output_dir / "pages.jsonl", PageRecord)
    old_sections = _read_jsonl(parent_output_dir / "sections.jsonl", SectionRecord)
    old_elements = _read_jsonl(parent_output_dir / "elements.jsonl", DocumentElement)
    _validate_incremental_lineage(papers, old_assets)
    corpus_version = _single_corpus_version(papers)
    if corpus_version != parent_manifest.corpus_version:
        raise IngestionRunError("增量更新不允许跨语料版本合并")

    ocr_backend = create_default_ocr_backend()
    config = parser_config(ocr_backend)
    config_sha256 = sha256_text(_canonical_json(config))
    if config_sha256 != parent_manifest.parser_config_sha256:
        raise IngestionRunError("解析器或 OCR 配置已变化；请执行新的全量构建")

    old_hashes = {asset.source_sha256 for asset in old_assets}
    new_papers = [paper for paper in papers if paper.download_sha256 not in old_hashes]
    if not new_papers:
        return IngestionRunResult(output_dir=parent_output_dir, manifest=parent_manifest)

    build_id = make_build_id(
        corpus_version,
        PARSER_NAME,
        PARSER_VERSION,
        config_sha256,
        (paper.download_sha256 for paper in papers),
    )
    output_dir = output_root / corpus_version / build_id
    if output_dir.exists():
        manifest_path = output_dir / "ingestion_manifest.json"
        if manifest_path.is_file():
            return IngestionRunResult(
                output_dir=output_dir,
                manifest=IngestionManifest.model_validate_json(manifest_path.read_text(encoding="utf-8")),
            )
        raise IngestionRunError(f"目标构建目录已存在但不完整: {output_dir}")

    new_assets = [_asset_from_paper(paper) for paper in new_papers]
    new_documents = [
        parse_pdf_asset(paper.local_pdf_path, asset, ocr_backend=ocr_backend)
        for paper, asset in zip(new_papers, new_assets, strict=True)
    ]
    assets = sorted([*old_assets, *new_assets], key=lambda item: item.corpus_id)
    pages = sorted(
        [*old_pages, *(page for document in new_documents for page in document.pages)],
        key=lambda item: (item.corpus_id, item.page_number),
    )
    sections = sorted(
        [*old_sections, *(section for document in new_documents for section in document.sections)],
        key=lambda item: (item.corpus_id, item.ordinal),
    )
    elements = sorted(
        [*old_elements, *(element for document in new_documents for element in document.elements)],
        key=lambda item: (item.corpus_id, item.page_number, item.reading_order),
    )
    _require_unique("asset_id", assets, lambda item: item.asset_id)
    _require_unique("page_id", pages, lambda item: item.page_id)
    _require_unique("section_id", sections, lambda item: item.section_id)
    _require_unique("element_id", elements, lambda item: item.element_id)
    return _write_ingestion_build(
        output_dir,
        assets=assets,
        pages=pages,
        sections=sections,
        elements=elements,
        config=config,
        corpus_version=corpus_version,
        build_id=build_id,
        config_sha256=config_sha256,
        parent_build_id=parent_manifest.build_id,
        added_asset_count=len(new_assets),
    )


def run_corpus_ingestion(
    corpus_dir: Path,
    output_root: Path,
) -> IngestionRunResult:
    """校验源文件，顺序解析全部论文，并写出确定性产物。"""

    papers = _load_and_validate_papers(corpus_dir)
    corpus_version = _single_corpus_version(papers)
    ocr_backend = create_default_ocr_backend()
    config = parser_config(ocr_backend)
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
        parsed_documents.append(
            parse_pdf_asset(paper.local_pdf_path, asset, ocr_backend=ocr_backend)
        )

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

    return _write_ingestion_build(
        output_dir,
        assets=assets,
        pages=pages,
        sections=sections,
        elements=elements,
        config=config,
        corpus_version=corpus_version,
        build_id=build_id,
        config_sha256=config_sha256,
        parent_build_id=None,
        added_asset_count=len(assets),
    )


def _write_ingestion_build(
    output_dir: Path,
    *,
    assets: Sequence[DocumentAsset],
    pages: Sequence[PageRecord],
    sections: Sequence[SectionRecord],
    elements: Sequence[DocumentElement],
    config: dict[str, object],
    corpus_version: str,
    build_id: str,
    config_sha256: str,
    parent_build_id: str | None,
    added_asset_count: int,
) -> IngestionRunResult:
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
        parent_build_id=parent_build_id,
        added_asset_count=added_asset_count,
    )
    _write_json(output_dir / "ingestion_manifest.json", manifest)
    return IngestionRunResult(output_dir=output_dir, manifest=manifest)


def _load_and_validate_papers(corpus_dir: Path) -> list[FrozenPaper]:
    papers = load_frozen_papers(
        [corpus_dir / "core_frozen.jsonl", corpus_dir / "challenge_frozen.jsonl"]
    )
    papers = sorted(papers, key=lambda paper: paper.corpus_id)
    _validate_source_assets(papers)
    return papers


def _single_corpus_version(papers: Sequence[FrozenPaper]) -> str:
    versions = {paper.corpus_version for paper in papers}
    if len(versions) != 1:
        raise IngestionRunError(f"语料版本不唯一: {sorted(versions)}")
    return next(iter(versions))


def _read_jsonl(path: Path, model: type[RecordT]) -> list[RecordT]:
    if not path.is_file():
        raise IngestionRunError(f"父构建缺少产物: {path}")
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_incremental_lineage(
    papers: Sequence[FrozenPaper],
    old_assets: Sequence[DocumentAsset],
) -> None:
    papers_by_hash = {paper.download_sha256: paper for paper in papers}
    for asset in old_assets:
        paper = papers_by_hash.get(asset.source_sha256)
        if paper is None:
            raise IngestionRunError(f"增量清单缺少既有来源: {asset.corpus_id}")
        if paper.corpus_id != asset.corpus_id:
            raise IngestionRunError(f"既有来源不得重新分配语料 ID: {asset.corpus_id}")
    _require_unique("corpus_id", papers, lambda item: item.corpus_id)
    _require_unique("source_sha256", papers, lambda item: item.download_sha256)


def _require_unique(name: str, values: Sequence[Any], key: Any) -> None:
    keys = [key(value) for value in values]
    duplicates = sorted({value for value in keys if keys.count(value) > 1})
    if duplicates:
        raise IngestionRunError(f"重复 {name}: {duplicates[:3]}")


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
