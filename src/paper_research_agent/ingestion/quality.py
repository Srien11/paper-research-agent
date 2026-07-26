"""对本地解析产物执行跨记录完整性与文本质量审计。"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from paper_research_agent.ingestion.models import (
    DocumentAsset,
    DocumentElement,
    IngestionManifest,
    PageRecord,
    SectionRecord,
)


class QualityAuditError(RuntimeError):
    """质量审计无法读取或验证产物。"""


class QualityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["ingestion-quality-v1"] = "ingestion-quality-v1"
    build_id: str
    corpus_version: str
    status: Literal["pass", "manual_review_required", "fail"]
    counts: dict[str, int]
    integrity_errors: dict[str, int]
    gates: dict[str, bool]
    text_quality: dict[str, int | float | list[str]]
    structure_quality: dict[str, int | float | list[str]]
    manual_review_ids: list[str] = Field(default_factory=list)
    known_warnings: list[str] = Field(default_factory=list)


def assess_ingestion(
    build_dir: Path,
    *,
    minimum_chars_per_paper: int = 4_000,
    minimum_sections_per_paper: int = 3,
    manual_review_ids: tuple[str, ...] = (),
    known_warnings: tuple[str, ...] = (),
) -> QualityAssessment:
    """读取一个确定性构建目录并生成不含论文正文的聚合评估。"""

    manifest = IngestionManifest.model_validate(
        _read_json(build_dir / "ingestion_manifest.json")
    )
    assets = _read_jsonl(build_dir / "assets.jsonl", DocumentAsset)
    pages = _read_jsonl(build_dir / "pages.jsonl", PageRecord)
    sections = _read_jsonl(build_dir / "sections.jsonl", SectionRecord)
    elements = _read_jsonl(build_dir / "elements.jsonl", DocumentElement)

    errors: Counter[str] = Counter()
    _check_artifact_hashes(build_dir, manifest, errors)
    _check_unique_ids(assets, pages, sections, elements, errors)
    _check_record_links(assets, pages, sections, elements, errors)
    _check_manifest_counts(manifest, assets, pages, sections, elements, errors)

    pages_per_corpus = Counter(page.corpus_id for page in pages)
    sections_per_corpus = Counter(section.corpus_id for section in sections)
    elements_per_corpus = Counter(element.corpus_id for element in elements)
    chars_per_corpus: Counter[str] = Counter()
    replacement_characters = 0
    total_normalized_characters = 0
    for page in pages:
        text = page.normalized_text or ""
        chars_per_corpus[page.corpus_id] += len(text)
        total_normalized_characters += len(text)
        replacement_characters += text.count("\ufffd")

    corpus_ids = sorted(asset.corpus_id for asset in assets)
    low_text_ids = [
        corpus_id
        for corpus_id in corpus_ids
        if chars_per_corpus[corpus_id] < minimum_chars_per_paper
    ]
    low_section_ids = [
        corpus_id
        for corpus_id in corpus_ids
        if sections_per_corpus[corpus_id] < minimum_sections_per_paper
    ]
    section_density_outliers = [
        corpus_id
        for corpus_id in corpus_ids
        if sections_per_corpus[corpus_id]
        > max(50, pages_per_corpus[corpus_id] * 3)
    ]
    replacement_ratio = (
        replacement_characters / total_normalized_characters
        if total_normalized_characters
        else 0.0
    )

    gates = {
        "全部页面有状态记录": len(pages) == manifest.expected_page_count,
        "零失败或隔离页面": manifest.failed_page_count == 0
        and manifest.quarantined_page_count == 0,
        "零跨记录完整性错误": sum(errors.values()) == 0,
        "每篇正文字符量达标": not low_text_ids,
        "替换字符比例不高于千分之一": replacement_ratio <= 0.001,
        "每篇至少识别三个章节": not low_section_ids,
    }
    status: Literal["pass", "manual_review_required", "fail"]
    if not all(gates.values()):
        status = "fail"
    elif manual_review_ids or known_warnings or section_density_outliers:
        status = "manual_review_required"
    else:
        status = "pass"

    section_counts = list(sections_per_corpus.values())
    return QualityAssessment(
        build_id=manifest.build_id,
        corpus_version=manifest.corpus_version,
        status=status,
        counts={
            "assets": len(assets),
            "pages": len(pages),
            "sections": len(sections),
            "elements": len(elements),
            "normalized_characters": total_normalized_characters,
        },
        integrity_errors=dict(sorted(errors.items())),
        gates=gates,
        text_quality={
            "minimum_chars_per_paper": min(chars_per_corpus.values(), default=0),
            "maximum_chars_per_paper": max(chars_per_corpus.values(), default=0),
            "replacement_characters": replacement_characters,
            "replacement_ratio": replacement_ratio,
            "low_text_corpus_ids": low_text_ids,
        },
        structure_quality={
            "minimum_sections_per_paper": min(section_counts, default=0),
            "median_sections_per_paper": (
                float(statistics.median(section_counts)) if section_counts else 0.0
            ),
            "maximum_sections_per_paper": max(section_counts, default=0),
            "low_section_corpus_ids": low_section_ids,
            "section_density_outlier_ids": section_density_outliers,
            "maximum_elements_per_paper": max(elements_per_corpus.values(), default=0),
        },
        manual_review_ids=sorted(set(manual_review_ids)),
        known_warnings=list(known_warnings),
    )


def write_quality_report(path: Path, assessment: QualityAssessment) -> None:
    """写出不包含正文和本地路径的中文质量报告。"""

    content = json.dumps(
        assessment.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(f"{content}\n", encoding="utf-8", newline="\n")
    temporary_path.replace(path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityAuditError(f"无法读取 {path.name}: {exc}") from exc


def _read_jsonl(path: Path, model_type: type[BaseModel]) -> list[Any]:
    records: list[Any] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    records.append(model_type.model_validate_json(line))
    except (OSError, ValueError) as exc:
        raise QualityAuditError(
            f"无法读取 {path.name} 第 {line_number if 'line_number' in locals() else 0} 行: {exc}"
        ) from exc
    return records


def _check_artifact_hashes(
    build_dir: Path,
    manifest: IngestionManifest,
    errors: Counter[str],
) -> None:
    for name, expected_hash in manifest.artifact_sha256.items():
        path = build_dir / name
        if not path.is_file() or _sha256_file(path) != expected_hash:
            errors["artifact_hash_mismatch"] += 1


def _check_unique_ids(
    assets: list[DocumentAsset],
    pages: list[PageRecord],
    sections: list[SectionRecord],
    elements: list[DocumentElement],
    errors: Counter[str],
) -> None:
    for name, values in (
        ("asset", [record.asset_id for record in assets]),
        ("page", [record.page_id for record in pages]),
        ("section", [record.section_id for record in sections]),
        ("element", [record.element_id for record in elements]),
    ):
        duplicate_count = len(values) - len(set(values))
        if duplicate_count:
            errors[f"duplicate_{name}_id"] += duplicate_count


def _check_record_links(
    assets: list[DocumentAsset],
    pages: list[PageRecord],
    sections: list[SectionRecord],
    elements: list[DocumentElement],
    errors: Counter[str],
) -> None:
    assets_by_id = {asset.asset_id: asset for asset in assets}
    pages_by_id = {page.page_id: page for page in pages}
    sections_by_id = {section.section_id: section for section in sections}
    page_numbers_by_asset: dict[str, list[int]] = defaultdict(list)

    for page in pages:
        page_numbers_by_asset[page.asset_id].append(page.page_number)
        asset = assets_by_id.get(page.asset_id)
        if asset is None:
            errors["orphan_page"] += 1
        elif (
            page.corpus_id != asset.corpus_id
            or page.source_sha256 != asset.source_sha256
        ):
            errors["page_asset_mismatch"] += 1
    for asset in assets:
        actual_pages = sorted(page_numbers_by_asset[asset.asset_id])
        expected_pages = list(range(1, asset.expected_page_count + 1))
        if actual_pages != expected_pages:
            errors["non_contiguous_pages"] += 1

    for section in sections:
        asset = assets_by_id.get(section.asset_id)
        if asset is None:
            errors["orphan_section"] += 1
        elif (
            section.corpus_id != asset.corpus_id
            or section.end_page > asset.expected_page_count
        ):
            errors["section_asset_mismatch"] += 1
        if (
            section.parent_section_id is not None
            and section.parent_section_id not in sections_by_id
        ):
            errors["orphan_parent_section"] += 1

    reading_orders: dict[str, set[int]] = defaultdict(set)
    for element in elements:
        page = pages_by_id.get(element.page_id)
        if page is None:
            errors["orphan_element_page"] += 1
            continue
        if (
            element.asset_id != page.asset_id
            or element.corpus_id != page.corpus_id
            or element.page_number != page.page_number
            or element.source_sha256 != page.source_sha256
        ):
            errors["element_page_mismatch"] += 1
        if element.reading_order in reading_orders[element.page_id]:
            errors["duplicate_reading_order"] += 1
        reading_orders[element.page_id].add(element.reading_order)
        if element.section_id is not None:
            section = sections_by_id.get(element.section_id)
            if section is None:
                errors["orphan_element_section"] += 1
            elif section.asset_id != element.asset_id:
                errors["element_section_mismatch"] += 1
        if (
            page.raw_text is not None
            and element.raw_start is not None
            and element.raw_end is not None
        ):
            if page.raw_text[element.raw_start : element.raw_end] != element.raw_text:
                errors["raw_span_mismatch"] += 1
        if element.bbox is not None:
            x0, y0, x1, y1 = element.bbox
            if (
                x0 < -1
                or y0 < -1
                or x1 > page.width_points + 1
                or y1 > page.height_points + 1
            ):
                errors["bbox_out_of_page"] += 1


def _check_manifest_counts(
    manifest: IngestionManifest,
    assets: list[DocumentAsset],
    pages: list[PageRecord],
    sections: list[SectionRecord],
    elements: list[DocumentElement],
    errors: Counter[str],
) -> None:
    expected = {
        "assets": manifest.asset_count,
        "pages": manifest.expected_page_count,
        "sections": manifest.section_count,
        "elements": manifest.element_count,
    }
    actual = {
        "assets": len(assets),
        "pages": len(pages),
        "sections": len(sections),
        "elements": len(elements),
    }
    for key in expected:
        if expected[key] != actual[key]:
            errors[f"manifest_{key}_count_mismatch"] += 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
