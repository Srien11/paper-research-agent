"""把图片裁剪任务转换为严格的图片语义记录。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paper_research_agent.figures.models import FigureRecord
from paper_research_agent.figures.summarizer import (
    VisionSummarizer,
    ensure_unique_figure_ids,
)


def run_figure_summarization(
    candidates_path: Path,
    output_path: Path,
    summarizer: VisionSummarizer,
    *,
    limit: int | None = None,
) -> list[FigureRecord]:
    """逐张生成图片语义并可恢复地写入 figures.jsonl。"""

    if limit is not None and limit <= 0:
        raise ValueError("limit 必须为正整数")
    candidates = [
        json.loads(line)
        for line in candidates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ensure_unique_figure_ids(candidates)
    if limit is not None:
        candidates = candidates[:limit]

    existing = _read_existing(output_path)
    records_by_id = {record.figure_id: record for record in existing}
    candidates_by_id = {str(candidate["figure_id"]): candidate for candidate in candidates}
    unknown_existing = set(records_by_id) - set(candidates_by_id)
    if unknown_existing:
        raise ValueError("现有图片语义记录不属于当前候选清单")

    root = candidates_path.parent.resolve()
    for candidate in candidates:
        figure_id = str(candidate["figure_id"])
        if figure_id in records_by_id:
            continue
        image_path = _resolve_image_path(root, str(candidate["image_path"]))
        summary = summarizer.summarize(
            image_path,
            figure_name=str(candidate["figure_name"]),
            caption=str(candidate["caption"]),
        )
        record = FigureRecord(
            figure_id=figure_id,
            asset_id=str(candidate["asset_id"]),
            figure_name=str(candidate["figure_name"]),
            page_number=int(candidate["page_number"]),
            bbox=_parse_bbox(candidate["bbox"]),
            caption=str(candidate["caption"]),
            image_path=str(candidate["image_path"]),
            figure_type=summary.figure_type,
            summary=summary.summary,
            key_findings=summary.key_findings,
            recognition_confidence=summary.recognition_confidence,
            model_id=summarizer.model_id,
            prompt_version=summarizer.prompt_version,
        )
        records_by_id[record.figure_id] = record
        _write_records(output_path, list(records_by_id.values()))
    return sorted(
        records_by_id.values(),
        key=lambda record: (
            record.asset_id,
            record.page_number,
            record.figure_name,
            record.figure_id,
        ),
    )


def _read_existing(path: Path) -> list[FigureRecord]:
    if not path.exists():
        return []
    return [
        FigureRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_image_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("图片路径越出候选清单目录") from error
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在: {relative_path}")
    return path


def _parse_bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError("图片候选 bbox 必须包含四个坐标")
    values = tuple(float(item) for item in value)
    return values[0], values[1], values[2], values[3]


def _write_records(path: Path, records: list[FigureRecord]) -> None:
    ordered = sorted(
        records,
        key=lambda record: (
            record.asset_id,
            record.page_number,
            record.figure_name,
            record.figure_id,
        ),
    )
    content = "".join(
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in ordered
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)
