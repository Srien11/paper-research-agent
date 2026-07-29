"""运行全语料图片裁剪并生成本地任务清单。"""

from __future__ import annotations

import json
from pathlib import Path

from paper_research_agent.corpus import load_frozen_papers
from paper_research_agent.figures.cropper import FigureCrop, crop_pdf_figures
from paper_research_agent.ingestion.models import DocumentElement
from paper_research_agent.ingestion.text import normalize_text


def run_figure_cropping(
    elements_path: Path,
    corpus_dir: Path,
    output_dir: Path,
    *,
    dpi: int = 160,
    limit: int | None = None,
) -> tuple[Path, list[FigureCrop]]:
    """裁剪全部图注对应区域，并写出可恢复的视觉识别任务清单。"""

    if limit is not None and limit <= 0:
        raise ValueError("limit 必须为正整数")
    papers = load_frozen_papers(
        [
            corpus_dir / "core_frozen.jsonl",
            corpus_dir / "challenge_frozen.jsonl",
        ]
    )
    papers_by_corpus = {paper.corpus_id: paper for paper in papers}
    elements = [
        DocumentElement.model_validate_json(line)
        for line in elements_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    captions = [
        element
        for element in elements
        if element.element_type == "figure_caption" and element.normalized_text.strip()
    ]
    captions.sort(
        key=lambda element: (
            element.corpus_id,
            element.page_number,
            element.reading_order,
            element.element_id,
        )
    )
    if limit is not None:
        captions = captions[:limit]

    page_elements: dict[tuple[str, int], list[DocumentElement]] = {}
    for element in elements:
        page_elements.setdefault((element.corpus_id, element.page_number), []).append(
            element
        )
    grouped: dict[str, list[dict[str, object]]] = {}
    for caption in captions:
        payload = caption.model_dump(mode="json")
        payload["normalized_text"] = merge_caption_text(
            caption,
            page_elements[(caption.corpus_id, caption.page_number)],
        )
        grouped.setdefault(caption.corpus_id, []).append(payload)

    crops: list[FigureCrop] = []
    for corpus_id in sorted(grouped):
        paper = papers_by_corpus.get(corpus_id)
        if paper is None:
            raise ValueError(f"图注没有对应冻结论文: {corpus_id}")
        crops.extend(
            crop_pdf_figures(
                paper.local_pdf_path,
                grouped[corpus_id],
                output_dir,
                dpi=dpi,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "figure_candidates.jsonl"
    content = "".join(
        json.dumps(
            crop.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for crop in crops
    )
    temporary_path = manifest_path.with_suffix(".jsonl.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(manifest_path)
    prune_orphaned_crops(output_dir, crops)
    return manifest_path, crops


def merge_caption_text(
    caption: DocumentElement,
    page_elements: list[DocumentElement],
) -> str:
    """合并同栏、紧邻图注首行的后续文本行。"""

    if caption.bbox is None:
        return caption.normalized_text
    ordered = sorted(page_elements, key=lambda element: element.reading_order)
    try:
        position = next(
            index
            for index, element in enumerate(ordered)
            if element.element_id == caption.element_id
        )
    except StopIteration:
        return caption.normalized_text

    x0, top, x1, bottom = caption.bbox
    line_height = max(1.0, bottom - top)
    current_bottom = bottom
    parts = [caption.normalized_text]
    for candidate in ordered[position + 1 :]:
        if candidate.bbox is None:
            break
        if candidate.element_type in {
            "title",
            "heading",
            "table_caption",
            "figure_caption",
            "reference",
        }:
            break
        candidate_x0, candidate_top, candidate_x1, candidate_bottom = candidate.bbox
        if candidate_top > current_bottom + max(6.0, line_height * 0.9):
            break
        if candidate_x1 < x0 - 6 or candidate_x0 > x1 + 6:
            break
        parts.append(candidate.normalized_text)
        current_bottom = max(current_bottom, candidate_bottom)
    return normalize_text(" ".join(parts))


def prune_orphaned_crops(
    output_dir: Path,
    crops: list[FigureCrop],
) -> int:
    """只删除当前图片输出目录中未被候选清单引用的旧 PNG。"""

    output_root = output_dir.resolve()
    figures_root = (output_root / "figures").resolve()
    try:
        figures_root.relative_to(output_root)
    except ValueError as error:
        raise ValueError("图片目录越出输出目录") from error
    referenced: set[Path] = set()
    for crop in crops:
        path = (output_root / crop.image_path).resolve()
        try:
            path.relative_to(figures_root)
        except ValueError as error:
            raise ValueError("候选图片路径越出图片目录") from error
        referenced.add(path)
    if not figures_root.exists():
        return 0
    removed = 0
    for path in figures_root.rglob("*.png"):
        resolved = path.resolve()
        if resolved not in referenced:
            path.unlink()
            removed += 1
    return removed
