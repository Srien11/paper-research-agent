"""运行全语料图片裁剪并生成本地任务清单。"""

from __future__ import annotations

import json
from pathlib import Path

from paper_research_agent.corpus import load_frozen_papers
from paper_research_agent.figures.cropper import FigureCrop, crop_pdf_figures
from paper_research_agent.ingestion.models import DocumentElement


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

    grouped: dict[str, list[dict[str, object]]] = {}
    for caption in captions:
        grouped.setdefault(caption.corpus_id, []).append(caption.model_dump(mode="json"))

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
    return manifest_path, crops
