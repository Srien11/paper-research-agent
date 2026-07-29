"""生成稳定的图片标识。"""

from __future__ import annotations

from paper_research_agent.ingestion.identity import stable_id


def make_figure_id(
    asset_id: str,
    page_number: int,
    figure_name: str,
    bbox: tuple[float, float, float, float],
) -> str:
    if page_number < 1:
        raise ValueError("页码必须从 1 开始")
    normalized_bbox = ",".join(f"{value:.3f}" for value in bbox)
    return stable_id(
        "figure",
        asset_id,
        page_number,
        figure_name.strip().casefold(),
        normalized_bbox,
    )
