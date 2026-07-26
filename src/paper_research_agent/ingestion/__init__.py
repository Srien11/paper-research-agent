"""论文解析数据契约。"""

from paper_research_agent.ingestion.models import (
    DocumentAsset,
    DocumentElement,
    IngestionManifest,
    PageRecord,
    SectionRecord,
)
from paper_research_agent.ingestion.text import normalize_text

__all__ = [
    "DocumentAsset",
    "DocumentElement",
    "IngestionManifest",
    "PageRecord",
    "SectionRecord",
    "normalize_text",
]
