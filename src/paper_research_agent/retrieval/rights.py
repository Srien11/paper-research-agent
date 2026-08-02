"""Read-only corpus redistribution rights used at online output boundaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeAlias

from paper_research_agent.corpus import load_frozen_papers
from paper_research_agent.retrieval.contracts import SearchHit

StorageClass: TypeAlias = Literal["redistributable", "internal_research_only"]


class CorpusRightsMap:
    def __init__(self, storage_classes: Mapping[str, StorageClass]):
        self._storage_classes = dict(storage_classes)

    @classmethod
    def from_manifest_paths(cls, paths: Sequence[Path]) -> CorpusRightsMap:
        papers = load_frozen_papers(paths)
        return cls({paper.corpus_id: paper.storage_class for paper in papers})

    @classmethod
    def from_corpus_dir(cls, corpus_dir: Path) -> CorpusRightsMap:
        return cls.from_manifest_paths(
            [corpus_dir / "core_frozen.jsonl", corpus_dir / "challenge_frozen.jsonl"]
        )

    def for_hits(self, hits: Iterable[SearchHit]) -> dict[str, StorageClass]:
        result: dict[str, StorageClass] = {}
        missing: set[str] = set()
        for hit in hits:
            storage_class = self._storage_classes.get(hit.corpus_id)
            if storage_class is None:
                missing.add(hit.corpus_id)
            else:
                result[hit.corpus_id] = storage_class
        if missing:
            raise ValueError(f"missing storage rights for corpus IDs: {sorted(missing)}")
        return result
