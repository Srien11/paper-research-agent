"""File runner for reproducible chunk artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from paper_research_agent.chunking.chunker import build_chunks, build_paper_cards
from paper_research_agent.ingestion.models import DocumentElement, SectionRecord
from paper_research_agent.retrieval.config import load_chunking_config

RecordT = TypeVar("RecordT", bound=BaseModel)


def _read_jsonl(path: Path, model: type[RecordT]) -> list[RecordT]:
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, records: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [record.model_dump_json() for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run_chunking(
    elements_path: Path,
    sections_path: Path,
    config_path: Path,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    config = load_chunking_config(config_path)
    elements = _read_jsonl(elements_path, DocumentElement)
    sections = _read_jsonl(sections_path, SectionRecord)
    abstract_ids = {
        section.section_id
        for section in sections
        if section.title_normalized.strip().casefold() == "abstract"
    }
    chunks = build_chunks(elements, config, abstract_section_ids=abstract_ids)
    cards = build_paper_cards(elements, chunks, config, abstract_section_ids=abstract_ids)
    target = output_dir or config.output_dir
    chunks_path = target / "chunks.jsonl"
    cards_path = target / "paper_cards.jsonl"
    _write_jsonl(chunks_path, chunks)
    _write_jsonl(cards_path, cards)
    return chunks_path, cards_path
