"""Read and validate frozen JSONL corpus manifests."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import ValidationError

from paper_research_agent.models import CorpusReport, FrozenPaper


class CorpusValidationError(ValueError):
    """Raised when a frozen corpus violates an ingestion precondition."""


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, object]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusValidationError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise CorpusValidationError(f"{path}:{line_number}: record must be an object")
            yield line_number, value


def load_frozen_papers(paths: Sequence[Path]) -> list[FrozenPaper]:
    """Load records while preserving useful file and line diagnostics."""

    papers: list[FrozenPaper] = []
    errors: list[str] = []
    for path in paths:
        if not path.is_file():
            errors.append(f"{path}: file does not exist")
            continue
        try:
            records = _read_jsonl(path)
            for line_number, record in records:
                try:
                    papers.append(FrozenPaper.model_validate(record))
                except ValidationError as exc:
                    errors.append(f"{path}:{line_number}: {exc}")
        except CorpusValidationError as exc:
            errors.append(str(exc))
    if errors:
        raise CorpusValidationError("\n".join(errors))
    return papers


def validate_corpus_files(
    paths: Sequence[Path], *, require_local_pdfs: bool = True
) -> CorpusReport:
    """Apply the ingestion gate and return a reproducible corpus summary."""

    papers = load_frozen_papers(paths)
    if not papers:
        raise CorpusValidationError("corpus contains no papers")

    errors: list[str] = []
    corpus_ids = [paper.corpus_id for paper in papers]
    canonical_keys = [paper.canonical_key for paper in papers]
    versions = {paper.corpus_version for paper in papers}

    duplicate_ids = sorted({value for value in corpus_ids if corpus_ids.count(value) > 1})
    duplicate_keys = sorted(
        {value for value in canonical_keys if canonical_keys.count(value) > 1}
    )
    if duplicate_ids:
        errors.append(f"duplicate corpus_id values: {duplicate_ids}")
    if duplicate_keys:
        errors.append(f"duplicate canonical_key values: {duplicate_keys}")
    if len(versions) != 1:
        errors.append(f"expected one corpus_version, found: {sorted(versions)}")

    local_pdf_count = sum(paper.local_pdf_path.is_file() for paper in papers)
    if require_local_pdfs and local_pdf_count != len(papers):
        missing = [
            f"{paper.corpus_id}:{paper.local_pdf_path}"
            for paper in papers
            if not paper.local_pdf_path.is_file()
        ]
        errors.append(f"missing local PDFs: {missing}")

    split_prefix_errors = [
        paper.corpus_id
        for paper in papers
        if (paper.dataset_split == "core") != paper.corpus_id.startswith("C")
    ]
    if split_prefix_errors:
        errors.append(f"corpus_id/split mismatch: {split_prefix_errors}")

    if errors:
        raise CorpusValidationError("\n".join(errors))

    return CorpusReport(
        corpus_version=next(iter(versions)),
        paper_count=len(papers),
        core_count=sum(paper.dataset_split == "core" for paper in papers),
        challenge_count=sum(paper.dataset_split == "challenge" for paper in papers),
        redistributable_count=sum(
            paper.storage_class == "redistributable" for paper in papers
        ),
        internal_research_only_count=sum(
            paper.storage_class == "internal_research_only" for paper in papers
        ),
        total_pages=sum(paper.pdf_pages for paper in papers),
        canonical_key_count=len(set(canonical_keys)),
        local_pdf_count=local_pdf_count,
    )

