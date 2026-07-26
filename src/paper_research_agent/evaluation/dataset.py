"""Validated silver diagnostic dataset contract."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DiagnosticQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(pattern=r"^Q\d{3}$")
    query: str = Field(min_length=1)
    relevant_paper_ids: tuple[str, ...] = Field(min_length=1)
    relevant_chunk_ids: tuple[str, ...] = ()
    answerable: bool
    annotation_status: Literal["silver_single_reviewer"]
    reviewer_count: Literal[1] = 1
    notes: str | None = None

    @model_validator(mode="after")
    def validate_unique_labels(self) -> DiagnosticQuery:
        if len(set(self.relevant_paper_ids)) != len(self.relevant_paper_ids):
            raise ValueError("relevant_paper_ids must be unique")
        if any(not __import__("re").fullmatch(r"[CT]\d{3}", value) for value in self.relevant_paper_ids):
            raise ValueError("invalid corpus paper ID")
        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError("relevant_chunk_ids must be unique")
        return self


def load_dataset(path: Path) -> list[DiagnosticQuery]:
    queries = [
        DiagnosticQuery.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identifiers = [query.query_id for query in queries]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("query_id values must be unique")
    return queries
