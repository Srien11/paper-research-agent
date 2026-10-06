"""Minimal, backend-owned bibliographic provenance, never a provider payload."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Literal
from urllib.parse import quote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

ScholarlySource = Literal["crossref", "arxiv", "semantic_scholar"]
ScholarlyToolName = Literal[
    "search_scholarly_sources",
    "resolve_paper_identifier",
    "get_citation_graph",
    "check_paper_status",
]


class ScholarlySourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^X[0-9a-f]{16}$")
    title: str = Field(min_length=1, max_length=2000)
    identifier: str = Field(min_length=1, max_length=500)
    url: str = Field(max_length=1000)

    @field_validator("url")
    @classmethod
    def official_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.hostname
            not in {"doi.org", "arxiv.org", "www.semanticscholar.org", "api.semanticscholar.org"}
            or parsed.username
            or parsed.password
            or parsed.port not in {None, 443}
        ):
            raise ValueError("scholarly source requires an official HTTPS URL")
        return value


class ScholarlyLookupRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: ScholarlySource
    tool_name: ScholarlyToolName
    queried_at: datetime
    status: Literal["ok", "not_found", "insufficient"]
    cache_hit: bool = False
    network_accessed: bool
    returned_count: int = Field(ge=0, le=50)
    sources: tuple[ScholarlySourceRecord, ...] = Field(default=(), max_length=50)

    @field_validator("queried_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("query timestamp requires a timezone")
        return value


def source_record(item: dict[str, object]) -> ScholarlySourceRecord | None:
    external = item.get("external_ids")
    ids = external if isinstance(external, dict) else {}
    doi = ids.get("DOI") or item.get("doi")
    arxiv = ids.get("ArXiv")
    url: str | None = None
    identifier: str | None = None
    if isinstance(arxiv, str) and re.fullmatch(
        r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", arxiv
    ):
        identifier = f"arxiv:{arxiv}"
        url = f"https://arxiv.org/abs/{arxiv}"
    elif isinstance(doi, str) and re.fullmatch(r"10\.\d+/[^\s<>]{1,450}", doi):
        identifier = f"doi:{doi}"
        url = f"https://doi.org/{quote(doi, safe='/')}"
    else:
        paper_id = item.get("paper_id")
        if isinstance(paper_id, str) and re.fullmatch(r"[0-9a-f]{40}", paper_id):
            identifier = f"semantic_scholar:{paper_id}"
            url = f"https://www.semanticscholar.org/paper/{paper_id}"
    if identifier is None or url is None:
        return None
    title = item.get("title")
    return ScholarlySourceRecord(
        source_id="X" + hashlib.sha256(identifier.casefold().encode()).hexdigest()[:16],
        title=title[:2000] if isinstance(title, str) and title.strip() else identifier,
        identifier=identifier,
        url=url,
    )


def bounded_lookups(records: Sequence[ScholarlyLookupRecord]) -> tuple[ScholarlyLookupRecord, ...]:
    """Fit source cards within existing durable event limits without dropping call status."""
    projected: list[ScholarlyLookupRecord] = []
    remaining = 12_000
    for record in records[-20:]:
        minimal = record.model_copy(update={"sources": ()})
        base_size = len(minimal.model_dump_json()) + 1
        if base_size > remaining:
            break
        remaining -= base_size
        sources: list[ScholarlySourceRecord] = []
        for source in record.sources:
            candidate = source.model_copy(update={"title": source.title[:300]})
            size = len(candidate.model_dump_json()) + 1
            if size > remaining:
                break
            remaining -= size
            sources.append(candidate)
        projected.append(record.model_copy(update={"sources": tuple(sources)}))
    return tuple(projected)
