"""Bounded scholarly metadata tools using official Semantic Scholar and Crossref APIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

import httpx

from paper_research_agent.agent.tooling.contracts import (
    CitationGraphInput,
    IdentifierInput,
    ScholarlySearchInput,
    ToolExecutionResult,
)

S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
CROSSREF_BASE_URL = "https://api.crossref.org"
PAPER_FIELDS = (
    "paperId,externalIds,title,authors,abstract,year,venue,url,citationCount,openAccessPdf"
)


class ScholarlyResearchTools:
    def __init__(self, client: httpx.AsyncClient, *, api_key: str | None = None):
        self._client = client
        self._headers = {"x-api-key": api_key} if api_key else {}

    async def search_scholarly_sources(self, request: ScholarlySearchInput) -> ToolExecutionResult:
        params: dict[str, object] = {
            "query": request.query,
            "limit": request.limit,
            "fields": PAPER_FIELDS,
        }
        if request.year_from or request.year_to:
            params["year"] = f"{request.year_from or ''}-{request.year_to or ''}"
        payload = await self._get_json(
            f"{S2_BASE_URL}/paper/search",
            params=params,
            headers=self._headers,
        )
        data = payload.get("data", ())
        papers = data if isinstance(data, Sequence) and not isinstance(data, str) else ()
        items = tuple(
            item for paper in papers[: request.limit] if (item := _paper_item(paper)) is not None
        )
        return ToolExecutionResult(
            tool_name="search_scholarly_sources",
            status="ok" if items else "not_found",
            items=items,
            summary={"returned_count": len(items), "provider": "semantic_scholar"},
        )

    async def resolve_paper_identifier(self, request: IdentifierInput) -> ToolExecutionResult:
        identifier = request.identifier.strip()
        if _looks_like_identifier(identifier):
            payload = await self._get_json(
                f"{S2_BASE_URL}/paper/{quote(identifier, safe='')}",
                params={"fields": PAPER_FIELDS},
                headers=self._headers,
            )
            papers: Sequence[object] = (payload,)
        else:
            payload = await self._get_json(
                f"{S2_BASE_URL}/paper/search",
                params={"query": identifier, "limit": 5, "fields": PAPER_FIELDS},
                headers=self._headers,
            )
            raw = payload.get("data", ())
            papers = raw if isinstance(raw, Sequence) and not isinstance(raw, str) else ()
        items = tuple(item for paper in papers if (item := _paper_item(paper)) is not None)
        return ToolExecutionResult(
            tool_name="resolve_paper_identifier",
            status="ok" if items else "not_found",
            items=items[:5],
            summary={"returned_count": min(5, len(items)), "provider": "semantic_scholar"},
        )

    async def get_citation_graph(self, request: CitationGraphInput) -> ToolExecutionResult:
        items: list[dict[str, Any]] = []
        directions = (
            ("references", "citedPaper"),
            ("citations", "citingPaper"),
        )
        for direction, paper_key in directions:
            if request.direction not in {direction, "both"}:
                continue
            payload = await self._get_json(
                f"{S2_BASE_URL}/paper/{quote(request.identifier, safe='')}/{direction}",
                params={"limit": request.limit, "fields": "title,authors,year,externalIds,url"},
                headers=self._headers,
            )
            raw = payload.get("data", ())
            edges = raw if isinstance(raw, Sequence) and not isinstance(raw, str) else ()
            for edge in edges[: request.limit]:
                if not isinstance(edge, Mapping):
                    continue
                paper = _paper_item(edge.get(paper_key))
                if paper is not None:
                    items.append({"direction": direction, **paper})
        return ToolExecutionResult(
            tool_name="get_citation_graph",
            status="ok" if items else "not_found",
            items=tuple(items[:50]),
            summary={"returned_count": min(50, len(items)), "provider": "semantic_scholar"},
        )

    async def check_paper_status(self, request: IdentifierInput) -> ToolExecutionResult:
        doi = _normalize_doi(request.identifier)
        if doi is None:
            resolved = await self.resolve_paper_identifier(request)
            doi = _first_doi(resolved.items)
        if doi is None:
            return ToolExecutionResult(
                tool_name="check_paper_status",
                status="not_found",
                summary={"provider": "crossref"},
            )
        payload = await self._get_json(f"{CROSSREF_BASE_URL}/works/{quote(doi, safe='')}")
        message = payload.get("message")
        if not isinstance(message, Mapping):
            return ToolExecutionResult(tool_name="check_paper_status", status="not_found")
        updates = message.get("update-to")
        relations = message.get("relation")
        item = {
            "doi": str(message.get("DOI") or doi),
            "type": message.get("type"),
            "publisher": message.get("publisher"),
            "indexed": message.get("indexed"),
            "updates": tuple(updates) if isinstance(updates, Sequence) else (),
            "relations": relations if isinstance(relations, Mapping) else {},
            "has_update": bool(updates),
        }
        return ToolExecutionResult(
            tool_name="check_paper_status",
            items=(item,),
            summary={"provider": "crossref", "has_update": bool(updates)},
        )

    async def _get_json(self, url: str, **kwargs: Any) -> Mapping[str, Any]:
        response = await self._client.get(url, **kwargs)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise TypeError("scholarly provider returned a non-object response")
        return payload


def _paper_item(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    authors = value.get("authors")
    author_names = (
        tuple(
            str(author.get("name"))
            for author in authors
            if isinstance(author, Mapping) and author.get("name")
        )
        if isinstance(authors, Sequence) and not isinstance(authors, str)
        else ()
    )
    external = value.get("externalIds")
    return {
        "paper_id": value.get("paperId"),
        "title": value.get("title"),
        "year": value.get("year"),
        "authors": author_names,
        "venue": value.get("venue"),
        "abstract": value.get("abstract"),
        "url": value.get("url"),
        "citation_count": value.get("citationCount"),
        "external_ids": dict(external) if isinstance(external, Mapping) else {},
        "open_access_pdf": value.get("openAccessPdf"),
    }


def _looks_like_identifier(value: str) -> bool:
    lowered = value.casefold()
    return lowered.startswith(("doi:", "arxiv:", "corpusid:", "10."))


def _normalize_doi(value: str) -> str | None:
    normalized = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.casefold().startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized if normalized.startswith("10.") and "/" in normalized else None


def _first_doi(items: Sequence[Mapping[str, Any]]) -> str | None:
    for item in items:
        external = item.get("external_ids")
        if isinstance(external, Mapping) and external.get("DOI"):
            return str(external["DOI"])
    return None
