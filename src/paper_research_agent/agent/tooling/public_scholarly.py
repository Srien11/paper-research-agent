"""Keyless official Crossref/arXiv adapters; metadata never becomes local evidence."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections.abc import Callable
from html import unescape
from typing import Any
from urllib.parse import quote

import httpx

from paper_research_agent.agent.tooling.contracts import (
    CitationGraphInput,
    IdentifierInput,
    ScholarlySearchInput,
)
from paper_research_agent.agent.tooling.scholarly import _normalize_doi
from paper_research_agent.agent.tooling.scholarly_providers import (
    SCHOLARLY_OPERATIONS,
    ScholarlyOperation,
    ScholarlyProviderInvalidResponse,
    ScholarlyProviderRateLimited,
    ScholarlyProviderRequest,
    ScholarlyProviderResult,
    ScholarlyProviderTimeout,
    ScholarlyProviderUnavailable,
)

_MAX_BYTES = 2_000_000
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"
_ARXIV_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?")


class _PublicTransport:
    """Serialize cache misses, enforce spacing and bound in-memory response retention."""

    def __init__(self, client: httpx.AsyncClient, *, interval: float):
        self._client = client
        self._interval = interval
        self._lock = asyncio.Lock()
        self._next_request = 0.0
        self._cooldown = 0.0
        self._cache: OrderedDict[
            tuple[str, tuple[tuple[str, str], ...]], tuple[float, bytes | None]
        ] = OrderedDict()

    async def get(
        self, url: str, params: dict[str, str], validate: Callable[[bytes], object]
    ) -> tuple[bytes | None, bool]:
        key = (url, tuple(sorted(params.items())))
        async with self._lock:
            now = time.monotonic()
            cached = self._cache.get(key)
            if cached is not None and now - cached[0] < 300:
                self._cache.move_to_end(key)
                return cached[1], True
            if now < self._cooldown:
                raise ScholarlyProviderRateLimited(retry_after_seconds=self._cooldown - now)
            delay = self._next_request - now
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_request = time.monotonic() + self._interval
            try:
                async with self._client.stream("GET", url, params=params) as response:
                    if response.status_code == 429:
                        try:
                            retry = float(response.headers.get("Retry-After", "60"))
                        except ValueError:
                            retry = 60.0
                        retry = (
                            min(300.0, max(self._interval, retry)) if math.isfinite(retry) else 60.0
                        )
                        self._cooldown = time.monotonic() + retry
                        raise ScholarlyProviderRateLimited(retry_after_seconds=retry)
                    if response.status_code == 404:
                        body = None
                    elif response.status_code != 200:
                        raise ScholarlyProviderUnavailable() from None
                    else:
                        chunks = bytearray()
                        async for chunk in response.aiter_bytes():
                            chunks.extend(chunk)
                            if len(chunks) > _MAX_BYTES:
                                raise ScholarlyProviderInvalidResponse() from None
                        body = bytes(chunks)
            except httpx.TimeoutException:
                raise ScholarlyProviderTimeout() from None
            except httpx.RequestError:
                raise ScholarlyProviderUnavailable() from None
            if body is not None:
                validate(body)
            self._cache[key] = (time.monotonic(), body)
            while (
                len(self._cache) > 128
                or sum(len(v[1] or b"") for v in self._cache.values()) > 8_000_000
            ):
                self._cache.popitem(last=False)
            return body, False


def _crossref_message(body: bytes, *, collection: bool) -> dict[str, Any]:
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError):
        raise ScholarlyProviderInvalidResponse() from None
    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, dict) or (collection and not isinstance(message.get("items"), list)):
        raise ScholarlyProviderInvalidResponse() from None
    return message


def _arxiv_feed(body: bytes) -> ET.Element:
    # Official feeds are UTF-8. Reject alternate encodings that could hide DTD/entity markup.
    if b"\x00" in body or b"<!DOCTYPE" in body.upper() or b"<!ENTITY" in body.upper():
        raise ScholarlyProviderInvalidResponse() from None
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        raise ScholarlyProviderInvalidResponse() from None
    if root.tag != _ATOM + "feed":
        raise ScholarlyProviderInvalidResponse() from None
    # arXiv represents invalid requests as Atom error entries, sometimes with HTTP 200.
    if any(
        not _arxiv_id(entry.findtext(_ATOM + "id", ""))
        or not _text(entry.findtext(_ATOM + "title"))
        for entry in root.findall(_ATOM + "entry")
    ):
        raise ScholarlyProviderInvalidResponse() from None
    return root


def _text(value: object, limit: int = 2000) -> str:
    return " ".join(unescape(re.sub(r"<[^>]*>", " ", str(value or ""))).split())[:limit]


def _first(value: object) -> str:
    return _text(value[0]) if isinstance(value, list) and value else ""


def _arxiv_id(value: str) -> str | None:
    normalized = value.strip()
    for prefix in (
        "arxiv:",
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "https://arxiv.org/pdf/",
        "http://arxiv.org/pdf/",
    ):
        if normalized.casefold().startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    normalized = normalized.removesuffix(".pdf")
    return normalized if _ARXIV_ID.fullmatch(normalized) else None


def _crossref_item(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    doi = _normalize_doi(str(value.get("DOI", "")))
    title = _first(value.get("title"))
    if not doi or not title:
        return None
    year: int | None = None
    for field in ("published", "published-print", "published-online", "issued"):
        date = value.get(field)
        parts = date.get("date-parts") if isinstance(date, dict) else None
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            candidate = parts[0][0]
            if isinstance(candidate, int) and 1900 <= candidate <= 2100:
                year = candidate
                break
    raw_authors = value.get("author", [])
    authors = (
        tuple(
            _text(f"{author.get('given', '')} {author.get('family', '')}", 300)
            or _text(author.get("name"), 300)
            for author in raw_authors[:100]
            if isinstance(author, dict)
        )
        if isinstance(raw_authors, list)
        else ()
    )
    return {
        "paper_id": f"doi:{doi}",
        "title": title,
        "year": year,
        "authors": authors,
        "venue": _first(value.get("container-title")),
        "abstract": _text(value.get("abstract"), 6000) or None,
        "url": f"https://doi.org/{quote(doi, safe='/')}",
        "citation_count": value.get("is-referenced-by-count"),
        "external_ids": {"DOI": doi},
        "open_access_pdf": None,
    }


class CrossrefProvider:
    provider_id = "crossref"
    capabilities = SCHOLARLY_OPERATIONS

    def __init__(self, client: httpx.AsyncClient):
        self._transport = _PublicTransport(client, interval=1.0)

    async def _message(self, doi: str | None, params: dict[str, str]) -> tuple[Any, bool]:
        url = "https://api.crossref.org/works"
        if doi is not None:
            url += "/" + quote(doi, safe="")
        body, cached = await self._transport.get(
            url, params, lambda value: _crossref_message(value, collection=doi is None)
        )
        if body is None:
            return None, cached
        return _crossref_message(body, collection=doi is None), cached

    async def execute(
        self,
        operation: ScholarlyOperation,
        request: ScholarlyProviderRequest,
    ) -> ScholarlyProviderResult:
        if isinstance(request, ScholarlySearchInput):
            params = {"query.bibliographic": request.query, "rows": str(request.limit)}
            filters = []
            if request.year_from:
                filters.append(f"from-pub-date:{request.year_from}-01-01")
            if request.year_to:
                filters.append(f"until-pub-date:{request.year_to}-12-31")
            if filters:
                params["filter"] = ",".join(filters)
            message, cached = await self._message(None, params)
            raw = message["items"] if message else []
            items = tuple(item for value in raw[: request.limit] if (item := _crossref_item(value)))
            return self._result(items, cached=cached)
        doi = _normalize_doi(request.identifier)
        if _arxiv_id(request.identifier) or request.identifier.casefold().startswith(
            ("arxiv:", "corpusid:")
        ):
            return self._result((), reason_code="identifier_not_supported")
        if operation == "resolve" and doi is None:
            result = await self.execute(
                "search", ScholarlySearchInput(query=request.identifier, limit=5)
            )
            return result.model_copy(
                update={"summary": {**result.summary, "match_type": "title_candidates"}}
            )
        if doi is None:
            return self._result((), reason_code="doi_required")
        if isinstance(request, CitationGraphInput) and request.direction == "citations":
            return self._result((), reason_code="inbound_citations_not_supported")
        message, cached = await self._message(doi, {})
        if message is None:
            return self._result((), cached=cached)
        if operation == "resolve":
            item = _crossref_item(message)
            if item is None:
                raise ScholarlyProviderInvalidResponse() from None
            return self._result((item,), cached=cached, match_type="doi")
        if operation == "status":
            # Only normalized bibliographic update fields, not the original provider payload.
            updates = message.get("update-to", [])
            normalized = (
                tuple(
                    {"doi": _text(update.get("DOI")), "type": _text(update.get("type"))}
                    for update in updates[:50]
                    if isinstance(update, dict)
                )
                if isinstance(updates, list)
                else ()
            )
            return self._result(
                (
                    {
                        "doi": doi,
                        "type": _text(message.get("type")),
                        "publisher": _text(message.get("publisher")),
                        "updates": normalized,
                        "has_update": bool(normalized),
                    },
                ),
                cached=cached,
                status_scope="deposited_update_metadata",
                retraction_verified=False,
            )
        if operation == "citations" and isinstance(request, CitationGraphInput):
            references = message.get("reference", [])
            items_list: list[dict[str, Any]] = []
            for ref in references[: request.limit] if isinstance(references, list) else []:
                if not isinstance(ref, dict):
                    continue
                ref_doi = _normalize_doi(str(ref.get("DOI", "")))
                title = _text(ref.get("article-title"))
                unstructured = _text(ref.get("unstructured"))
                if ref_doi or title or unstructured:
                    items_list.append(
                        {
                            "direction": "references",
                            "paper_id": f"doi:{ref_doi}" if ref_doi else None,
                            "title": title or None,
                            "unstructured": unstructured or None,
                            "external_ids": {"DOI": ref_doi} if ref_doi else {},
                            "url": f"https://doi.org/{quote(ref_doi, safe='/')}"
                            if ref_doi
                            else None,
                        }
                    )
            return self._result(
                tuple(items_list),
                cached=cached,
                partial=True,
                supported_directions=("references",),
                unsupported_directions=("citations",) if request.direction == "both" else (),
                coverage="deposited_references_only",
            )
        raise ValueError("unsupported Crossref operation")

    def _result(
        self, items: tuple[dict[str, Any], ...], *, cached: bool = False, **summary: Any
    ) -> ScholarlyProviderResult:
        return ScholarlyProviderResult(
            provider_id=self.provider_id,
            status="insufficient" if "reason_code" in summary else "ok" if items else "not_found",
            items=copy.deepcopy(items),
            summary={
                "source": self.provider_id,
                "returned_count": len(items),
                "cache_hit": cached,
                "network_accessed": not cached and "reason_code" not in summary,
                **summary,
            },
        )


class ArxivProvider:
    provider_id = "arxiv"
    capabilities: frozenset[ScholarlyOperation] = frozenset({"search", "resolve"})

    def __init__(self, client: httpx.AsyncClient):
        self._transport = _PublicTransport(client, interval=3.0)

    async def execute(
        self,
        operation: ScholarlyOperation,
        request: ScholarlyProviderRequest,
    ) -> ScholarlyProviderResult:
        if operation not in self.capabilities:
            raise ValueError("unsupported arXiv operation")
        identifier = _arxiv_id(request.identifier) if isinstance(request, IdentifierInput) else None
        if isinstance(request, IdentifierInput) and _normalize_doi(request.identifier):
            return ScholarlyProviderResult(
                provider_id=self.provider_id,
                status="insufficient",
                summary={"reason_code": "arxiv_id_required", "network_accessed": False},
            )
        limit = request.limit if isinstance(request, ScholarlySearchInput) else 5
        params = {"start": "0", "max_results": str(limit)}
        if identifier:
            params["id_list"] = identifier
        else:
            query = (
                request.query if isinstance(request, ScholarlySearchInput) else request.identifier
            )
            # Accept plain keywords/title, not an unbounded raw query-language expression.
            words = re.findall(r"[\w-]+", query)
            if not words:
                return ScholarlyProviderResult(
                    provider_id=self.provider_id,
                    status="not_found",
                    summary={"returned_count": 0, "network_accessed": False},
                )
            params["search_query"] = " AND ".join(f'all:"{word}"' for word in words[:50])
            if isinstance(request, ScholarlySearchInput) and (request.year_from or request.year_to):
                params["search_query"] += (
                    f" AND submittedDate:[{request.year_from or 1900}01010000 TO "
                    f"{request.year_to or 2100}12312359]"
                )
            params.update(sortBy="relevance", sortOrder="descending")
        url = "https://export.arxiv.org/api/query"
        body, cached = await self._transport.get(url, params, _arxiv_feed)
        items: list[dict[str, Any]] = []
        if body is not None:
            root = _arxiv_feed(body)
            for entry in root.findall(_ATOM + "entry")[:limit]:
                entry_id = entry.findtext(_ATOM + "id", "")
                paper_id = _arxiv_id(entry_id)
                title = _text(entry.findtext(_ATOM + "title"))
                if not paper_id or not title:
                    raise ScholarlyProviderInvalidResponse() from None
                published = entry.findtext(_ATOM + "published", "")
                year = int(published[:4]) if published[:4].isdigit() else None
                doi = _normalize_doi(entry.findtext(_ARXIV + "doi", ""))
                items.append(
                    {
                        "paper_id": f"arxiv:{paper_id}",
                        "title": title,
                        "year": year,
                        "authors": tuple(
                            _text(a.findtext(_ATOM + "name"), 300)
                            for a in entry.findall(_ATOM + "author")[:100]
                        ),
                        "venue": _text(entry.findtext(_ARXIV + "journal_ref")) or "arXiv",
                        "abstract": _text(entry.findtext(_ATOM + "summary"), 6000) or None,
                        "url": f"https://arxiv.org/abs/{paper_id}",
                        "citation_count": None,
                        "external_ids": {"ArXiv": paper_id, **({"DOI": doi} if doi else {})},
                        "open_access_pdf": {"url": f"https://arxiv.org/pdf/{paper_id}"},
                    }
                )
        return ScholarlyProviderResult(
            provider_id=self.provider_id,
            status="ok" if items else "not_found",
            items=tuple(items),
            summary={
                "source": self.provider_id,
                "returned_count": len(items),
                "cache_hit": cached,
                "network_accessed": not cached,
                "match_type": "arxiv_id" if identifier else "search_candidates",
                "year_filter_basis": "submitted_date",
            },
        )
