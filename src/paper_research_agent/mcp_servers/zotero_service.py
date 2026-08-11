"""Bounded read-only projection of Zotero's loopback Local API."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

import httpx

_ITEM_KEY = re.compile(r"^[A-Z0-9]{8}$")
_MAX_RESPONSE_BYTES = 1_048_576
_BASE_URL = "http://127.0.0.1:23119/api/"


class ZoteroServiceError(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


def _validate_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port != 23119
        or parsed.path.rstrip("/") != "/api"
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Zotero base URL must use the fixed loopback API endpoint")
    return value.rstrip("/") + "/"


def _bounded_text(value: Any, limit: int) -> str:
    return str(value or "")[:limit]


def _item_key(value: str) -> str:
    if _ITEM_KEY.fullmatch(value) is None:
        raise ValueError("Zotero item key must contain exactly eight uppercase letters or digits")
    return value


def _bounded_limit(value: int) -> int:
    if isinstance(value, bool) or value < 1 or value > 20:
        raise ValueError("Zotero result limit must be between 1 and 20")
    return value


class ZoteroLocalService:
    def __init__(
        self,
        *,
        base_url: str = _BASE_URL,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        validated_url = _validate_base_url(base_url)
        if client is not None:
            _validate_base_url(str(client.base_url))
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=validated_url,
            timeout=httpx.Timeout(5),
            follow_redirects=False,
            headers={"Zotero-API-Version": "3"},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def search_items(self, *, query: str, limit: int = 10) -> list[dict[str, Any]]:
        normalized = query.strip()
        if not normalized or len(normalized) > 500:
            raise ValueError("Zotero search query must contain between 1 and 500 characters")
        payload = await self._get_json(
            "users/0/items",
            params={"q": normalized, "qmode": "everything", "limit": _bounded_limit(limit)},
        )
        return [_project_item(item) for item in _object_list(payload)]

    async def get_item(self, *, item_key: str) -> dict[str, Any]:
        payload = await self._get_json(f"users/0/items/{_item_key(item_key)}")
        return _project_item(_object(payload))

    async def list_collections(self, *, limit: int = 20) -> list[dict[str, Any]]:
        payload = await self._get_json(
            "users/0/collections", params={"limit": _bounded_limit(limit)}
        )
        return [_project_collection(item) for item in _object_list(payload)]

    async def get_annotations(
        self, *, item_key: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        key = _item_key(item_key)
        payload = await self._get_json(
            f"users/0/items/{key}/children",
            params={"itemType": "annotation", "limit": _bounded_limit(limit)},
        )
        return [
            _project_annotation(item)
            for item in _object_list(payload)
            if _data(item).get("itemType") == "annotation"
        ]

    async def get_attachment_metadata(self, *, item_key: str) -> dict[str, Any]:
        payload = _object(await self._get_json(f"users/0/items/{_item_key(item_key)}"))
        data = _data(payload)
        return {
            "item_key": _bounded_text(payload.get("key") or data.get("key"), 8),
            "item_type": _bounded_text(data.get("itemType"), 50),
            "content_type": _bounded_text(data.get("contentType"), 200),
            "parent_key": _bounded_text(data.get("parentItem"), 8),
            "link_mode": _bounded_text(data.get("linkMode"), 50),
            "page_count": _safe_int(data.get("numPages")),
        }

    async def get_fulltext(self, *, item_key: str) -> dict[str, Any]:
        payload = _object(
            await self._get_json(f"users/0/items/{_item_key(item_key)}/fulltext")
        )
        content = str(payload.get("content") or "")
        return {
            "item_key": item_key,
            "content": content[:20_000],
            "truncated": len(content) > 20_000,
            "indexed_pages": _safe_int(payload.get("indexedPages")),
            "total_pages": _safe_int(payload.get("totalPages")),
            "indexed_chars": _safe_int(payload.get("indexedChars")),
            "total_chars": _safe_int(payload.get("totalChars")),
        }

    async def _get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        try:
            response = await self._client.get(
                path,
                params=params,
                headers={"Zotero-API-Version": "3"},
            )
        except httpx.TimeoutException:
            raise ZoteroServiceError("zotero_timeout") from None
        except httpx.RequestError:
            raise ZoteroServiceError("zotero_offline") from None
        if response.is_redirect:
            raise ZoteroServiceError("zotero_redirect_rejected")
        if response.status_code == 403:
            raise ZoteroServiceError("zotero_local_api_disabled")
        if response.status_code == 404:
            raise ZoteroServiceError("zotero_not_found")
        if response.status_code != 200:
            raise ZoteroServiceError("zotero_http_error")
        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > _MAX_RESPONSE_BYTES:
                    raise ZoteroServiceError("zotero_response_too_large")
            except ValueError:
                raise ZoteroServiceError("zotero_invalid_response") from None
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise ZoteroServiceError("zotero_response_too_large")
        if "json" not in response.headers.get("content-type", "").casefold():
            raise ZoteroServiceError("zotero_invalid_response")
        try:
            return response.json()
        except ValueError:
            raise ZoteroServiceError("zotero_invalid_response") from None


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ZoteroServiceError("zotero_invalid_response")
    return value


def _object_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ZoteroServiceError("zotero_invalid_response")
    return value


def _data(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("data")
    return value if isinstance(value, dict) else item


def _project_item(item: dict[str, Any]) -> dict[str, Any]:
    data = _data(item)
    date = _bounded_text(data.get("date"), 100)
    year_match = re.search(r"\b(1[89]\d{2}|20\d{2}|2100)\b", date)
    raw_creators = data.get("creators")
    creators: list[Any] = raw_creators if isinstance(raw_creators, list) else []
    raw_tags = data.get("tags")
    tags: list[Any] = raw_tags if isinstance(raw_tags, list) else []
    return {
        "item_key": _bounded_text(item.get("key") or data.get("key"), 8),
        "item_type": _bounded_text(data.get("itemType"), 50),
        "title": _bounded_text(data.get("title"), 500),
        "creators": [
            {
                "creator_type": _bounded_text(creator.get("creatorType"), 50),
                "first_name": _bounded_text(creator.get("firstName"), 100),
                "last_name": _bounded_text(creator.get("lastName"), 100),
                "name": _bounded_text(creator.get("name"), 200),
            }
            for creator in creators[:20]
            if isinstance(creator, dict)
        ],
        "year": int(year_match.group(1)) if year_match else None,
        "doi": _bounded_text(data.get("DOI"), 300),
        "url": _bounded_text(data.get("url"), 2_000),
        "parent_key": _bounded_text(data.get("parentItem"), 8),
        "tags": [
            _bounded_text(tag.get("tag"), 200)
            for tag in tags[:50]
            if isinstance(tag, dict) and tag.get("tag")
        ],
    }


def _project_collection(item: dict[str, Any]) -> dict[str, Any]:
    data = _data(item)
    result = {
        "collection_key": _bounded_text(item.get("key") or data.get("key"), 8),
        "name": _bounded_text(data.get("name"), 500),
    }
    parent = data.get("parentCollection")
    if isinstance(parent, str) and parent:
        result["parent_collection_key"] = _bounded_text(parent, 8)
    return result


def _project_annotation(item: dict[str, Any]) -> dict[str, Any]:
    data = _data(item)
    return {
        "annotation_key": _bounded_text(item.get("key") or data.get("key"), 8),
        "parent_key": _bounded_text(data.get("parentItem"), 8),
        "annotation_type": _bounded_text(data.get("annotationType"), 50),
        "annotation_text": _bounded_text(data.get("annotationText"), 2_000),
        "annotation_comment": _bounded_text(data.get("annotationComment"), 2_000),
        "page_label": _bounded_text(data.get("annotationPageLabel"), 50),
    }


def _safe_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
