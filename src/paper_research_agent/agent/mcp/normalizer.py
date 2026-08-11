"""Deterministic low-trust normalization for MCP tool results."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from paper_research_agent.agent.tooling.catalog import ToolTrust
from paper_research_agent.agent.tooling.contracts import ToolExecutionResult

_SENSITIVE_FIELDS = {
    "api_key",
    "apikey",
    "approval_token",
    "authorization",
    "password",
    "secret",
    "token",
}


def _field_is_sensitive(name: str) -> bool:
    normalized = name.casefold().replace("-", "_")
    return normalized in _SENSITIVE_FIELDS or normalized.endswith("_token")


def _sanitize(value: Any, *, max_string_chars: int) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize(item, max_string_chars=max_string_chars)
            for key, item in value.items()
            if not _field_is_sensitive(str(key))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize(item, max_string_chars=max_string_chars) for item in value]
    if isinstance(value, str):
        return value[:max_string_chars]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:max_string_chars]


def _raw_field(raw: Any, name: str, default: Any = None) -> Any:
    if isinstance(raw, Mapping):
        return raw.get(name, default)
    return getattr(raw, name, default)


def _insufficient(tool_name: str, trust: ToolTrust, reason_code: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_name=tool_name,
        status="insufficient",
        trust=trust,
        summary={"reason_code": reason_code},
    )


def _as_item(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {"value": value}


def normalize_mcp_result(
    raw: Any,
    *,
    tool_name: str,
    trust: ToolTrust,
    max_result_items: int,
    max_output_bytes: int,
) -> ToolExecutionResult:
    if bool(_raw_field(raw, "isError", False)):
        return _insufficient(tool_name, trust, "mcp_server_error")

    structured = _raw_field(raw, "structuredContent")
    raw_items: list[Any]
    raw_summary: dict[str, Any] = {}
    if structured is not None:
        if isinstance(structured, Mapping):
            candidate_items = structured.get("items")
            candidate_summary = structured.get("summary")
            if isinstance(candidate_items, Sequence) and not isinstance(
                candidate_items, (str, bytes, bytearray)
            ):
                raw_items = list(candidate_items)
            else:
                payload = {
                    str(key): value
                    for key, value in structured.items()
                    if key not in {"items", "summary"}
                }
                raw_items = [payload] if payload else []
            if isinstance(candidate_summary, Mapping):
                raw_summary = {str(key): value for key, value in candidate_summary.items()}
        elif isinstance(structured, Sequence) and not isinstance(
            structured, (str, bytes, bytearray)
        ):
            raw_items = list(structured)
        else:
            raw_items = [structured]
    else:
        content = tuple(_raw_field(raw, "content", ()) or ())
        if any(_raw_field(item, "type") != "text" for item in content):
            return _insufficient(tool_name, trust, "mcp_content_type_rejected")
        raw_items = [{"text": str(_raw_field(item, "text", ""))} for item in content]

    truncated = len(raw_items) > max_result_items
    bounded_raw_items = raw_items[:max_result_items]
    max_string_chars = max(64, min(10_000, max_output_bytes // 4))
    items = tuple(
        _as_item(_sanitize(item, max_string_chars=max_string_chars))
        for item in bounded_raw_items
    )
    summary = _sanitize(raw_summary, max_string_chars=max_string_chars)
    if not isinstance(summary, dict):
        summary = {}
    if any(
        isinstance(value, str) and len(value) > max_string_chars
        for item in bounded_raw_items
        for value in _walk_values(item)
    ):
        truncated = True
    if truncated:
        summary["truncated"] = True

    result = ToolExecutionResult(
        tool_name=tool_name,
        trust=trust,
        items=items,
        summary=summary,
    )
    while len(result.model_dump_json().encode("utf-8")) > max_output_bytes and result.items:
        truncated = True
        result = result.model_copy(
            update={
                "items": result.items[:-1],
                "summary": {**result.summary, "truncated": True},
            }
        )
    if len(result.model_dump_json().encode("utf-8")) > max_output_bytes:
        result = result.model_copy(update={"items": (), "summary": {"truncated": True}})
    return result


def _walk_values(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return [nested for item in value.values() for nested in _walk_values(item)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [nested for item in value for nested in _walk_values(item)]
    return [value]


def normalized_result_size(result: ToolExecutionResult) -> int:
    """Return the canonical UTF-8 size used by release diagnostics."""
    return len(json.dumps(result.model_dump(mode="json"), ensure_ascii=False).encode("utf-8"))
