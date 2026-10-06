"""Request-local preview callbacks and a bounded projection of partial JSON strings."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

PreviewCallback = Callable[[str], Awaitable[None]]
answer_preview: ContextVar[PreviewCallback | None] = ContextVar("answer_preview", default=None)


@contextmanager
def preview_scope(callback: PreviewCallback | None) -> Iterator[None]:
    token = answer_preview.set(callback)
    try:
        yield
    finally:
        answer_preview.reset(token)


def project_preview(content: str, *, synthesis: bool = False) -> str:
    """Expose only answer text, never JSON keys, citation metadata or thinking tokens."""
    if len(content) > 262_144:
        return ""
    try:
        value, _, _ = _value(content, 0, 0)
    except (ValueError, RecursionError):
        return ""
    if not isinstance(value, dict):
        return ""
    if synthesis:
        return (
            str(value.get("conclusion", ""))[:20_000]
            if isinstance(value.get("conclusion"), str)
            else ""
        )
    if value.get("status") == "insufficient_evidence":
        return ""
    claims = value.get("claims")
    if not isinstance(claims, list):
        return ""
    return "\n\n".join(
        claim["text"]
        for claim in claims
        if isinstance(claim, dict) and isinstance(claim.get("text"), str)
    )[:20_000]


def _value(text: str, pos: int, depth: int) -> tuple[Any, int, bool]:
    if depth > 32:
        raise ValueError("JSON depth exceeded")
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos == len(text):
        return None, pos, False
    ch = text[pos]
    if ch == '"':
        start = pos
        pos += 1
        escaped = False
        while pos < len(text):
            c = text[pos]
            if c == '"' and not escaped:
                return json.loads(text[start : pos + 1]).encode("utf-8", "ignore").decode("utf-8"), pos + 1, True
            if c == "\\" and not escaped:
                escaped = True
            else:
                escaped = False
            pos += 1
        # Incomplete escapes (including unicode escapes) are withheld, never printed.
        partial = text[start:]
        for drop in range(min(12, len(partial))):
            candidate = partial if drop == 0 else partial[:-drop]
            try:
                decoded = json.loads(candidate + '"')
                # A split UTF-16 pair must not leave an invalid surrogate on the wire.
                decoded = decoded.encode("utf-8", "ignore").decode("utf-8")
                return decoded, len(text), False
            except ValueError:
                pass
        return "", len(text), False
    if ch in "[{":
        mapping = ch == "{"
        result: Any = {} if mapping else []
        close = "}" if mapping else "]"
        pos += 1
        while True:
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos == len(text):
                return result, pos, False
            if text[pos] == close:
                return result, pos + 1, True
            key = None
            if mapping:
                key, pos, complete = _value(text, pos, depth + 1)
                if not complete:
                    return result, pos, False
                if not isinstance(key, str) or key in result:
                    raise ValueError("invalid or duplicate JSON key")
                while pos < len(text) and text[pos].isspace():
                    pos += 1
                if pos == len(text):
                    return result, pos, False
                if text[pos] != ":":
                    raise ValueError("expected colon")
                pos += 1
            item, pos, complete = _value(text, pos, depth + 1)
            if mapping:
                result[key] = item
            else:
                result.append(item)
            if not complete:
                return result, pos, False
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos == len(text):
                return result, pos, False
            if text[pos] == close:
                return result, pos + 1, True
            if text[pos] != ",":
                raise ValueError("expected comma")
            pos += 1
    try:
        value, end = json.JSONDecoder().raw_decode(text, pos)
        return value, end, True
    except ValueError:
        return None, len(text), False
