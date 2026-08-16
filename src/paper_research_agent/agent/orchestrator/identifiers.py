"""Deterministic internal identifiers shared by execution and cleanup paths."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from paper_research_agent.memory.models import SESSION_ID_PATTERN

CHILD_SESSION_ID_MAX_LENGTH = 128
DYNAMIC_THREAD_ID_MAX_LENGTH = 240


def child_session_id(
    kind: str,
    conversation_id: str,
    run_id: str,
    task_id: str,
) -> str:
    """Keep legacy-safe child session IDs and hash only invalid composites."""

    parts = (kind, conversation_id, run_id, task_id)
    raw = "::".join(parts)
    return _bounded_identifier(
        raw,
        scope=kind,
        parts=parts,
        max_length=CHILD_SESSION_ID_MAX_LENGTH,
        require_safe_session_chars=True,
    )


def dynamic_thread_id(conversation_id: str, run_id: str, task_id: str) -> str:
    """Return the legacy dynamic thread ID unless its composed length is unsafe."""

    parts = (conversation_id, run_id, task_id)
    raw = "::".join(parts)
    return _bounded_identifier(
        raw,
        scope="dynamic",
        parts=parts,
        max_length=DYNAMIC_THREAD_ID_MAX_LENGTH,
        require_safe_session_chars=False,
    )


def main_checkpoint_thread_id(conversation_id: str, run_id: str) -> str:
    """Build the main checkpoint key in one place for invoke, resume, and cleanup."""

    return f"main::{conversation_id}::{run_id}"


def dynamic_checkpoint_thread_id(thread_id: str) -> str:
    """Build the checkpoint namespace used by the dynamic graph."""

    return f"dynamic::{thread_id}"


def _bounded_identifier(
    raw: str,
    *,
    scope: str,
    parts: Sequence[str],
    max_length: int,
    require_safe_session_chars: bool,
) -> str:
    if len(raw) <= max_length and (
        not require_safe_session_chars or SESSION_ID_PATTERN.fullmatch(raw) is not None
    ):
        return raw
    digest = hashlib.sha256(_canonical_payload(scope, parts)).hexdigest()
    fallback = f"{scope}:h1:{digest}"
    if len(fallback) > max_length:
        raise ValueError("identifier scope is too long for the configured boundary")
    if require_safe_session_chars and SESSION_ID_PATTERN.fullmatch(fallback) is None:
        raise ValueError("identifier scope contains unsafe session characters")
    return fallback


def _canonical_payload(scope: str, parts: Sequence[str]) -> bytes:
    payload = {"scope": scope, "parts": list(parts)}
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
