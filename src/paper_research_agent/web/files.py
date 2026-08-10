"""Bounded, session-isolated attachment storage for the private Web UI."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

ALLOWED_SUFFIXES = frozenset({".pdf", ".txt", ".md", ".csv", ".json"})
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_CONTEXT_CHARS = 60_000


@dataclass(frozen=True)
class Attachment:
    attachment_id: str
    filename: str
    content_type: str
    size_bytes: int


@dataclass(frozen=True)
class AttachmentContent:
    filename: str
    content_type: str
    data: bytes


class AttachmentStore:
    def __init__(self, root: Path, *, max_file_bytes: int = MAX_FILE_BYTES) -> None:
        self._root = root.resolve()
        self._max_file_bytes = max_file_bytes

    async def save(
        self,
        *,
        session_id: str,
        filename: str,
        content_type: str,
        chunks: object,
    ) -> Attachment:
        safe_name = _safe_filename(filename)
        suffix = Path(safe_name).suffix.casefold()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError("unsupported attachment type")
        attachment_id = uuid.uuid4().hex
        directory = self._session_directory(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{attachment_id}{suffix}"
        size = 0
        try:
            with target.open("xb") as handle:
                async for chunk in chunks:  # type: ignore[attr-defined]
                    size += len(chunk)
                    if size > self._max_file_bytes:
                        raise ValueError("attachment exceeds the size limit")
                    handle.write(chunk)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        if size == 0:
            target.unlink(missing_ok=True)
            raise ValueError("attachment cannot be empty")
        metadata = {
            "attachment_id": attachment_id,
            "filename": safe_name,
            "content_type": content_type[:120],
            "size_bytes": size,
            "path": target.name,
        }
        (directory / f"{attachment_id}.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )
        return Attachment(
            attachment_id=attachment_id,
            filename=safe_name,
            content_type=str(metadata["content_type"]),
            size_bytes=size,
        )

    async def save_generated_text(
        self,
        *,
        session_id: str,
        filename: str,
        text: str,
    ) -> Attachment:
        """Persist a generated text artifact as a new, session-scoped attachment."""
        normalized = text.strip()
        if not normalized:
            raise ValueError("generated attachment cannot be empty")
        suffix = Path(filename).suffix.casefold()
        content_types = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".csv": "text/csv",
            ".json": "application/json",
        }
        content_type = content_types.get(suffix)
        if content_type is None:
            raise ValueError("generated attachment type must be textual")

        async def chunks() -> AsyncIterator[bytes]:
            yield normalized.encode("utf-8")

        return await self.save(
            session_id=session_id,
            filename=filename,
            content_type=content_type,
            chunks=chunks(),
        )

    def extract(self, session_id: str, attachment_ids: tuple[str, ...]) -> tuple[str, ...]:
        texts: list[str] = []
        remaining = MAX_CONTEXT_CHARS
        for attachment_id in attachment_ids[:5]:
            record, path = self._resolve(session_id, attachment_id)
            text = _extract_text(path)
            clipped = text[:remaining]
            texts.append(f"附件：{record['filename']}\n{clipped}")
            remaining -= len(clipped)
            if remaining <= 0:
                break
        return tuple(texts)

    def delete(self, session_id: str, attachment_id: str) -> bool:
        try:
            _record, path = self._resolve(session_id, attachment_id)
        except FileNotFoundError:
            return False
        path.unlink(missing_ok=True)
        (path.parent / f"{attachment_id}.json").unlink(missing_ok=True)
        return True

    def validate_ownership(
        self, session_id: str, attachment_ids: tuple[str, ...]
    ) -> None:
        """Validate session ownership without extracting or returning file contents."""
        for attachment_id in attachment_ids:
            self._resolve(session_id, attachment_id)

    def read(self, session_id: str, attachment_id: str) -> AttachmentContent:
        record, path = self._resolve(session_id, attachment_id)
        return AttachmentContent(
            filename=str(record["filename"]),
            content_type=str(record["content_type"]),
            data=path.read_bytes(),
        )

    def _resolve(self, session_id: str, attachment_id: str) -> tuple[dict[str, object], Path]:
        if not re.fullmatch(r"[0-9a-f]{32}", attachment_id):
            raise FileNotFoundError("attachment not found")
        directory = self._session_directory(session_id)
        metadata_path = directory / f"{attachment_id}.json"
        if not metadata_path.is_file():
            raise FileNotFoundError("attachment not found")
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        path = (directory / str(record["path"])).resolve()
        if path.parent != directory or not path.is_file():
            raise FileNotFoundError("attachment not found")
        return record, path

    def _session_directory(self, session_id: str) -> Path:
        fingerprint = hashlib.sha256(session_id.encode()).hexdigest()[:32]
        return (self._root / fingerprint).resolve()


def _safe_filename(value: str) -> str:
    name = Path(value.strip()).name
    if not name or name != value.strip() or len(name) > 180 or any(char in name for char in "\x00\r\n"):
        raise ValueError("invalid attachment filename")
    return name


def _extract_text(path: Path) -> str:
    if path.suffix.casefold() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("PDF reading dependency is unavailable") from error
        return "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="replace")
