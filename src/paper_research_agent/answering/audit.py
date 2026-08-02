"""Minimal answer audit storage that never persists evidence or answer bodies."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from paper_research_agent.answering.models import RAGAnswer


class SQLiteAnswerAuditLogger:
    """Persist reproducibility metadata and hashes without private text."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                """
                    CREATE TABLE IF NOT EXISTS answer_audit (
                        event_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        requested_model TEXT NOT NULL,
                        actual_model TEXT,
                        prompt_version TEXT NOT NULL,
                        input_tokens INTEGER NOT NULL,
                        output_tokens INTEGER NOT NULL,
                        latency_ms REAL NOT NULL,
                        attempts INTEGER NOT NULL,
                        citation_ids_json TEXT NOT NULL,
                        chunk_ids_json TEXT NOT NULL,
                        storage_classes_json TEXT NOT NULL,
                        answer_sha256 TEXT NOT NULL
                    )
                    """
            )

    def log(self, result: RAGAnswer) -> bool:
        citation_ids = [citation.citation_id for citation in result.citations]
        chunk_ids = [citation.chunk_id for citation in result.citations]
        storage_classes = sorted({citation.storage_class for citation in result.citations})
        answer_sha256 = hashlib.sha256(result.answer_markdown.encode("utf-8")).hexdigest()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                """
                    INSERT INTO answer_audit (
                        event_id, created_at, status, requested_model, actual_model,
                        prompt_version, input_tokens, output_tokens, latency_ms, attempts,
                        citation_ids_json, chunk_ids_json, storage_classes_json, answer_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                (
                    uuid.uuid4().hex,
                    datetime.now(UTC).isoformat(),
                    result.status,
                    result.requested_model,
                    result.actual_model,
                    result.prompt_version,
                    result.input_tokens,
                    result.output_tokens,
                    result.latency_ms,
                    result.attempts,
                    json.dumps(citation_ids, separators=(",", ":")),
                    json.dumps(chunk_ids, separators=(",", ":")),
                    json.dumps(storage_classes, separators=(",", ":")),
                    answer_sha256,
                ),
            )
        return True
