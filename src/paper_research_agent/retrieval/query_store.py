"""SQLite persistence for query rewrites and privacy-bounded retrieval audits."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from paper_research_agent.retrieval.contracts import QueryRewriteTrace

CACHE_SCHEMA_VERSION = "query-rewrite-cache-v2"
AUDIT_SCHEMA_VERSION = "query-audit-v1"


@dataclass(frozen=True)
class CachedRewrite:
    english_query: str
    actual_model: str
    created_at: datetime
    latency_ms: float
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class CacheLookup:
    fresh: CachedRewrite | None = None
    stale: CachedRewrite | None = None


class QueryRewriteCache(Protocol):
    def lookup(
        self,
        query: str,
        *,
        model: str,
        prompt_version: str,
        fresh_days: int,
        stale_days: int,
    ) -> CacheLookup: ...

    def put(
        self,
        query: str,
        *,
        model: str,
        prompt_version: str,
        english_query: str,
        actual_model: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        stale_days: int,
    ) -> None: ...


class NullQueryRewriteCache:
    """Fail-open cache used when local persistence cannot be initialized."""

    def lookup(
        self,
        query: str,
        *,
        model: str,
        prompt_version: str,
        fresh_days: int,
        stale_days: int,
    ) -> CacheLookup:
        del query, model, prompt_version, fresh_days, stale_days
        return CacheLookup()

    def put(
        self,
        query: str,
        *,
        model: str,
        prompt_version: str,
        english_query: str,
        actual_model: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        stale_days: int,
    ) -> None:
        del (
            query,
            model,
            prompt_version,
            english_query,
            actual_model,
            latency_ms,
            input_tokens,
            output_tokens,
            stale_days,
        )


@dataclass(frozen=True)
class AuditRanking:
    stage: str
    chunk_id: str
    rank: int
    score: float
    final_rank: int | None = None


@dataclass(frozen=True)
class QueryAuditRecord:
    request_id: str
    created_at: datetime
    original_query: str
    rewrite: QueryRewriteTrace
    pipeline_id: str
    index_id: str
    config_sha256: str
    degraded_reason: str | None
    latency_ms: Mapping[str, float]
    rankings: Sequence[AuditRanking]
    plaintext_days: int | None = None

    def __post_init__(self) -> None:
        if self.plaintext_days is not None and self.plaintext_days < 0:
            raise ValueError("plaintext_days cannot be negative")


def normalize_query(query: str) -> str:
    normalized = unicodedata.normalize("NFC", query).strip()
    if not normalized:
        raise ValueError("query cannot be blank")
    return normalized


def query_sha256(query: str) -> str:
    return hashlib.sha256(normalize_query(query).encode("utf-8")).hexdigest()


def rewrite_cache_key(query: str, *, model: str, prompt_version: str) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "prompt_version": prompt_version,
        "requested_model": model,
        "temperature": 0.1,
        "top_p": 0.7,
        "enable_thinking": False,
        "max_tokens": 512,
        "query": normalize_query(query),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SQLiteQueryRewriteCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def lookup(
        self,
        query: str,
        *,
        model: str,
        prompt_version: str,
        fresh_days: int,
        stale_days: int,
    ) -> CacheLookup:
        key = rewrite_cache_key(query, model=model, prompt_version=prompt_version)
        now = datetime.now(UTC)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM rewrites WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now.isoformat(),),
            )
            row = connection.execute(
                """SELECT english_query, actual_model, created_at, latency_ms,
                          input_tokens, output_tokens
                   FROM rewrites WHERE cache_key = ?""",
                (key,),
            ).fetchone()
            if row is not None:
                created_at = datetime.fromisoformat(str(row[2]))
                if now - created_at.astimezone(UTC) > timedelta(days=stale_days):
                    connection.execute("DELETE FROM rewrites WHERE cache_key = ?", (key,))
                    row = None
            connection.commit()
        if row is None:
            return CacheLookup()
        entry = CachedRewrite(
            english_query=str(row[0]),
            actual_model=str(row[1]),
            created_at=datetime.fromisoformat(str(row[2])),
            latency_ms=float(row[3]),
            input_tokens=int(row[4]),
            output_tokens=int(row[5]),
        )
        age = datetime.now(UTC) - entry.created_at.astimezone(UTC)
        if age <= timedelta(days=fresh_days):
            return CacheLookup(fresh=entry)
        if age <= timedelta(days=stale_days):
            return CacheLookup(stale=entry)
        return CacheLookup()

    def put(
        self,
        query: str,
        *,
        model: str,
        prompt_version: str,
        english_query: str,
        actual_model: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        stale_days: int,
    ) -> None:
        key = rewrite_cache_key(query, model=model, prompt_version=prompt_version)
        now = datetime.now(UTC)
        expires_at = (now + timedelta(days=stale_days)).isoformat()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO rewrites (
                       cache_key, english_query, actual_model, created_at,
                       expires_at, latency_ms, input_tokens, output_tokens
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET
                       english_query = excluded.english_query,
                       actual_model = excluded.actual_model,
                       created_at = excluded.created_at,
                       expires_at = excluded.expires_at,
                       latency_ms = excluded.latency_ms,
                       input_tokens = excluded.input_tokens,
                       output_tokens = excluded.output_tokens""",
                (
                    key,
                    english_query,
                    actual_model,
                    now.isoformat(),
                    expires_at,
                    latency_ms,
                    input_tokens,
                    output_tokens,
                ),
            )
            connection.execute("DELETE FROM rewrites WHERE expires_at <= ?", (now.isoformat(),))
            connection.commit()

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS rewrites (
                    cache_key TEXT PRIMARY KEY,
                    english_query TEXT NOT NULL,
                    actual_model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL
                );
                PRAGMA user_version = 1;
                """
            )
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(rewrites)")}
            if "expires_at" not in columns:
                connection.execute("ALTER TABLE rewrites ADD COLUMN expires_at TEXT")
                legacy_expiry = datetime.now(UTC) + timedelta(days=365)
                connection.execute(
                    "UPDATE rewrites SET expires_at = ? WHERE expires_at IS NULL",
                    (legacy_expiry.isoformat(),),
                )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection


class SQLiteQueryAuditLogger:
    """Best-effort local audit persistence; never stores evidence text or provider payloads."""

    def __init__(self, path: Path, *, plaintext_days: int):
        if plaintext_days < 0:
            raise ValueError("plaintext_days cannot be negative")
        self.path = path
        self.plaintext_days = plaintext_days
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def write(self, record: QueryAuditRecord) -> bool:
        try:
            plaintext_days = (
                self.plaintext_days
                if record.plaintext_days is None
                else min(self.plaintext_days, record.plaintext_days)
            )
            plaintext_expires_at = (
                record.created_at.astimezone(UTC) + timedelta(days=plaintext_days)
            ).isoformat()
            with self._lock, closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """INSERT INTO runs (
                           request_id, created_at, query_sha256, original_query,
                           rewritten_query, rewrite_status, requested_model, actual_model,
                           prompt_version, rewrite_latency_ms, input_tokens, output_tokens,
                           error_class, fallback_reason, cache_error_class, pipeline_id,
                           index_id, config_sha256, degraded_reason, latency_json,
                           plaintext_expires_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.request_id,
                        record.created_at.astimezone(UTC).isoformat(),
                        query_sha256(record.original_query),
                        record.original_query,
                        record.rewrite.english_query,
                        record.rewrite.status,
                        record.rewrite.requested_model,
                        record.rewrite.actual_model,
                        record.rewrite.prompt_version,
                        record.rewrite.latency_ms,
                        record.rewrite.input_tokens,
                        record.rewrite.output_tokens,
                        record.rewrite.error_class,
                        record.rewrite.fallback_reason,
                        record.rewrite.cache_error_class,
                        record.pipeline_id,
                        record.index_id,
                        record.config_sha256,
                        record.degraded_reason,
                        json.dumps(dict(record.latency_ms), sort_keys=True),
                        plaintext_expires_at,
                    ),
                )
                connection.executemany(
                    """INSERT INTO rankings (
                           request_id, stage, chunk_id, rank, score, final_rank
                       ) VALUES (?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            record.request_id,
                            item.stage,
                            item.chunk_id,
                            item.rank,
                            item.score,
                            item.final_rank,
                        )
                        for item in record.rankings
                    ],
                )
                connection.execute(
                    """UPDATE runs SET original_query = NULL, rewritten_query = NULL
                       WHERE plaintext_expires_at <= ?
                         AND (original_query IS NOT NULL OR rewritten_query IS NOT NULL)""",
                    (datetime.now(UTC).isoformat(),),
                )
                connection.commit()
            return True
        except sqlite3.Error:
            return False

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    request_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    query_sha256 TEXT NOT NULL,
                    original_query TEXT,
                    rewritten_query TEXT,
                    rewrite_status TEXT NOT NULL,
                    requested_model TEXT NOT NULL,
                    actual_model TEXT,
                    prompt_version TEXT NOT NULL,
                    rewrite_latency_ms REAL NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    error_class TEXT,
                    fallback_reason TEXT,
                    cache_error_class TEXT,
                    pipeline_id TEXT NOT NULL,
                    index_id TEXT NOT NULL,
                    config_sha256 TEXT NOT NULL,
                    degraded_reason TEXT,
                    latency_json TEXT NOT NULL,
                    plaintext_expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rankings (
                    request_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    score REAL NOT NULL,
                    final_rank INTEGER,
                    PRIMARY KEY (request_id, stage, chunk_id),
                    FOREIGN KEY (request_id) REFERENCES runs(request_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS rankings_request_stage
                    ON rankings(request_id, stage, rank);
                """
            )
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(runs)")}
            if "fallback_reason" not in columns:
                connection.execute("ALTER TABLE runs ADD COLUMN fallback_reason TEXT")
            if "cache_error_class" not in columns:
                connection.execute("ALTER TABLE runs ADD COLUMN cache_error_class TEXT")
            if "plaintext_expires_at" not in columns:
                connection.execute("ALTER TABLE runs ADD COLUMN plaintext_expires_at TEXT")
            missing_expiry = connection.execute(
                "SELECT request_id, created_at FROM runs WHERE plaintext_expires_at IS NULL"
            ).fetchall()
            connection.executemany(
                "UPDATE runs SET plaintext_expires_at = ? WHERE request_id = ?",
                [
                    (
                        (
                            datetime.fromisoformat(str(created_at)).astimezone(UTC)
                            + timedelta(days=self.plaintext_days)
                        ).isoformat(),
                        str(request_id),
                    )
                    for request_id, created_at in missing_expiry
                ],
            )
            connection.execute(
                """UPDATE runs SET original_query = NULL, rewritten_query = NULL
                   WHERE plaintext_expires_at <= ?
                     AND (original_query IS NOT NULL OR rewritten_query IS NOT NULL)""",
                (datetime.now(UTC).isoformat(),),
            )
            connection.execute("PRAGMA user_version = 2")
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
