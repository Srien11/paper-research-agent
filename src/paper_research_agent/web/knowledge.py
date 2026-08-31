"""Owner-scoped staging and durable jobs for incremental knowledge-base updates."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

KnowledgeItemStatus = Literal["staged", "ready", "queued", "running", "published", "failed"]


class KnowledgeItem(BaseModel):
    """A PDF that is not part of the corpus until an explicit publish succeeds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    filename: str = Field(min_length=1, max_length=180)
    size_bytes: int = Field(gt=0, le=100 * 1024 * 1024)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: KnowledgeItemStatus
    corpus_id: str | None = Field(default=None, pattern=r"^[CT]\d{3}$")
    title: str | None = Field(default=None, max_length=500)
    authors: tuple[str, ...] = Field(default=(), max_length=30)
    year: int | None = Field(default=None, ge=1900, le=2100)
    official_url: str | None = Field(default=None, max_length=2_000)
    storage_class: Literal["redistributable", "internal_research_only"] = "internal_research_only"
    error: str | None = Field(default=None, max_length=1_000)
    build_id: str | None = Field(default=None, max_length=256)
    created_at: datetime
    updated_at: datetime

    @property
    def ready_for_publish(self) -> bool:
        return bool(self.corpus_id and self.title and self.authors and self.year and self.official_url)


class KnowledgeItemPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str | None = Field(default=None, min_length=1, max_length=500)
    authors: tuple[str, ...] | None = Field(default=None, min_length=1, max_length=30)
    year: int | None = Field(default=None, ge=1900, le=2100)
    official_url: str | None = Field(default=None, min_length=8, max_length=2_000)
    storage_class: Literal["redistributable", "internal_research_only"] | None = None


@dataclass(frozen=True, slots=True)
class KnowledgePublishResult:
    item: KnowledgeItem
    changed: bool


class KnowledgeBaseStore:
    """Small SQLite ledger. It deliberately never stores provider payloads or file paths."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.staging_dir = root / "staging"
        self.path = root / "knowledge-base-v1.sqlite3"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def stage(self, *, filename: str, chunks: Iterable[bytes]) -> KnowledgeItem:
        safe_name = _safe_filename(filename)
        item_id = uuid.uuid4().hex
        target = self.staging_dir / f"{item_id}.pdf"
        digest = hashlib.sha256()
        size = 0
        with target.open("xb") as stream:
            for chunk in chunks:
                if not chunk:
                    continue
                size += len(chunk)
                if size > 100 * 1024 * 1024:
                    target.unlink(missing_ok=True)
                    raise ValueError("知识库 PDF 超过 100 MiB 限制")
                digest.update(chunk)
                stream.write(chunk)
        if size == 0:
            target.unlink(missing_ok=True)
            raise ValueError("知识库 PDF 不能为空")
        now = datetime.now(UTC)
        item = KnowledgeItem(
            item_id=item_id,
            filename=safe_name,
            size_bytes=size,
            sha256=digest.hexdigest(),
            status="staged",
            created_at=now,
            updated_at=now,
        )
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                """INSERT INTO knowledge_items (
                    item_id, filename, size_bytes, sha256, status, corpus_id, title,
                    authors_json, year, official_url, storage_class, error, build_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, '[]', NULL, NULL, ?, NULL, NULL, ?, ?)""",
                (
                    item.item_id, item.filename, item.size_bytes, item.sha256, item.status,
                    item.storage_class, item.created_at.isoformat(), item.updated_at.isoformat(),
                ),
            )
            connection.commit()
        return item

    def list(self) -> tuple[KnowledgeItem, ...]:
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM knowledge_items ORDER BY created_at DESC"
            ).fetchall()
        return tuple(_item(row) for row in rows)

    def get(self, item_id: str) -> KnowledgeItem:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_items WHERE item_id = ?", (item_id,)
            ).fetchone()
        if row is None:
            raise KeyError("知识库资料不存在")
        return _item(row)

    def update(self, item_id: str, patch: KnowledgeItemPatch, *, corpus_id: str) -> KnowledgeItem:
        current = self.get(item_id)
        if current.status not in {"staged", "ready", "failed"}:
            raise ValueError("正在处理或已发布的资料不能修改")
        values = current.model_dump()
        for field, value in patch.model_dump(exclude_none=True).items():
            values[field] = value
        values["corpus_id"] = current.corpus_id or corpus_id
        values["updated_at"] = datetime.now(UTC)
        candidate = KnowledgeItem.model_validate(values)
        status: KnowledgeItemStatus = "ready" if candidate.ready_for_publish else "staged"
        candidate = candidate.model_copy(update={"status": status, "error": None})
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                """UPDATE knowledge_items SET corpus_id=?, title=?, authors_json=?, year=?,
                    official_url=?, storage_class=?, status=?, error=NULL, updated_at=?
                    WHERE item_id=?""",
                (
                    candidate.corpus_id, candidate.title, json.dumps(candidate.authors, ensure_ascii=False),
                    candidate.year, candidate.official_url, candidate.storage_class, candidate.status,
                    candidate.updated_at.isoformat(), item_id,
                ),
            )
            connection.commit()
        return candidate

    def set_status(self, item_id: str, status: KnowledgeItemStatus, *, error: str | None = None, build_id: str | None = None) -> KnowledgeItem:
        current = self.get(item_id)
        updated = current.model_copy(update={
            "status": status, "error": error, "build_id": build_id or current.build_id,
            "updated_at": datetime.now(UTC),
        })
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                "UPDATE knowledge_items SET status=?, error=?, build_id=?, updated_at=? WHERE item_id=?",
                (updated.status, updated.error, updated.build_id, updated.updated_at.isoformat(), item_id),
            )
            connection.commit()
        return updated

    def staged_path(self, item_id: str) -> Path:
        path = self.staging_dir / f"{item_id}.pdf"
        if not path.is_file():
            raise FileNotFoundError("知识库暂存文件不存在")
        return path

    def next_corpus_id(self, corpus_dir: Path) -> str:
        used = {item.corpus_id for item in self.list() if item.corpus_id}
        for name in ("core_frozen.jsonl", "challenge_frozen.jsonl"):
            path = corpus_dir / name
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    value = json.loads(line)
                    if isinstance(value, dict) and isinstance(value.get("corpus_id"), str):
                        used.add(value["corpus_id"])
                except json.JSONDecodeError:
                    continue
        for number in range(1, 1_000):
            candidate = f"C{number:03d}"
            if candidate not in used:
                return candidate
        raise ValueError("没有可用的 corpus_id")

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS knowledge_items (
                    item_id TEXT PRIMARY KEY, filename TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL, status TEXT NOT NULL, corpus_id TEXT, title TEXT,
                    authors_json TEXT NOT NULL, year INTEGER, official_url TEXT,
                    storage_class TEXT NOT NULL, error TEXT, build_id TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                )"""
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


def publish_item(
    store: KnowledgeBaseStore,
    item_id: str,
    *,
    corpus_dir: Path,
    output_root: Path,
    chunking_config: Path,
    retrieval_config: Path,
) -> KnowledgePublishResult:
    """Copy one reviewed source into the frozen manifest and atomically publish a complete build."""
    from paper_research_agent.chunking.models import EvidenceChunk
    from paper_research_agent.chunking.runner import run_chunking
    from paper_research_agent.ingestion.continuous import (
        activate_current_knowledge_base,
        update_current_ingestion_build,
    )
    from paper_research_agent.retrieval.config import load_retrieval_config
    from paper_research_agent.retrieval.indexer import build_index
    from paper_research_agent.retrieval.model_adapters import FastEmbedEncoder

    item = store.get(item_id)
    if item.status not in {"ready", "failed"} or not item.ready_for_publish:
        raise ValueError("资料尚未补齐或不可发布")
    store.set_status(item_id, "running")
    try:
        _freeze_source(store, item, corpus_dir)
        update = update_current_ingestion_build(corpus_dir, output_root)
        pointer = output_root / "current_knowledge_base.json"
        if update.changed or _pointer_build_id(pointer) != update.result.manifest.build_id:
            chunks_path, _ = run_chunking(
                update.result.output_dir / "elements.jsonl",
                update.result.output_dir / "sections.jsonl",
                chunking_config,
                output_dir=update.result.output_dir / "chunks",
            )
            config = load_retrieval_config(retrieval_config)
            chunks = [
                EvidenceChunk.model_validate_json(line)
                for line in chunks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            build_index(
                chunks,
                FastEmbedEncoder(config.embedding_model, revision=config.embedding_revision),
                update.result.output_dir / "index",
                embedding_model=config.embedding_model,
                embedding_revision=config.embedding_revision,
                chunk_build_sha256=hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
            )
            activate_current_knowledge_base(output_root, update.result)
        published = store.set_status(item_id, "published", build_id=update.result.manifest.build_id)
        return KnowledgePublishResult(item=published, changed=update.changed)
    except Exception as error:
        store.set_status(item_id, "failed", error=str(error)[:1_000])
        raise


def _freeze_source(store: KnowledgeBaseStore, item: KnowledgeItem, corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    source_dir = corpus_dir / "uploads"
    source_dir.mkdir(exist_ok=True)
    target = source_dir / f"{item.corpus_id}.pdf"
    if not target.exists():
        shutil.copyfile(store.staged_path(item.item_id), target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != item.sha256:
        raise ValueError("暂存文件哈希与冻结目标不一致")
    record = {
        "corpus_id": item.corpus_id,
        "corpus_version": _corpus_version(corpus_dir),
        "dataset_split": "core",
        "canonical_key": f"owner:{item.sha256}",
        "title": item.title,
        "year": item.year,
        "authors": list(item.authors),
        "official_url": item.official_url,
        "fulltext_url": item.official_url,
        "selection_status": "frozen",
        "content_status": "downloaded_and_parse_verified",
        "storage_class": item.storage_class,
        "local_pdf_path": str(target.resolve()),
        "download_sha256": item.sha256,
        "download_bytes": item.size_bytes,
        "pdf_pages": 1,
        "parse_quality_status": "machine_parse_pass",
    }
    manifest = corpus_dir / "core_frozen.jsonl"
    existing = manifest.read_text(encoding="utf-8").splitlines() if manifest.is_file() else []
    if any(json.loads(line).get("corpus_id") == item.corpus_id for line in existing if line.strip()):
        return
    manifest.write_text("\n".join([*existing, json.dumps(record, ensure_ascii=False)]) + "\n", encoding="utf-8")
    (corpus_dir / "challenge_frozen.jsonl").touch(exist_ok=True)


def _corpus_version(corpus_dir: Path) -> str:
    for name in ("core_frozen.jsonl", "challenge_frozen.jsonl"):
        path = corpus_dir / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    version = value.get("corpus_version")
                    if isinstance(version, str):
                        return version
    raise ValueError("语料清单为空，无法确定 corpus_version")


def _pointer_build_id(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value.get("build_id") if isinstance(value, dict) and isinstance(value.get("build_id"), str) else None


def _safe_filename(value: str) -> str:
    name = Path(value).name.strip()
    if not name.lower().endswith(".pdf"):
        raise ValueError("知识库目前只接受 PDF")
    if not name or len(name) > 180:
        raise ValueError("无效的知识库文件名")
    return name


def _item(row: sqlite3.Row) -> KnowledgeItem:
    return KnowledgeItem(
        item_id=row["item_id"], filename=row["filename"], size_bytes=row["size_bytes"],
        sha256=row["sha256"], status=row["status"], corpus_id=row["corpus_id"],
        title=row["title"], authors=tuple(json.loads(row["authors_json"])), year=row["year"],
        official_url=row["official_url"], storage_class=row["storage_class"], error=row["error"],
        build_id=row["build_id"], created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )
