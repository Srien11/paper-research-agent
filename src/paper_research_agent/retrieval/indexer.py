"""持久化 FAISS IndexFlatIP 与可追溯的 SQLite 检索元数据。"""

from __future__ import annotations

import hashlib
import json
import platform
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from paper_research_agent.chunking.chunker import canonical_sha256
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.retrieval.contracts import IndexManifest
from paper_research_agent.retrieval.vector import Encoder, VectorIndex


def build_index(
    chunks: list[EvidenceChunk],
    encoder: Encoder,
    output_dir: Path,
    *,
    embedding_model: str,
    embedding_revision: str,
    chunk_build_sha256: str,
) -> IndexManifest:
    try:
        import faiss
        import numpy as np
    except ImportError as error:
        raise RuntimeError("install the retrieval extra to build a FAISS index") from error
    vector_index = VectorIndex(chunks, encoder)
    if not chunks:
        raise ValueError("cannot build an empty retrieval index")
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(vector_index.vectors, dtype="float32")
    faiss_index: Any = faiss.IndexFlatIP(vector_index.dimension)
    faiss_index.add(vectors)
    faiss_path = output_dir / "vectors.faiss"
    metadata_path = output_dir / "metadata.sqlite"
    faiss.write_index(faiss_index, str(faiss_path))
    write_chunk_metadata(vector_index.chunks, metadata_path)
    files_sha256 = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (faiss_path, metadata_path)
    }
    identity = {
        "chunks": [chunk.chunk_id for chunk in vector_index.chunks],
        "model": embedding_model,
        "revision": embedding_revision,
        "dimension": vector_index.dimension,
    }
    manifest = IndexManifest(
        index_id=f"idx_{canonical_sha256(identity)[:24]}",
        chunk_build_sha256=chunk_build_sha256,
        chunk_count=len(chunks),
        embedding_model=embedding_model,
        embedding_revision=embedding_revision,
        vector_dimension=vector_index.dimension,
        files_sha256=files_sha256,
        cpu_fingerprint=f"{platform.machine()}|{platform.processor()}|{platform.python_version()}",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_chunk_metadata(
    chunks: Sequence[EvidenceChunk],
    metadata_path: Path,
) -> None:
    """写入与向量位置严格对齐的本地检索元数据。"""

    with closing(sqlite3.connect(metadata_path)) as connection, connection:
        connection.execute("DROP TABLE IF EXISTS chunks")
        connection.execute(
            """
            CREATE TABLE chunks (
                position INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE NOT NULL,
                corpus_id TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                section_id TEXT,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                text_sha256 TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                figure_json TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    position,
                    chunk.chunk_id,
                    chunk.corpus_id,
                    chunk.asset_id,
                    chunk.section_id,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.text_sha256,
                    chunk.evidence_type,
                    (
                        chunk.figure.model_dump_json()
                        if chunk.figure is not None
                        else None
                    ),
                )
                for position, chunk in enumerate(chunks)
            ],
        )
