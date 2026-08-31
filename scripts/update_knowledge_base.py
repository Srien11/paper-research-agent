"""持续检查冻结论文清单；新增 PDF 将写入新的不可变解析构建并原子切换当前指针。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.chunking.runner import run_chunking
from paper_research_agent.ingestion.continuous import (
    activate_current_knowledge_base,
    update_current_ingestion_build,
)
from paper_research_agent.ingestion.runner import IngestionRunError
from paper_research_agent.retrieval.config import load_retrieval_config
from paper_research_agent.retrieval.indexer import build_index
from paper_research_agent.retrieval.model_adapters import FastEmbedEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--watch-interval-seconds",
        type=float,
        help="可选轮询间隔；未提供时只执行一次更新。",
    )
    parser.add_argument(
        "--chunking-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "chunking" / "baseline-v1.json",
    )
    parser.add_argument(
        "--retrieval-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "retrieval" / "hybrid-rerank-v1.json",
    )
    return parser.parse_args()


def run_once(args: argparse.Namespace) -> int:
    try:
        update = update_current_ingestion_build(args.corpus_dir, args.output_root)
        knowledge_pointer = args.output_root / "current_knowledge_base.json"
        needs_publish = update.changed or _pointer_build_id(knowledge_pointer) != update.result.manifest.build_id
        if needs_publish:
            chunks_path, _ = run_chunking(
                update.result.output_dir / "elements.jsonl",
                update.result.output_dir / "sections.jsonl",
                args.chunking_config,
                output_dir=update.result.output_dir / "chunks",
            )
            retrieval_config = load_retrieval_config(args.retrieval_config)
            chunks = [
                EvidenceChunk.model_validate_json(line)
                for line in chunks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            build_index(
                chunks,
                FastEmbedEncoder(
                    retrieval_config.embedding_model,
                    revision=retrieval_config.embedding_revision,
                ),
                update.result.output_dir / "index",
                embedding_model=retrieval_config.embedding_model,
                embedding_revision=retrieval_config.embedding_revision,
                chunk_build_sha256=hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
            )
            knowledge_pointer = activate_current_knowledge_base(
                args.output_root,
                update.result,
            )
    except (IngestionRunError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "changed": update.changed,
                "output_dir": str(update.result.output_dir),
                "build_id": update.result.manifest.build_id,
                "parent_build_id": update.result.manifest.parent_build_id,
                "added_asset_count": update.result.manifest.added_asset_count,
                "pointer_path": str(update.pointer_path),
                "knowledge_base_pointer_path": str(knowledge_pointer),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _pointer_build_id(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    build_id = value.get("build_id") if isinstance(value, dict) else None
    return build_id if isinstance(build_id, str) else None


def main() -> int:
    args = parse_args()
    interval = args.watch_interval_seconds
    if interval is not None and interval <= 0:
        raise SystemExit("--watch-interval-seconds 必须大于 0")
    while True:
        status = run_once(args)
        if interval is None or status != 0:
            return status
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
