from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.dataset import load_dataset
from paper_research_agent.evaluation.end_to_end import (
    evaluate_end_to_end,
    write_end_to_end_report,
)
from paper_research_agent.web.runtime import RAGRuntime


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluation_context(runtime: RAGRuntime) -> dict[str, object]:
    retrieval_path = PROJECT_ROOT / "configs/retrieval/hybrid-rerank-v1.json"
    bilingual_path = PROJECT_ROOT / "configs/retrieval/bilingual-qwen-v1.json"
    answer_path = PROJECT_ROOT / "configs/answering/qwen-rag-v1.json"
    manifest_path = PROJECT_ROOT / "data/indexes/retrieval-v1/manifest.json"
    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip() or "unknown"
    status_output = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "code_revision": revision,
        "working_tree_dirty": bool(status_output),
        "index_id": manifest["index_id"],
        "chunk_count": runtime.chunk_count,
        "answer_model": answer["model"],
        "answer_prompt_version": answer["prompt_version"],
        "retrieval_config_sha256": _sha256(retrieval_path),
        "bilingual_config_sha256": _sha256(bilingual_path),
        "answer_config_sha256": _sha256(answer_path),
        "provider_base_url_configured": bool(os.getenv("DASHSCOPE_BASE_URL", "").strip()),
    }


async def run(args: argparse.Namespace) -> None:
    queries = load_dataset(args.dataset)
    if args.limit is not None:
        queries = queries[: args.limit]
    runtime = RAGRuntime.from_environment()
    try:
        result = await evaluate_end_to_end(
            runtime,
            queries,
            args.output,
            evaluation_context=_evaluation_context(runtime),
        )
        write_end_to_end_report(result, args.report)
    finally:
        with suppress(Exception):
            await runtime.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run privacy-safe end-to-end RAG evaluation.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "evaluation/datasets/dev-silver-v1.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/runs/end-to-end-v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/端到端RAG自动评测-v1.md",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(run(args))
    print(args.output)
    print(args.report)


if __name__ == "__main__":
    main()
