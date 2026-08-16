from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import suppress
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.web.runtime import RAGRuntime


def _load_local_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


async def run_smoke(question: str) -> dict[str, object]:
    """Run one production-shaped query and return no evidence excerpts or local paths."""
    smoke_started = time.perf_counter()
    session_id = f"local-web-smoke-{uuid.uuid4().hex}"
    load_started = time.perf_counter()
    runtime = (
        await RAGRuntime.from_environment_with_agent()
        if RAGRuntime.research_agent_enabled_from_environment()
        else RAGRuntime.from_environment()
    )
    runtime_load_ms = round((time.perf_counter() - load_started) * 1000, 2)
    try:
        ask_started = time.perf_counter()
        result = await runtime.ask(question, session_id=session_id)
        ask_ms = round((time.perf_counter() - ask_started) * 1000, 2)
        return {
            "smoke_timings": {
                "runtime_load_ms": runtime_load_ms,
                "ask_ms": ask_ms,
                "total_ms": round((time.perf_counter() - smoke_started) * 1000, 2),
            },
            "answer": result.answer.model_dump(mode="json"),
            "sources": [
                {
                    "citation_id": source.citation_id,
                    "corpus_id": source.corpus_id,
                    "title": source.title,
                    "official_url": source.official_url,
                    "page_start": source.page_start,
                    "page_end": source.page_end,
                    "evidence_type": source.evidence_type,
                    "storage_class": source.storage_class,
                }
                for source in result.sources
            ],
            "retrieval": result.retrieval.model_dump(mode="json", exclude={"hits"}),
            "context": result.context.model_dump(mode="json"),
            "generation": result.generation.model_dump(mode="json"),
        }
    finally:
        with suppress(Exception):
            await runtime.clear_conversation(session_id)
        await runtime.aclose()


def main() -> None:
    _load_local_env(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Run one safe, owner-only Web runtime smoke query."
    )
    parser.add_argument("question")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print timings and bounded result counts without answer or source text.",
    )
    args = parser.parse_args()
    try:
        payload = asyncio.run(run_smoke(args.question))
    except Exception as error:  # noqa: BLE001 - never print provider messages or paths
        parser.exit(1, f"Web runtime smoke failed: {type(error).__name__}\n")
    if args.summary_only:
        answer = payload.get("answer", {})
        retrieval = payload.get("retrieval", {})
        generation = payload.get("generation", {})
        sources = payload.get("sources", [])
        summary = {
            "smoke_timings": payload["smoke_timings"],
            "answer_status": answer.get("status") if isinstance(answer, dict) else None,
            "source_count": len(sources) if isinstance(sources, list) else 0,
            "rewrite_status": (
                retrieval.get("rewrite_status") if isinstance(retrieval, dict) else None
            ),
            "generation_latency_ms": (
                generation.get("latency_ms") if isinstance(generation, dict) else None
            ),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
