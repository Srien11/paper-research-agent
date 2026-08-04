from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from contextlib import suppress
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.web.runtime import RAGRuntime


async def run_smoke(question: str) -> dict[str, object]:
    """Run one production-shaped query and return no evidence excerpts or local paths."""
    session_id = f"local-web-smoke-{uuid.uuid4().hex}"
    runtime = (
        await RAGRuntime.from_environment_with_agent()
        if RAGRuntime.research_agent_enabled_from_environment()
        else RAGRuntime.from_environment()
    )
    try:
        result = await runtime.ask(question, session_id=session_id)
        return {
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
    parser = argparse.ArgumentParser(
        description="Run one safe, owner-only Web runtime smoke query."
    )
    parser.add_argument("question")
    args = parser.parse_args()
    try:
        payload = asyncio.run(run_smoke(args.question))
    except Exception as error:  # noqa: BLE001 - never print provider messages or paths
        parser.exit(1, f"Web runtime smoke failed: {type(error).__name__}\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
