from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
import uuid
from contextlib import suppress
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.agent.orchestrator.artifacts import LocalRAGArtifact
from paper_research_agent.agent.orchestrator.models import MainAgentRequest
from paper_research_agent.web.bootstrap import (
    ApplicationEnvironment,
    create_application_services_from_environment,
)
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


async def run_main_agent_smoke(
    question: str,
    *,
    repeat: int,
    reuse_conversation: bool,
) -> dict[str, object]:
    """Run body-free production main-Agent checks and return aggregate safe traces."""
    smoke_started = time.perf_counter()
    environment = ApplicationEnvironment.from_environment()
    load_started = time.perf_counter()
    services = await create_application_services_from_environment()
    runtime_load_ms = round((time.perf_counter() - load_started) * 1000, 2)
    runtime = services.main_agent_runtime
    if runtime is None:
        await services.aclose()
        raise RuntimeError("main agent runtime is unavailable")
    shared_conversation = f"main-smoke-{uuid.uuid4().hex}"
    run_summaries: list[dict[str, object]] = []
    conversations: set[str] = set()
    try:
        for index in range(repeat):
            conversation_id = (
                shared_conversation
                if reuse_conversation
                else f"main-smoke-{uuid.uuid4().hex}"
            )
            conversations.add(conversation_id)
            request = MainAgentRequest(
                request_id=f"main-smoke-request-{uuid.uuid4().hex}",
                conversation_id=conversation_id,
                message=question,
                rag_mode="preferred",
            )
            event_path = environment.main_checkpoint_path.with_name(
                "agent-events-v1.sqlite3"
            )
            after_event_id = _latest_agent_event_id(event_path)
            run_started = time.perf_counter()
            result = await runtime.run(request)
            run_ms = round((time.perf_counter() - run_started) * 1000, 2)
            hydration = _hydration_trace(
                event_path,
                result.run_id,
            )
            rag_summary = _rag_child_summary(result.child_results)
            comparison_stages = _comparison_stage_trace(
                event_path,
                after_event_id,
            )
            run_summaries.append(
                {
                    "sequence": index + 1,
                    "run_ms": run_ms,
                    "status": result.status,
                    "route_trace": result.route_trace,
                    "child_count": len(result.child_results),
                    "source_count": sum(
                        len(child.source_ids) for child in result.child_results
                    ),
                    **rag_summary,
                    **comparison_stages,
                    "hydration": hydration,
                }
            )
    finally:
        for conversation_id in conversations:
            with suppress(Exception):
                await runtime.clear(conversation_id)
            with suppress(Exception):
                await asyncio.to_thread(services.conversation_store.clear, conversation_id)
        await services.aclose()

    run_times: list[float] = []
    hydrate_times: list[float] = []
    for item in run_summaries:
        run_ms_value = item.get("run_ms")
        if isinstance(run_ms_value, (int, float)):
            run_times.append(float(run_ms_value))
        hydration_value = item.get("hydration")
        if not isinstance(hydration_value, dict):
            continue
        duration_ms = hydration_value.get("duration_ms")
        if isinstance(duration_ms, (int, float)):
            hydrate_times.append(float(duration_ms))
    return {
        "smoke_timings": {
            "runtime_load_ms": runtime_load_ms,
            "run_p50_ms": round(median(run_times), 2),
            "run_p95_ms": _p95(run_times),
            "hydrate_p50_ms": round(median(hydrate_times), 2)
            if hydrate_times
            else None,
            "hydrate_p95_ms": _p95(hydrate_times) if hydrate_times else None,
            "total_ms": round((time.perf_counter() - smoke_started) * 1000, 2),
        },
        "repeat": repeat,
        "statuses": [item["status"] for item in run_summaries],
        "route_traces": [item["route_trace"] for item in run_summaries],
        "child_counts": [item["child_count"] for item in run_summaries],
        "source_counts": [item["source_count"] for item in run_summaries],
        "runs": run_summaries,
        "hydration": [item["hydration"] for item in run_summaries],
    }


def _rag_child_summary(
    child_results: tuple[object, ...],
) -> dict[str, int | float]:
    artifacts: list[LocalRAGArtifact] = []
    for child in child_results:
        artifact = getattr(child, "artifact", None)
        if getattr(artifact, "kind", None) != "local_rag":
            continue
        artifacts.append(LocalRAGArtifact.model_validate(artifact))
    return {
        "research_agent_ms": sum(item.metrics.elapsed_ms for item in artifacts),
        "answer_provider_ms": round(
            sum(item.answer.latency_ms for item in artifacts), 2
        ),
        "answer_attempts": sum(item.answer.attempts for item in artifacts),
    }


def _comparison_stage_trace(
    path: Path,
    after_event_id: int,
) -> dict[str, float | None]:
    empty: dict[str, float | None] = {
        "comparison_plan_ms": None,
        "comparison_search_batch_ms": None,
        "compiler_ms": None,
    }
    try:
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                "SELECT name, duration_ms FROM agent_events "
                "WHERE event_id > ? "
                "AND event_type = 'node_completed' AND status = 'succeeded' "
                "AND name IN ('plan', 'execute_tools', 'assess_evidence')",
                (after_event_id,),
            ).fetchall()
    except (OSError, sqlite3.Error):
        return empty
    totals: dict[str, float] = {}
    for name, duration_ms in rows:
        if isinstance(duration_ms, (int, float)):
            totals[str(name)] = totals.get(str(name), 0.0) + float(duration_ms)
    return {
        "comparison_plan_ms": _optional_milliseconds(totals.get("plan")),
        "comparison_search_batch_ms": _optional_milliseconds(
            totals.get("execute_tools")
        ),
        "compiler_ms": _optional_milliseconds(totals.get("assess_evidence")),
    }


def _optional_milliseconds(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def _latest_agent_event_id(path: Path) -> int:
    try:
        with sqlite3.connect(path) as connection:
            row = connection.execute("SELECT MAX(event_id) FROM agent_events").fetchone()
    except (OSError, sqlite3.Error):
        return 0
    value = None if row is None else row[0]
    return int(value) if isinstance(value, int) else 0


def _hydration_trace(path: Path, run_id: str) -> dict[str, object]:
    try:
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                "SELECT name, duration_ms, returned_count, recent_message_count, "
                "recalled_conversation_count, recalled_memory_count, context_char_count, "
                "estimated_context_tokens, degraded FROM agent_events "
                "WHERE run_id = ? AND name LIKE 'main_hydrate_%' ORDER BY event_id",
                (run_id,),
            ).fetchall()
    except (OSError, sqlite3.Error):
        return {"available": False}
    stages = {
        str(row[0]): {
            "duration_ms": row[1],
            "returned_count": row[2],
            "degraded": bool(row[8]) if row[8] is not None else None,
        }
        for row in rows
    }
    context = next((row for row in rows if row[0] == "main_hydrate_context"), None)
    if context is None:
        return {"available": False, "stages": stages}
    return {
        "available": True,
        "duration_ms": context[1],
        "recent_message_count": context[3],
        "recalled_conversation_count": context[4],
        "recalled_memory_count": context[5],
        "context_char_count": context[6],
        "estimated_context_tokens": context[7],
        "degraded": bool(context[8]) if context[8] is not None else None,
        "stages": stages,
    }


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return round(ordered[index], 2)


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
    parser.add_argument(
        "--main-agent",
        action="store_true",
        help="Run the production main Agent and print only body-free aggregate traces.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--reuse-conversation",
        action="store_true",
        help="Reuse one temporary conversation across repeated main-Agent runs.",
    )
    args = parser.parse_args()
    if args.repeat < 1 or args.repeat > 20:
        parser.error("--repeat must be between 1 and 20")
    try:
        payload = asyncio.run(
            run_main_agent_smoke(
                args.question,
                repeat=args.repeat,
                reuse_conversation=args.reuse_conversation,
            )
            if args.main_agent
            else run_smoke(args.question)
        )
    except Exception as error:  # noqa: BLE001 - never print provider messages or paths
        parser.exit(1, f"Web runtime smoke failed: {type(error).__name__}\n")
    if args.main_agent:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.summary_only:
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
