from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.context import ContextRequest, assemble_context
from paper_research_agent.context.adapters import join_retrieval_evidence
from paper_research_agent.retrieval.contracts import BilingualRetrievalRun, RetrievalRun

DEFAULT_SYSTEM_RULES = (
    "Answer only from supplied evidence. Preserve uncertainty and attach a citation "
    "marker to every factual claim."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a safe, citation-ready RAG context.")
    parser.add_argument("--run", type=Path, required=True, help="RetrievalRun JSON file")
    parser.add_argument("--chunks", type=Path, required=True, help="EvidenceChunk JSONL file")
    parser.add_argument("--question", help="Override the query stored in the retrieval run")
    parser.add_argument("--system-rules", default=DEFAULT_SYSTEM_RULES)
    parser.add_argument("--task-state")
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--output-reserve", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run = _load_run(args.run)
    chunks = [
        EvidenceChunk.model_validate_json(line)
        for line in args.chunks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    context = assemble_context(
        ContextRequest(
            system_rules=args.system_rules,
            user_question=args.question or _run_query(run),
            evidence=join_retrieval_evidence(run, chunks),
            task_state=args.task_state,
            token_budget=args.token_budget,
            output_reserve_tokens=args.output_reserve,
        )
    )
    payload = context.model_dump_json(indent=2)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)


def _load_run(path: Path) -> RetrievalRun | BilingualRetrievalRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("retrieval run must be a JSON object")
    if payload.get("schema_version") == "bilingual-retrieval-run-v1":
        return BilingualRetrievalRun.model_validate(payload)
    return RetrievalRun.model_validate(payload)


def _run_query(run: RetrievalRun | BilingualRetrievalRun) -> str:
    return run.original_query if isinstance(run, BilingualRetrievalRun) else run.query


if __name__ == "__main__":
    main()
