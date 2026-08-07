from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.gold_dataset import GoldQuestion, load_gold_dataset
from paper_research_agent.evaluation.rag_gold_runner import (
    RAGJudgeResult,
    evaluate_rag_gold,
    write_rag_gold_report,
)
from paper_research_agent.web.runtime import RAGRuntime


class DashScopeRubricJudge:
    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self.model_id = model
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=httpx.Timeout(60)
        )

    async def score(
        self,
        question: GoldQuestion,
        answer: object,
        sources: Sequence[object],
    ) -> RAGJudgeResult:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": _judge_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        _judge_input(question, answer, sources),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "top_p": 0.7,
            "enable_thinking": False,
            "max_tokens": 1000,
        }
        last_error: Exception | None = None
        for _ in range(3):
            try:
                response = await self._client.post(self._endpoint, json=payload)
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                result = RAGJudgeResult.model_validate(json.loads(content))
                _validate_judge_ids(question, result)
                return result
            except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                last_error = error
        raise RuntimeError(f"rubric judge failed: {type(last_error).__name__}") from last_error

    async def aclose(self) -> None:
        await self._client.aclose()


def _judge_prompt() -> str:
    return (
        "You are a strict evidence-grounded RAG evaluator. Return exactly one JSON object with "
        "must_have, forbidden, supported_answer_claim_count, and "
        "citation_supported_answer_claim_count. For every supplied must-have claim, decide "
        "whether the system answer expresses it with all important scope qualifiers, and whether "
        "the answer's cited source excerpts support it. For every forbidden claim, decide whether "
        "the answer states or clearly implies it. Count an answer claim as supported only when its "
        "content is entailed by its cited excerpts; topical similarity is insufficient. Do not "
        "reward verbosity. Do not use outside knowledge. Return every rubric claim ID exactly once "
        "and no rationale or free-form fields."
    )


def _judge_input(question: GoldQuestion, answer: object, sources: Sequence[object]) -> dict[str, Any]:
    value = lambda source, name, default=None: (
        source.get(name, default) if isinstance(source, dict) else getattr(source, name, default)
    )
    claims = tuple(value(answer, "claims", ()))
    return {
        "question": question.question,
        "gold_must_have": [claim.model_dump(mode="json") for claim in question.must_have_claims],
        "gold_forbidden": [claim.model_dump(mode="json") for claim in question.forbidden_claims],
        "gold_evidence": [
            {"span_id": span.span_id, "quote": span.raw_quote}
            for span in question.evidence_spans
            if span.support_role != "distractor"
        ],
        "system_answer_claims": [
            {
                "index": index,
                "text": value(claim, "text", ""),
                "citation_ids": list(value(claim, "citation_ids", ())),
            }
            for index, claim in enumerate(claims)
        ],
        "system_sources": [
            {
                "citation_id": value(source, "citation_id", ""),
                "chunk_id": value(source, "chunk_id", ""),
                "excerpt": value(source, "excerpt", ""),
            }
            for source in sources
        ],
        "required_output": {
            "must_have": [
                {"claim_id": "each gold ID", "satisfied": False, "citation_supported": False}
            ],
            "forbidden": [{"claim_id": "each gold ID", "present": False}],
            "supported_answer_claim_count": 0,
            "citation_supported_answer_claim_count": 0,
        },
    }


def _validate_judge_ids(question: GoldQuestion, result: RAGJudgeResult) -> None:
    if {item.claim_id for item in result.must_have} != {
        claim.claim_id for claim in question.must_have_claims
    }:
        raise ValueError("rubric judge returned invalid must-have IDs")
    if {item.claim_id for item in result.forbidden} != {
        claim.claim_id for claim in question.forbidden_claims
    }:
        raise ValueError("rubric judge returned invalid forbidden IDs")


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


async def run(args: argparse.Namespace) -> None:
    _load_env_file(args.env_file)
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is unavailable")
    config = json.loads(args.answer_config.read_text(encoding="utf-8"))
    questions = load_gold_dataset(args.dataset)
    if args.task_type is not None:
        questions = tuple(
            question for question in questions if question.task_type == args.task_type
        )
    if args.answerable_only:
        questions = tuple(question for question in questions if question.answerable)
    if args.limit is not None:
        questions = questions[: args.limit]
    runtime = (
        await RAGRuntime.from_environment_with_agent()
        if RAGRuntime.research_agent_enabled_from_environment()
        else RAGRuntime.from_environment()
    )
    judge = DashScopeRubricJudge(
        api_key=api_key,
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        model=str(config["model"]),
    )
    revision_result = await asyncio.to_thread(
        subprocess.run,
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        result = await evaluate_rag_gold(
            runtime,
            judge,
            questions,
            args.output,
            evaluation_context={
                "model_id": config["model"],
                "judge_model_id": config["model"],
                "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "index_id": json.loads(args.index_manifest.read_text(encoding="utf-8"))["index_id"],
                "code_revision": revision_result.stdout.strip() or "unknown",
            },
        )
        write_rag_gold_report(result, args.report)
    finally:
        with suppress(Exception):
            await judge.aclose()
        with suppress(Exception):
            await runtime.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run private RAG silver-rubric diagnostics.")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/gold/rag-answer-candidates-v1.jsonl",
    )
    parser.add_argument(
        "--answer-config",
        type=Path,
        default=PROJECT_ROOT / "configs/answering/qwen-rag-v1.json",
    )
    parser.add_argument(
        "--index-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/indexes/retrieval-v1/manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/runs/rag-gold-live-v1.json",
    )
    parser.add_argument(
        "--report", type=Path, default=PROJECT_ROOT / "reports/RAG银标Rubric诊断-v1.md"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-type")
    parser.add_argument("--answerable-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))
    print(args.output)
    print(args.report)


if __name__ == "__main__":
    main()
