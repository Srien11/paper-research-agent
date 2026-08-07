from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.gold_generation import (
    GeneratedCandidate,
    SourceEvidence,
    build_gold_question,
)
from paper_research_agent.evaluation.gold_selection import (
    CandidateBlueprint,
    build_candidate_blueprint,
)

DEFAULT_BUILD = (
    PROJECT_ROOT
    / "data/processed/llm-eval-reliability-v1.0.0-2026-07-26"
    / "build_70e844af8c83c5d89f757a8cd34af784"
)
DEFAULT_CHUNKS = PROJECT_ROOT / "data/processed/chunks/chunks.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluations/gold/rag-answer-candidates-v1.jsonl"
DEFAULT_BLUEPRINT = PROJECT_ROOT / "data/evaluations/gold/question-blueprint-v1.jsonl"
DEFAULT_CACHE = PROJECT_ROOT / "data/evaluations/gold/candidate-cache-v1"
MODEL_CONFIG = PROJECT_ROOT / "configs/answering/qwen-rag-v1.json"


class CandidateGenerationError(RuntimeError):
    pass


class DraftModel:
    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self._model = model
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(60),
        )

    async def generate(
        self,
        blueprint: CandidateBlueprint,
        evidence: list[SourceEvidence],
        titles: dict[str, str],
    ) -> GeneratedCandidate:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        _generation_input(blueprint, evidence, titles),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "top_p": 0.7,
            "enable_thinking": False,
            "max_tokens": 1200,
        }
        last_error: Exception | None = None
        for _ in range(3):
            try:
                response = await self._client.post(self._endpoint, json=payload)
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                draft = GeneratedCandidate.model_validate(json.loads(content))
                _validate_generated_draft(blueprint, draft, titles)
                return draft
            except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                last_error = error
        raise CandidateGenerationError(
            f"candidate {blueprint.case_id} failed after retries: {type(last_error).__name__}"
        ) from last_error

    async def aclose(self) -> None:
        await self._client.aclose()


async def run(args: argparse.Namespace) -> None:
    _load_env_file(args.env_file)
    corpus_dir = Path(os.environ.get("PRA_CORPUS_DIR", ""))
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    base_url = os.environ.get(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    if not corpus_dir.is_dir():
        raise RuntimeError("PRA_CORPUS_DIR is unavailable")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is unavailable")

    papers = _load_papers(corpus_dir)
    blueprint = build_candidate_blueprint(papers, seed=args.seed)
    if args.limit is not None:
        blueprint = blueprint[: args.limit]
    _write_jsonl(args.blueprint_output, [row.model_dump(mode="json") for row in blueprint])

    chunks_by_element = _chunk_projection(args.chunks)
    evidence_pool = _load_evidence_pool(args.elements, chunks_by_element)
    titles = {str(row["corpus_id"]): str(row["title"]) for row in papers}
    selected_by_case = {
        slot.case_id: _select_evidence(slot, evidence_pool) for slot in blueprint
    }
    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    model = DraftModel(api_key=api_key, base_url=base_url, model=str(config["model"]))
    semaphore = asyncio.Semaphore(args.concurrency)

    async def generate_one(slot: CandidateBlueprint) -> object:
        selected = selected_by_case[slot.case_id]
        cache_path = args.cache_dir / f"{slot.case_id}.json"
        if cache_path.exists():
            draft = GeneratedCandidate.model_validate_json(cache_path.read_text(encoding="utf-8"))
            _validate_generated_draft(slot, draft, titles)
        else:
            async with semaphore:
                draft = await model.generate(slot, selected, titles)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(draft.model_dump_json(), encoding="utf-8")
        return build_gold_question(
            slot,
            selected,
            draft,
            corpus_version=_corpus_version(papers),
            knowledge_cutoff=date.fromisoformat(args.knowledge_cutoff),
        )

    try:
        questions = await asyncio.gather(*(generate_one(slot) for slot in blueprint))
    finally:
        await model.aclose()
    normalized = [_normalized_question(item.question) for item in questions]
    if len(normalized) != len(set(normalized)):
        raise CandidateGenerationError("generated candidate questions are not unique")
    _write_jsonl(args.output, [item.model_dump(mode="json") for item in questions])
    print(
        json.dumps(
            {
                "candidate_count": len(questions),
                "answerable_count": sum(item.answerable for item in questions),
                "unanswerable_count": sum(not item.answerable for item in questions),
                "annotation_status": "silver_generated",
                "output": args.output.name,
            },
            ensure_ascii=False,
        )
    )


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _load_papers(corpus_dir: Path) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    for name in ("core_frozen.jsonl", "challenge_frozen.jsonl"):
        path = next(iter(corpus_dir.rglob(name)), None)
        if path is None:
            raise RuntimeError(f"frozen corpus manifest is missing: {name}")
        papers.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return papers


def _corpus_version(papers: list[dict[str, Any]]) -> str:
    versions = {str(row.get("corpus_version", "")).strip() for row in papers}
    versions.discard("")
    if len(versions) != 1:
        raise RuntimeError("frozen manifests do not share one corpus version")
    return versions.pop()


def _chunk_projection(path: Path) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        chunk_id = str(chunk["chunk_id"])
        for element_id in chunk.get("element_ids", []):
            if len(mapping[str(element_id)]) < 20:
                mapping[str(element_id)].append(chunk_id)
    return {key: tuple(values) for key, values in mapping.items()}


def _load_evidence_pool(
    path: Path,
    chunks_by_element: dict[str, tuple[str, ...]],
) -> dict[str, list[dict[str, Any]]]:
    pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    allowed_types = {"paragraph", "figure_caption", "table_caption"}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        element = json.loads(line)
        raw_text = element.get("raw_text")
        element_id = str(element.get("element_id", ""))
        if (
            element.get("element_type") not in allowed_types
            or not isinstance(raw_text, str)
            or not 40 <= len(raw_text) <= 1600
            or element_id not in chunks_by_element
        ):
            continue
        item = {
            "paper_id": str(element["corpus_id"]),
            "evidence_version_id": str(element["asset_id"]),
            "page": int(element["page_number"]),
            "element_id": element_id,
            "element_type": str(element["element_type"]),
            "raw_quote": raw_text,
            "projected_chunk_ids": chunks_by_element[element_id],
        }
        pool[item["paper_id"]].append(item)
    return pool


def _select_evidence(
    slot: CandidateBlueprint,
    pool: dict[str, list[dict[str, Any]]],
) -> list[SourceEvidence]:
    selected: list[dict[str, Any]] = []
    per_paper = 2 if len(slot.target_paper_ids) > 1 else 4
    for paper_id in slot.target_paper_ids:
        ranked = sorted(
            pool.get(paper_id, []),
            key=lambda item: (-_evidence_score(slot, item), item["page"], item["element_id"]),
        )
        for item in ranked[:per_paper]:
            if item not in selected:
                selected.append(item)
    if not selected:
        raise CandidateGenerationError(f"candidate {slot.case_id} has no source evidence")
    role = "required" if slot.answerable else "distractor"
    return [
        SourceEvidence(
            span_id=f"S{index:03d}",
            paper_id=item["paper_id"],
            evidence_version_id=item["evidence_version_id"],
            page=item["page"],
            element_id=item["element_id"],
            raw_quote=item["raw_quote"],
            span_hash=hashlib.sha256(item["raw_quote"].encode()).hexdigest(),
            support_role=role,
            projected_chunk_ids=item["projected_chunk_ids"],
        )
        for index, item in enumerate(selected, start=1)
    ]


def _evidence_score(slot: CandidateBlueprint, item: dict[str, Any]) -> int:
    text = str(item["raw_quote"]).lower()
    element_type = item["element_type"]
    score = min(len(text), 1000) // 20
    if slot.evidence_source == "figure_caption" and element_type == "figure_caption":
        score += 200
    if slot.evidence_source == "table_appendix" and element_type == "table_caption":
        score += 200
    if slot.task_type == "experimental_result" and re.search(r"\d|%|result|accuracy|score", text):
        score += 100
    if slot.task_type == "method_mechanism" and re.search(r"method|approach|framework|we (?:use|propose)", text):
        score += 80
    if slot.task_type == "definition_scope" and re.search(r"define|consist|include|benchmark|evaluate", text):
        score += 60
    if re.search(r"references|acknowledg", text):
        score -= 200
    return score


def _system_prompt() -> str:
    return (
        "You create SILVER DRAFT evaluation questions for a private paper RAG benchmark. "
        "Return exactly one JSON object with question, must_have_claims, forbidden_claims, "
        "and optional unanswerable_reason. Every must_have claim must be atomic and fully "
        "supported by the supplied evidence span IDs. A forbidden claim is a plausible, "
        "high-risk contradiction, not merely extra information. Never mention corpus IDs, "
        "span IDs, evaluation labels, or paper titles verbatim in the question. Do not use "
        "outside knowledge. This output is a candidate for human review, not gold truth."
    )


def _generation_input(
    slot: CandidateBlueprint,
    evidence: list[SourceEvidence],
    titles: dict[str, str],
) -> dict[str, object]:
    return {
        "case_id": slot.case_id,
        "answerable": slot.answerable,
        "task_type": slot.task_type,
        "language": slot.language,
        "difficulty": slot.difficulty,
        "unanswerable_taxonomy": slot.unanswerable_reason,
        "required_output": {
            "question": "string",
            "must_have_claims": [
                {"claim_id": "M1", "text": "atomic claim", "span_ids": ["S001"]}
            ],
            "forbidden_claims": [
                {"claim_id": "F1", "text": "plausible false claim", "span_ids": []}
            ],
            "unanswerable_reason": (
                "null for answerable; concise reason for unanswerable"
            ),
        },
        "rules": (
            "For unanswerable cases return no must-have claims and construct a difficult "
            "near-neighbor question that the supplied evidence cannot support."
        ),
        "evidence": [
            {
                "span_id": span.span_id,
                "paper_title": titles[span.paper_id],
                "page": span.page,
                "quote": span.raw_quote,
            }
            for span in evidence
        ],
    }


def _validate_generated_draft(
    slot: CandidateBlueprint,
    draft: GeneratedCandidate,
    titles: dict[str, str],
) -> None:
    if slot.answerable != bool(draft.must_have_claims):
        raise ValueError("generated answerability does not match the blueprint")
    if slot.answerable and draft.unanswerable_reason is not None:
        raise ValueError("answerable draft contains an unanswerable reason")
    if not slot.answerable and draft.unanswerable_reason is None:
        raise ValueError("unanswerable draft lacks a reason")
    if re.search(r"\b[CT]\d{3}\b", draft.question):
        raise ValueError("generated question leaks a corpus ID")
    normalized = _normalized_question(draft.question)
    for paper_id in slot.target_paper_ids:
        title = _normalized_question(titles[paper_id])
        similarity = SequenceMatcher(a=normalized, b=title).ratio() if title else 0.0
        if title and (normalized == title or similarity >= 0.92):
            raise ValueError("generated question copies a paper title")


def _normalized_question(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.lower(), flags=re.UNICODE))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate private silver gold-set candidates.")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--elements", type=Path, default=DEFAULT_BUILD / "elements.jsonl")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blueprint-output", type=Path, default=DEFAULT_BLUEPRINT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--knowledge-cutoff", default="2026-07-26")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=3, choices=range(1, 6))
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
