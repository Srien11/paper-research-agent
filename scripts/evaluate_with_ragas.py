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

from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.evaluation.gold_dataset import (
    GoldQuestion,
    dataset_summary,
    load_gold_dataset,
)
from paper_research_agent.evaluation.ragas_eval import (
    RagasEvaluationConfig,
    RagasMetricSuite,
    evaluate_ragas_gold,
    write_ragas_report,
)
from paper_research_agent.evaluation.silver_quality import (
    SilverQuarantineManifest,
    SilverQuarantineReport,
    apply_silver_quarantine,
)

DEFAULT_DATASET = PROJECT_ROOT / "data/evaluations/gold/rag-answer-candidates-v1.jsonl"
DEFAULT_QUARANTINE_MANIFEST = PROJECT_ROOT / "configs/evaluation/ragas-silver-quarantine-v1.json"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _selected_questions(
    args: argparse.Namespace,
) -> tuple[tuple[GoldQuestion, ...], SilverQuarantineReport | None, Path | None]:
    questions = tuple(load_gold_dataset(args.dataset))
    quarantine_report: SilverQuarantineReport | None = None
    quarantine_path = _quarantine_path(args)
    if quarantine_path is not None:
        manifest = SilverQuarantineManifest.from_path(quarantine_path)
        questions, quarantine_report = apply_silver_quarantine(questions, manifest)
    if args.task_type is not None:
        questions = tuple(
            question for question in questions if question.task_type == args.task_type
        )
    if args.answerable_only:
        questions = tuple(question for question in questions if question.answerable)
    if args.offset:
        questions = questions[args.offset :]
    if args.limit is not None:
        questions = questions[: args.limit]
    return questions, quarantine_report, quarantine_path


def _quarantine_path(args: argparse.Namespace) -> Path | None:
    if bool(args.include_quarantined):
        return None
    if args.quarantine_manifest is not None:
        return Path(args.quarantine_manifest)
    try:
        is_default_dataset = args.dataset.resolve() == DEFAULT_DATASET.resolve()
    except OSError:
        is_default_dataset = False
    return DEFAULT_QUARANTINE_MANIFEST if is_default_dataset else None


def _dependency_check(expected_version: str) -> str:
    try:
        import ragas
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            'Ragas evaluation dependencies are unavailable; install ".[ragas-eval]"'
        ) from error
    del AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    if ragas.__version__ != expected_version:
        raise RuntimeError(
            f"Ragas version mismatch: expected {expected_version}, got {ragas.__version__}"
        )
    return ragas.__version__


def _api_key() -> str:
    return os.getenv("PRA_RAGAS_API_KEY", "").strip() or os.getenv("DASHSCOPE_API_KEY", "").strip()


def _base_url() -> str:
    return (
        os.getenv("PRA_RAGAS_BASE_URL", "").strip()
        or os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).strip()
    )


def _effective_config(config: RagasEvaluationConfig) -> RagasEvaluationConfig:
    overrides: dict[str, object] = {}
    judge_model = os.getenv("PRA_RAGAS_JUDGE_MODEL", "").strip()
    embedding_model = os.getenv("PRA_RAGAS_EMBEDDING_MODEL", "").strip()
    if judge_model:
        overrides["judge_model"] = judge_model
    if embedding_model:
        overrides["embedding_model"] = embedding_model
    return config.model_copy(update=overrides)


def _runtime_chunks_path() -> Path:
    project_root = Path(os.getenv("PRA_PROJECT_ROOT", str(PROJECT_ROOT))).resolve()
    configured = os.getenv("PRA_CHUNKS_PATH", "").strip()
    if not configured:
        return project_root / "data/processed/chunks/chunks.jsonl"
    path = Path(configured)
    return path if path.is_absolute() else project_root / path


def _load_chunk_text_by_id(path: Path) -> dict[str, str]:
    chunks = [
        EvidenceChunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = {chunk.chunk_id: chunk.text for chunk in chunks}
    if len(texts) != len(chunks):
        raise ValueError("evaluation chunks contain duplicate chunk IDs")
    return texts


def _print_batch_progress(completed: int, total: int) -> None:
    print(
        json.dumps({"event": "ragas_batch_completed", "completed": completed, "total": total}),
        flush=True,
    )


def _safe_check_output(
    config: RagasEvaluationConfig,
    questions: tuple[GoldQuestion, ...],
    *,
    credentials_configured: bool,
    quarantine_report: SilverQuarantineReport | None,
) -> dict[str, object]:
    summary = dataset_summary(questions)
    return {
        "status": "ready",
        "ragas_version": config.ragas_version,
        "judge_model": config.judge_model,
        "embedding_model": config.embedding_model,
        "max_concurrency": config.max_concurrency,
        "enabled_metrics": list(config.enabled_metrics),
        "release_gate_enabled": config.release_gate_enabled,
        "credentials_configured": credentials_configured,
        "dataset": summary,
        "silver_quarantine": (
            quarantine_report.model_dump(mode="json") if quarantine_report is not None else None
        ),
    }


async def run(args: argparse.Namespace) -> None:
    _load_env_file(args.env_file)
    config = _effective_config(RagasEvaluationConfig.from_path(args.config))
    _dependency_check(config.ragas_version)
    questions, quarantine_report, quarantine_path = _selected_questions(args)
    if args.check:
        print(
            json.dumps(
                _safe_check_output(
                    config,
                    questions,
                    credentials_configured=bool(_api_key()),
                    quarantine_report=quarantine_report,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not questions:
        raise RuntimeError("no Ragas evaluation questions selected")
    api_key = _api_key()
    if not api_key:
        raise RuntimeError("PRA_RAGAS_API_KEY or DASHSCOPE_API_KEY is unavailable")

    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory

    from paper_research_agent.web.runtime import RAGRuntime

    client = AsyncOpenAI(api_key=api_key, base_url=_base_url())
    llm = llm_factory(config.judge_model, provider="openai", client=client)
    embeddings = (
        embedding_factory(
            "openai",
            model=config.embedding_model,
            client=client,
        )
        if config.embedding_model is not None
        else None
    )
    scorer = RagasMetricSuite(llm=llm, embeddings=embeddings, config=config)
    chunk_text_by_id = _load_chunk_text_by_id(_runtime_chunks_path())
    runtime = (
        await RAGRuntime.from_environment_with_agent()
        if RAGRuntime.research_agent_enabled_from_environment()
        else RAGRuntime.from_environment()
    )
    revision_result = await asyncio.to_thread(
        subprocess.run,
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status_result = await asyncio.to_thread(
        subprocess.run,
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        evaluation_context: dict[str, object] = {
            "ragas_version": config.ragas_version,
            "judge_model": config.judge_model,
            "embedding_model": config.embedding_model,
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "code_revision": revision_result.stdout.strip() or "unknown",
            "working_tree_dirty": bool(status_result.stdout.strip()),
            "evaluation_input_version": "full-chunk-must-have-supports-v1",
            "case_concurrency": config.max_concurrency,
        }
        if quarantine_report is not None and quarantine_path is not None:
            evaluation_context.update(
                {
                    "silver_quarantine_manifest_sha256": hashlib.sha256(
                        quarantine_path.read_bytes()
                    ).hexdigest(),
                    "silver_quarantined_question_count": (
                        quarantine_report.excluded_question_count
                    ),
                    "silver_quarantined_question_ids": list(
                        quarantine_report.excluded_question_ids
                    ),
                }
            )
        result = await evaluate_ragas_gold(
            runtime,
            scorer,
            questions,
            args.output,
            chunk_text_by_id=chunk_text_by_id,
            case_concurrency=config.max_concurrency,
            progress_callback=_print_batch_progress,
            evaluation_context=evaluation_context,
        )
        write_ragas_report(
            result,
            args.report,
            thresholds=config.diagnostic_thresholds,
            release_gate_enabled=config.release_gate_enabled,
        )
    finally:
        with suppress(Exception):
            await runtime.aclose()
        with suppress(Exception):
            await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run privacy-safe Ragas 0.4.3 evaluation over the live paper RAG Agent."
    )
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/evaluation/ragas-v1.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        "--quarantine-manifest",
        type=Path,
        help=(
            "Body-free silver quarantine manifest. The project default is applied "
            "automatically to the default dataset."
        ),
    )
    parser.add_argument(
        "--include-quarantined",
        action="store_true",
        help="Explicitly include quarantined silver cases for diagnostics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/runs/ragas-gold-v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/Ragas论文研究Agent评测-v1.md",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--task-type")
    parser.add_argument("--answerable-only", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only validate dependencies, configuration, and dataset without running the Agent.",
    )
    args = parser.parse_args()
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.include_quarantined and args.quarantine_manifest is not None:
        parser.error("--include-quarantined cannot be combined with --quarantine-manifest")
    asyncio.run(run(args))
    if not args.check:
        print(args.output)
        print(args.report)


if __name__ == "__main__":
    main()
