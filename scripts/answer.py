from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.answering.audit import SQLiteAnswerAuditLogger
from paper_research_agent.answering.config import load_answering_config
from paper_research_agent.answering.dashscope import (
    AsyncAnswerGenerator,
    DashScopeAnswerGenerator,
    UnavailableAnswerGenerator,
)
from paper_research_agent.answering.models import AnswerRequest, RAGAnswer
from paper_research_agent.answering.service import AnswerAuditLogger, answer_context
from paper_research_agent.context.models import AssembledContext

if TYPE_CHECKING:
    from paper_research_agent.answering.config import AnsweringConfig

DEFAULT_CONFIG = PROJECT_ROOT / "configs/answering/qwen-rag-v1.json"
DEFAULT_AUDIT_PATH = PROJECT_ROOT / "data/runtime/answer-audit-v1.sqlite3"


async def run_answer(
    context_path: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_path: Path | None = None,
    generator: AsyncAnswerGenerator | None = None,
    audit: AnswerAuditLogger | None = None,
) -> RAGAnswer:
    context = AssembledContext.model_validate_json(context_path.read_text(encoding="utf-8"))
    request = AnswerRequest(context=context)
    owned_generator = generator is None
    if generator is None:
        config: AnsweringConfig = load_answering_config(config_path)
        try:
            generator = DashScopeAnswerGenerator(config)
        except RuntimeError:
            generator = UnavailableAnswerGenerator(config.model, config.prompt_version)
    try:
        result = await answer_context(request, generator, audit=audit)
    finally:
        if owned_generator:
            close = getattr(generator, "aclose", None)
            if close is not None:
                await close()
    payload = result.model_dump_json(indent=2)
    if output_path is None:
        print(payload)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one validated private-research RAG answer from AssembledContext."
    )
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT_PATH)
    args = parser.parse_args()
    try:
        audit = _optional_audit(args.audit_path)
        asyncio.run(
            run_answer(
                args.context,
                config_path=args.config,
                output_path=args.output,
                audit=audit,
            )
        )
    except Exception as error:  # noqa: BLE001 - sanitize every CLI failure without echoing evidence
        parser.exit(1, f"answer failed: {type(error).__name__}\n")


def _optional_audit(path: Path) -> AnswerAuditLogger | None:
    try:
        return SQLiteAnswerAuditLogger(path)
    except (OSError, sqlite3.Error):
        return None


if __name__ == "__main__":
    main()
