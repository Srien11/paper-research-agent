from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
from contextlib import suppress
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.agent.dynamic.memory import LangChainMemoryProposer
from paper_research_agent.agent.dynamic.router import LangChainToolRouter
from paper_research_agent.answering.config import load_answering_config
from paper_research_agent.evaluation.tool_routing import load_tool_routing_dataset
from paper_research_agent.evaluation.tool_routing_runner import (
    evaluate_tool_routing,
    write_tool_routing_report,
)
from paper_research_agent.retrieval.query_rewrite import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_BASE_URL_ENV,
)


async def run(args: argparse.Namespace) -> None:
    config = load_answering_config(args.answer_config)
    api_key = os.getenv(DEFAULT_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"环境变量 {DEFAULT_API_KEY_ENV} 未配置")
    base_url = (os.getenv(DEFAULT_BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")
    model = ChatOpenAI(
        model=config.model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=0,
        top_p=config.top_p,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
        extra_body={"enable_thinking": config.enable_thinking},
    )
    try:
        cases = load_tool_routing_dataset(args.dataset)
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("limit must be positive")
            cases = cases[: args.limit]
        result = await evaluate_tool_routing(
            LangChainToolRouter(model),
            LangChainMemoryProposer(model),
            cases,
            args.output,
            evaluation_context={
                "model_id": config.model,
                "answer_config_sha256": _sha256(args.answer_config),
                "dataset_sha256": _sha256(args.dataset),
                "temperature": 0,
                "top_p": config.top_p,
                "timeout_seconds": config.timeout_seconds,
                "max_retries": config.max_retries,
                "enable_thinking": config.enable_thinking,
            },
        )
        write_tool_routing_report(result, args.report)
    finally:
        client = getattr(model, "root_async_client", None)
        close_client = getattr(client, "close", None)
        if close_client is not None:
            with suppress(Exception):
                await close_client()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run privacy-safe real-model tool-routing evaluation without executing tools."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "evaluation/datasets/tool-routing-v2.jsonl",
    )
    parser.add_argument(
        "--answer-config",
        type=Path,
        default=PROJECT_ROOT / "configs/answering/qwen-rag-v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/runs/tool-routing-live-v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/工具路由真实模型评测-v1.md",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as error:  # noqa: BLE001 - never print provider or dataset payloads
        parser.exit(1, f"Tool-routing evaluation failed: {type(error).__name__}\n")
    print(args.output)
    print(args.report)


if __name__ == "__main__":
    main()
