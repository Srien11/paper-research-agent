"""Thin LangChain adapters over the framework-independent research tools."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from paper_research_agent.agent.models import GetEvidenceInput, SearchCorpusInput
from paper_research_agent.agent.service import ResearchToolService


def build_langchain_tools(service: ResearchToolService) -> tuple[BaseTool, BaseTool]:
    """Return the fixed, read-only tool set exposed to a research Agent."""

    async def search_corpus(query: str, top_k: int = 10) -> dict[str, Any]:
        result = await service.search_corpus(SearchCorpusInput(query=query, top_k=top_k))
        return result.model_dump(mode="json")

    async def get_evidence(chunk_ids: tuple[str, ...]) -> dict[str, Any]:
        result = await service.get_evidence(GetEvidenceInput(chunk_ids=chunk_ids))
        return result.model_dump(mode="json")

    search_tool = StructuredTool.from_function(
        coroutine=search_corpus,
        name="search_corpus",
        description=(
            "Search the private local paper corpus for ranked evidence candidates. "
            "Returns provenance and rights metadata but no evidence body."
        ),
        args_schema=SearchCorpusInput,
    )
    evidence_tool = StructuredTool.from_function(
        coroutine=get_evidence,
        name="get_evidence",
        description=(
            "Read evidence bodies for explicit chunk IDs returned by search_corpus. "
            "This tool is local and read-only."
        ),
        args_schema=GetEvidenceInput,
    )
    return search_tool, evidence_tool
