"""Construct a closable production extended research toolkit."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from paper_research_agent.agent.observability import AgentEventSink
from paper_research_agent.agent.service import ResearchToolService
from paper_research_agent.agent.tooling.analysis import AnalysisResearchTools
from paper_research_agent.agent.tooling.approval import ApprovalManager
from paper_research_agent.agent.tooling.content import ContentResearchTools
from paper_research_agent.agent.tooling.local import LocalResearchTools
from paper_research_agent.agent.tooling.scholarly import ScholarlyResearchTools
from paper_research_agent.agent.tooling.service import ExtendedResearchToolkit
from paper_research_agent.agent.tooling.workspace import WorkspaceResearchTools
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.ingestion.models import DocumentElement, SectionRecord
from paper_research_agent.models import FrozenPaper


class AsyncClosable(Protocol):
    async def aclose(self) -> None: ...


@dataclass
class ExtendedToolkitHandle:
    toolkit: ExtendedResearchToolkit
    client: httpx.AsyncClient
    mcp_manager: AsyncClosable | None = None

    async def aclose(self) -> None:
        failures: list[Exception] = []
        try:
            await self.client.aclose()
        except Exception as exc:  # noqa: BLE001 - still close the independent MCP lifecycle
            failures.append(exc)
        if self.mcp_manager is not None:
            try:
                await self.mcp_manager.aclose()
            except Exception as exc:  # noqa: BLE001 - both resources must be attempted
                failures.append(exc)
        if failures:
            raise RuntimeError("extended toolkit shutdown failed") from None


def create_extended_research_toolkit(
    *,
    project_root: Path,
    rag: ResearchToolService,
    chunks: Sequence[EvidenceChunk],
    storage_classes: Mapping[str, str],
    papers: Sequence[FrozenPaper] = (),
    sections: Sequence[SectionRecord] = (),
    elements: Sequence[DocumentElement] = (),
    event_sink: AgentEventSink | None = None,
    semantic_scholar_api_key: str | None = None,
) -> ExtendedToolkitHandle:
    approvals = ApprovalManager()
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(10),
        follow_redirects=False,
        headers={"User-Agent": "paper-research-agent/0.1 (local research)"},
    )
    scholarly = ScholarlyResearchTools(
        client,
        api_key=semantic_scholar_api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
    )
    toolkit = ExtendedResearchToolkit(
        local=LocalResearchTools(
            chunks=chunks,
            storage_classes=storage_classes,
            papers=papers,
            sections=sections,
        ),
        content=ContentResearchTools(elements=elements, chunks=chunks),
        analysis=AnalysisResearchTools(chunks=chunks),
        scholarly=scholarly,
        workspace=WorkspaceResearchTools(
            project_root,
            approvals=approvals,
            source_chunk_ids=frozenset(chunk.chunk_id for chunk in chunks),
        ),
        rag=rag,
        event_sink=event_sink,
        approvals=approvals,
    )
    return ExtendedToolkitHandle(toolkit=toolkit, client=client)
