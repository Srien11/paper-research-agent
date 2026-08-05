"""Read tables, figures, and equations from local parsed artifacts."""

from __future__ import annotations

from collections.abc import Sequence

from paper_research_agent.agent.tooling.contracts import ElementLookupInput, ToolExecutionResult
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.ingestion.models import DocumentElement


class ContentResearchTools:
    def __init__(self, *, elements: Sequence[DocumentElement], chunks: Sequence[EvidenceChunk]):
        self._elements = tuple(elements)
        self._figures = tuple(
            (chunk.corpus_id, chunk.figure)
            for chunk in chunks
            if chunk.evidence_type == "figure_summary" and chunk.figure is not None
        )

    def extract_table(self, request: ElementLookupInput) -> ToolExecutionResult:
        return self._elements_result("extract_table", request, {"table", "table_caption"})

    def extract_equation(self, request: ElementLookupInput) -> ToolExecutionResult:
        return self._elements_result("extract_equation", request, {"formula"})

    def inspect_figure(self, request: ElementLookupInput) -> ToolExecutionResult:
        figures = [
            figure
            for corpus_id, figure in self._figures
            if corpus_id == request.corpus_id
            and (request.page is None or figure.page_number == request.page)
            and (request.label is None or request.label.casefold() in figure.figure_name.casefold())
        ][: request.limit]
        items = tuple(
            {
                "figure_id": figure.figure_id,
                "figure_name": figure.figure_name,
                "page_number": figure.page_number,
                "caption": figure.caption,
                "figure_type": figure.figure_type,
                "summary": figure.summary,
                "key_findings": figure.key_findings,
                "recognition_confidence": figure.recognition_confidence,
                "model_id": figure.model_id,
                "prompt_version": figure.prompt_version,
            }
            for figure in figures
        )
        return ToolExecutionResult(
            tool_name="inspect_figure",
            status="ok" if items else "not_found",
            items=items,
            summary={"count": len(items)},
        )

    def _elements_result(
        self,
        name: str,
        request: ElementLookupInput,
        types: set[str],
    ) -> ToolExecutionResult:
        matched = [
            element
            for element in self._elements
            if element.corpus_id == request.corpus_id
            and element.element_type in types
            and (request.page is None or element.page_number == request.page)
            and (
                request.label is None
                or request.label.casefold() in element.normalized_text.casefold()
            )
        ][: request.limit]
        items = tuple(
            {
                "element_id": element.element_id,
                "corpus_id": element.corpus_id,
                "section_id": element.section_id,
                "page_number": element.page_number,
                "element_type": element.element_type,
                "text": element.normalized_text,
                "text_sha256": element.normalized_text_sha256,
                "content_origin": element.content_origin,
            }
            for element in matched
        )
        return ToolExecutionResult(
            tool_name=name,
            status="ok" if items else "not_found",
            items=items,
            summary={"count": len(items)},
        )
