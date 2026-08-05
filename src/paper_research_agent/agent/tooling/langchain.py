"""LangChain adapters exposing the exact 19-tool extended registry."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from paper_research_agent.agent.tooling.catalog import EXTENDED_TOOL_SPECS
from paper_research_agent.agent.tooling.contracts import TOOL_INPUT_SCHEMAS
from paper_research_agent.agent.tooling.service import ExtendedResearchToolkit


def build_extended_langchain_tools(toolkit: ExtendedResearchToolkit) -> tuple[BaseTool, ...]:
    tools: list[BaseTool] = []
    for spec in EXTENDED_TOOL_SPECS:
        tools.append(
            StructuredTool.from_function(
                coroutine=_coroutine(toolkit, spec.name),
                name=spec.name,
                description=(
                    f"{spec.description} Risk={spec.risk}; timeout={spec.timeout_seconds}s; "
                    f"approval_required={spec.approval_required}."
                ),
                args_schema=TOOL_INPUT_SCHEMAS[spec.name],
            )
        )
    return tuple(tools)


def _coroutine(toolkit: ExtendedResearchToolkit, name: str) -> Any:
    async def invoke(**kwargs: Any) -> dict[str, Any]:
        result = await toolkit.execute(name, kwargs)
        return result.model_dump(mode="json")

    return invoke
