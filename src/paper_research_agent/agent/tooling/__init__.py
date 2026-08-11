"""Risk-classified research tools beyond the core local RAG pair."""

from paper_research_agent.agent.tooling.catalog import (
    EXTENDED_TOOL_SPECS,
    ExtendedToolPolicy,
    ToolSpec,
)
from paper_research_agent.agent.tooling.contracts import TOOL_INPUT_SCHEMAS, ToolExecutionResult
from paper_research_agent.agent.tooling.factory import (
    ExtendedToolkitHandle,
    create_extended_research_toolkit,
)
from paper_research_agent.agent.tooling.registry import (
    RegisteredTool,
    ToolProvider,
    ToolRegistrySnapshot,
)

__all__ = [
    "EXTENDED_TOOL_SPECS",
    "TOOL_INPUT_SCHEMAS",
    "ExtendedToolPolicy",
    "ExtendedToolkitHandle",
    "RegisteredTool",
    "ToolExecutionResult",
    "ToolProvider",
    "ToolRegistrySnapshot",
    "ToolSpec",
    "create_extended_research_toolkit",
]
