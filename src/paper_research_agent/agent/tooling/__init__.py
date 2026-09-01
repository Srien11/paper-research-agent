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
from paper_research_agent.agent.tooling.scholarly_providers import (
    OfflineScholarlyProvider,
    ScholarlyProvider,
    ScholarlyProviderRegistry,
    ScholarlyProviderResult,
)

__all__ = [
    "EXTENDED_TOOL_SPECS",
    "TOOL_INPUT_SCHEMAS",
    "ExtendedToolPolicy",
    "ExtendedToolkitHandle",
    "OfflineScholarlyProvider",
    "RegisteredTool",
    "ScholarlyProvider",
    "ScholarlyProviderRegistry",
    "ScholarlyProviderResult",
    "ToolExecutionResult",
    "ToolProvider",
    "ToolRegistrySnapshot",
    "ToolSpec",
    "create_extended_research_toolkit",
]
