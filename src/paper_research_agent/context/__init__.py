"""Context assembly contracts and utilities."""

from paper_research_agent.context.assembler import assemble_context
from paper_research_agent.context.budget import ContextBudgetExceeded
from paper_research_agent.context.models import (
    AssembledContext,
    CitationRef,
    ContextEvidence,
    ContextMemoryTurn,
    ContextRequest,
    PromptMessage,
)

__all__ = [
    "AssembledContext",
    "CitationRef",
    "ContextBudgetExceeded",
    "ContextEvidence",
    "ContextMemoryTurn",
    "ContextRequest",
    "PromptMessage",
    "assemble_context",
]
