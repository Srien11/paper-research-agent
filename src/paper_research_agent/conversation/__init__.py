"""Shared, route-agnostic conversation ledger and context resolution."""

from paper_research_agent.conversation.models import (
    ConversationCandidate,
    ConversationContextSnapshot,
    ConversationEpisode,
    ConversationResolution,
    ConversationTurn,
    TurnInterpretation,
)
from paper_research_agent.conversation.resolver import (
    build_conversation_context,
    fallback_resolution_from_context,
    resolution_from_interpretation,
    resolve_conversation_question,
)
from paper_research_agent.conversation.service import ConversationCoordinator
from paper_research_agent.conversation.store import (
    ConversationStore,
    InMemoryConversationStore,
    SQLiteConversationStore,
)

__all__ = [
    "ConversationCandidate",
    "ConversationContextSnapshot",
    "ConversationCoordinator",
    "ConversationEpisode",
    "ConversationResolution",
    "ConversationStore",
    "ConversationTurn",
    "InMemoryConversationStore",
    "SQLiteConversationStore",
    "TurnInterpretation",
    "build_conversation_context",
    "fallback_resolution_from_context",
    "resolution_from_interpretation",
    "resolve_conversation_question",
]
