"""Async orchestration boundary for route-agnostic conversation state."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from paper_research_agent.conversation.models import (
    ConversationContextSnapshot,
    ConversationResolution,
    ConversationStatus,
    ConversationTurn,
)
from paper_research_agent.conversation.resolver import (
    build_conversation_context,
    resolve_conversation_question,
)
from paper_research_agent.conversation.store import ConversationStore


class ConversationCoordinator:
    def __init__(self, store: ConversationStore, *, history_limit: int = 500) -> None:
        if history_limit <= 0 or history_limit > 2_000:
            raise ValueError("conversation history limit must be between 1 and 2000")
        self.store = store
        self.history_limit = history_limit

    async def begin(
        self, conversation_id: str, question: str
    ) -> tuple[ConversationTurn, ConversationResolution]:
        turn, snapshot = await self.prepare(conversation_id, question)
        history = await asyncio.to_thread(
            self.store.history, conversation_id, limit=self.history_limit
        )
        resolution = resolve_conversation_question(
            question,
            history,
            episodes=snapshot.episodes,
        )
        return turn, resolution

    async def prepare(
        self, conversation_id: str, question: str
    ) -> tuple[ConversationTurn, ConversationContextSnapshot]:
        turn = await asyncio.to_thread(self.store.begin_turn, conversation_id, question)
        history = await asyncio.to_thread(
            self.store.history, conversation_id, limit=self.history_limit
        )
        episodes = await asyncio.to_thread(self.store.episodes, conversation_id, limit=100)
        snapshot = build_conversation_context(question, history, episodes=episodes)
        return turn, snapshot

    async def complete(
        self,
        turn_id: str,
        *,
        route: str,
        status: ConversationStatus,
        resolution: ConversationResolution,
        assistant_summary: str | None = None,
        source_ids: Sequence[str] = (),
    ) -> bool:
        inherited = any(
            candidate.route is not None and candidate.route != route
            for candidate in resolution.selected_candidates
        )
        finalized = resolution.model_copy(update={"inherited_across_route": inherited})
        return await asyncio.to_thread(
            self.store.complete_turn,
            turn_id,
            route=route,
            status=status,
            resolution=finalized,
            assistant_summary=assistant_summary,
            source_ids=source_ids,
        )

    async def clear(self, conversation_id: str) -> int:
        return await asyncio.to_thread(self.store.clear, conversation_id)
