"""Unified context hydration for the main Agent before any child graph."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from paper_research_agent.agent.orchestrator.models import (
    AgentContextEnvelope,
    ContextMessage,
    ConversationWorkspace,
    MainAgentRequest,
    RecalledContext,
)
from paper_research_agent.conversation.models import ConversationTurn
from paper_research_agent.conversation.store import ConversationStore

MAX_RECENT_TURNS = 6
MAX_RECALLED_TURNS = 5
MAX_LONG_TERM_MEMORIES = 5

_RECENT_MESSAGE_CHARS = 12_000
_RECENT_MESSAGE_LIMIT = 12
_RECALLED_CHARS = 6_000
_MEMORY_CHARS = 5_000

_ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]{1,}")
_CHINESE_RUN = re.compile(r"[一-鿿]{2,}")


class LongTermMemoryProvider(Protocol):
    async def search(self, query: str, *, limit: int = 5) -> tuple[dict[str, object], ...]: ...


class ContextHydrator:
    """Assembles the immutable AgentContextEnvelope before interpretation or routing."""

    def __init__(
        self,
        store: ConversationStore,
        *,
        memory_provider: LongTermMemoryProvider | None = None,
        recent_turns: int = MAX_RECENT_TURNS,
        recalled_turns: int = MAX_RECALLED_TURNS,
        recalled_memories: int = MAX_LONG_TERM_MEMORIES,
        history_limit: int = 500,
    ) -> None:
        if recent_turns <= 0 or recent_turns > 12:
            raise ValueError("recent_turns must be between 1 and 12")
        if recalled_turns <= 0 or recalled_turns > 10:
            raise ValueError("recalled_turns must be between 1 and 10")
        if recalled_memories <= 0 or recalled_memories > 10:
            raise ValueError("recalled_memories must be between 1 and 10")
        if history_limit <= 0 or history_limit > 2_000:
            raise ValueError("history_limit must be between 1 and 2000")
        self.store = store
        self.memory_provider = memory_provider
        self.recent_turns = recent_turns
        self.recalled_turns = recalled_turns
        self.recalled_memories = recalled_memories
        self.history_limit = history_limit

    async def hydrate(
        self,
        request: MainAgentRequest,
        workspace: ConversationWorkspace,
        *,
        turn_id: str,
    ) -> AgentContextEnvelope:
        recent_turns = await asyncio.to_thread(
            self.store.recent, request.conversation_id, limit=self.recent_turns
        )
        history = await asyncio.to_thread(
            self.store.history, request.conversation_id, limit=self.history_limit
        )
        recall_query = self._recall_query(request.message, workspace)
        ranked = _rank_history(recall_query, history)
        recent_ids = {turn.turn_id for turn in recent_turns}
        recalled_turns = tuple(
            (turn, score)
            for turn, score in ranked
            if turn.turn_id not in recent_ids
        )[: self.recalled_turns]
        memories = await self._recall_memories(recall_query)
        recent_messages = _context_messages(recent_turns)
        recalled_context = _recalled_contexts(recalled_turns, memories)
        recent_messages, recalled_context = _apply_budgets(
            recent_messages,
            recalled_context,
            recalled_turns_limit=self.recalled_turns,
            memories_limit=self.recalled_memories,
        )
        return AgentContextEnvelope(
            conversation_id=request.conversation_id,
            request_id=request.request_id,
            turn_id=turn_id,
            current_message=request.message,
            rag_mode=request.rag_mode,
            attachment_ids=request.attachment_ids,
            workspace=workspace,
            recent_messages=recent_messages,
            recalled_context=recalled_context,
            prepared_at=datetime.now(UTC),
        )

    def _recall_query(self, message: str, workspace: ConversationWorkspace) -> str:
        parts = [message.strip()]
        if workspace.active_goal is not None:
            parts.append(workspace.active_goal.objective)
        if workspace.task_plan is not None:
            pending = [
                task
                for task in workspace.task_plan.tasks
                if task.status not in {"completed", "cancelled", "skipped", "failed"}
            ]
            for task in pending[:4]:
                parts.append(task.title)
                parts.append(task.objective)
        parts.extend(workspace.unresolved_questions)
        return " ".join(part for part in parts if part)

    async def _recall_memories(self, query: str) -> tuple[dict[str, object], ...]:
        if self.memory_provider is None:
            return ()
        try:
            memories = await self.memory_provider.search(query, limit=self.recalled_memories)
        except Exception:  # noqa: BLE001 - memory is recall-only; any failure degrades to empty
            return ()
        if not isinstance(memories, tuple):
            return ()
        return memories


def _tokens(value: str) -> set[str]:
    lowered = value.casefold()
    tokens = {match.group(0) for match in _ASCII_WORD.finditer(lowered)}
    for run in _CHINESE_RUN.finditer(lowered):
        text = run.group(0)
        tokens.add(text)
        tokens.update(text[index : index + 2] for index in range(len(text) - 1))
    return tokens


def _score(query: str, text: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    text_tokens = _tokens(text)
    return len(query_tokens & text_tokens) / len(query_tokens)


def _rank_history(
    query: str, turns: Sequence[ConversationTurn]
) -> list[tuple[ConversationTurn, float]]:
    ranked: list[tuple[ConversationTurn, float]] = []
    for turn in turns:
        text = turn.standalone_question or turn.user_question
        ranked.append((turn, _score(query, text)))
    ranked.sort(key=lambda pair: (-pair[1], -pair[0].sequence, pair[0].turn_id))
    return ranked


def _context_messages(turns: Sequence[ConversationTurn]) -> tuple[ContextMessage, ...]:
    messages: list[ContextMessage] = []
    for turn in turns:
        messages.append(
            ContextMessage(
                turn_id=turn.turn_id,
                sequence=turn.sequence * 2 - 1,
                role="user",
                content=turn.user_question,
            )
        )
        if turn.assistant_summary:
            messages.append(
                ContextMessage(
                    turn_id=turn.turn_id,
                    sequence=turn.sequence * 2,
                    role="assistant",
                    content=turn.assistant_summary,
                )
            )
    return tuple(messages)


def _recalled_contexts(
    recalled_turns: Sequence[tuple[ConversationTurn, float]],
    memories: Sequence[dict[str, object]],
) -> tuple[RecalledContext, ...]:
    items: list[RecalledContext] = []
    for turn, score in recalled_turns:
        content = turn.standalone_question or turn.user_question
        items.append(
            RecalledContext(
                source_id=turn.turn_id,
                kind="conversation_turn",
                content=content,
                relevance=round(min(max(score, 0.0), 1.0), 4),
                trust="non_evidence",
            )
        )
    for memory in memories:
        memory_id = memory.get("memory_id")
        content = memory.get("content")
        if not isinstance(memory_id, str) or not memory_id:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        raw_relevance = memory.get("relevance", 0.5)
        relevance = float(raw_relevance) if isinstance(raw_relevance, (int, float)) else 0.5
        items.append(
            RecalledContext(
                source_id=memory_id,
                kind="long_term_memory",
                content=content,
                relevance=min(max(relevance, 0.0), 1.0),
                trust="research_context",
            )
        )
    return tuple(items)


def _apply_budgets(
    recent_messages: tuple[ContextMessage, ...],
    recalled_context: tuple[RecalledContext, ...],
    *,
    recalled_turns_limit: int,
    memories_limit: int,
) -> tuple[tuple[ContextMessage, ...], tuple[RecalledContext, ...]]:
    recalled_turns = tuple(
        item for item in recalled_context if item.kind != "long_term_memory"
    )
    memories = tuple(item for item in recalled_context if item.kind == "long_term_memory")
    trimmed_turns = _trim_recalled(recalled_turns, _RECALLED_CHARS, recalled_turns_limit)
    trimmed_memories = _trim_recalled(memories, _MEMORY_CHARS, memories_limit)
    messages = _trim_recent(recent_messages, _RECENT_MESSAGE_CHARS, _RECENT_MESSAGE_LIMIT)
    return messages, (*trimmed_turns, *trimmed_memories)


def _trim_recalled(
    items: tuple[RecalledContext, ...], char_budget: int, max_items: int
) -> tuple[RecalledContext, ...]:
    ranked = sorted(items, key=lambda item: (-item.relevance, item.source_id))
    kept: list[RecalledContext] = []
    total = 0
    for item in ranked:
        if len(kept) >= max_items:
            break
        if total + len(item.content) > char_budget:
            continue
        kept.append(item)
        total += len(item.content)
    return tuple(kept)


def _trim_recent(
    messages: tuple[ContextMessage, ...], char_budget: int, max_items: int
) -> tuple[ContextMessage, ...]:
    kept = list(messages)
    while len(kept) > max_items:
        kept = kept[1:]
    total = sum(len(item.content) for item in kept)
    while kept and total > char_budget:
        kept = kept[1:]
        total = sum(len(item.content) for item in kept)
    return tuple(kept)
