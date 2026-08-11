"""Strict contracts for the shared conversation ledger."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ConversationStatus = Literal[
    "pending",
    "completed",
    "insufficient_evidence",
    "clarification_required",
    "failed",
    "cancelled",
]


class FrozenConversationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConversationTurn(FrozenConversationModel):
    turn_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    conversation_id: str = Field(min_length=1, max_length=256)
    sequence: int = Field(ge=1)
    user_question: str = Field(min_length=1, max_length=10_000)
    standalone_question: str | None = Field(default=None, max_length=2_000)
    route: str | None = Field(default=None, max_length=64)
    status: ConversationStatus
    assistant_summary: str | None = Field(default=None, max_length=3_000)
    source_ids: tuple[str, ...] = Field(default=(), max_length=100)
    episode_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    selected_history_turn_ids: tuple[str, ...] = Field(default=(), max_length=10)
    selected_history_relevances: tuple[float, ...] = Field(default=(), max_length=10)
    rewrite_confidence: float | None = Field(default=None, ge=0, le=1)
    created_at: datetime
    completed_at: datetime | None = None

    @field_validator("user_question", "standalone_question", "assistant_summary")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("conversation text cannot be blank")
        return normalized

    @field_validator("selected_history_relevances")
    @classmethod
    def validate_relevances(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("history relevance must be between 0 and 1")
        return values


class PersistedConversationMessage(FrozenConversationModel):
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=20_000)
    status: str = Field(min_length=1, max_length=64)
    created_at: datetime


class PersistedConversation(FrozenConversationModel):
    conversation_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=200)
    created_at: datetime
    updated_at: datetime
    messages: tuple[PersistedConversationMessage, ...] = Field(
        default=(), max_length=1_000
    )

class ConversationEpisode(FrozenConversationModel):
    conversation_id: str = Field(min_length=1, max_length=256)
    episode_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    summary: str = Field(min_length=1, max_length=2_000)
    last_sequence: int = Field(ge=1)
    updated_at: datetime


class ConversationCandidate(FrozenConversationModel):
    turn_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    sequence: int = Field(ge=1)
    user_question: str = Field(min_length=1, max_length=10_000)
    standalone_question: str = Field(min_length=1, max_length=2_000)
    route: str | None = Field(default=None, max_length=64)
    assistant_summary: str | None = Field(default=None, max_length=3_000)
    status: ConversationStatus
    episode_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    relevance: float = Field(ge=0, le=1)


class ConversationResolution(FrozenConversationModel):
    original_question: str = Field(min_length=1, max_length=10_000)
    standalone_question: str = Field(min_length=1, max_length=2_000)
    chinese_query: str = Field(min_length=1, max_length=2_000)
    candidates: tuple[ConversationCandidate, ...] = Field(default=(), max_length=10)
    selected_turn_ids: tuple[str, ...] = Field(default=(), max_length=10)
    confidence: float = Field(ge=0, le=1)
    needs_clarification: bool = False
    clarification_question: str | None = Field(default=None, max_length=500)
    inherited_across_route: bool = False
    episode_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    recent_context_turn_count: int = Field(default=0, ge=0, le=12)
    recalled_candidate_count: int = Field(default=0, ge=0, le=10)
    interpretation_source: Literal["model", "fallback", "deterministic"] = "deterministic"

    @property
    def selected_candidates(self) -> tuple[ConversationCandidate, ...]:
        selected = set(self.selected_turn_ids)
        return tuple(item for item in self.candidates if item.turn_id in selected)


class ConversationContextSnapshot(FrozenConversationModel):
    original_question: str = Field(min_length=1, max_length=10_000)
    recent_turns: tuple[ConversationCandidate, ...] = Field(default=(), max_length=12)
    recalled_turns: tuple[ConversationCandidate, ...] = Field(default=(), max_length=10)
    episodes: tuple[ConversationEpisode, ...] = Field(default=(), max_length=100)
    prepared_at: datetime

    @property
    def candidates(self) -> tuple[ConversationCandidate, ...]:
        values: list[ConversationCandidate] = []
        seen: set[str] = set()
        for candidate in (*self.recent_turns, *self.recalled_turns):
            if candidate.turn_id in seen:
                continue
            seen.add(candidate.turn_id)
            values.append(candidate)
        return tuple(values)


class TurnInterpretation(FrozenConversationModel):
    depends_on_history: bool
    selected_history_turn_ids: tuple[str, ...] = Field(default=(), max_length=10)
    standalone_question: str = Field(min_length=1, max_length=2_000)
    chinese_query: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(ge=0, le=1)
    needs_clarification: bool = False
    clarification_question: str | None = Field(default=None, max_length=500)
    route: Literal[
        "normal_chat",
        "local_rag",
        "web_research",
        "attachment_qa",
        "file_edit",
    ]
    use_local_papers: bool = False
    use_web_research: bool = False
    use_dynamic_tools: bool = False
    use_attachments: bool = False
    research_mode: Literal["single", "planned"] = "single"
    reason: str = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def validate_consistency(self) -> TurnInterpretation:
        if (
            self.depends_on_history
            and not self.selected_history_turn_ids
            and not self.needs_clarification
        ):
            raise ValueError("history-dependent interpretation must select a turn")
        if not self.depends_on_history and self.selected_history_turn_ids:
            raise ValueError("history-independent interpretation cannot select history")
        if self.needs_clarification and not self.clarification_question:
            raise ValueError("clarification question is required")
        if not self.needs_clarification and self.clarification_question is not None:
            raise ValueError("clarification question must be null when clarification is not needed")
        return self
