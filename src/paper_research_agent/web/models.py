"""Strict JSON contracts exposed by the owner-only Web API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from paper_research_agent.answering.models import RAGAnswer, StorageClass

RAGMode = Literal["disabled", "preferred", "required"]


class WebModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)


class LoginRequest(WebModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=512)


class QuestionRequest(WebModel):
    question: str = Field(min_length=1, max_length=10_000)
    attachment_ids: tuple[str, ...] = Field(default=(), max_length=5)
    rag_mode: RAGMode = "disabled"
    request_id: str | None = Field(default=None, min_length=1, max_length=256)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question cannot be blank")
        return normalized


class ToolApprovalRequest(WebModel):
    approved: bool


class SessionResponse(WebModel):
    authenticated: Literal[True] = True
    conversation_id: str = Field(min_length=16)
    expires_at: int = Field(gt=0)


class AnonymousSessionResponse(WebModel):
    authenticated: Literal[False] = False


class OperationResponse(WebModel):
    ok: Literal[True] = True


class AttachmentResponse(WebModel):
    attachment_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    filename: str = Field(min_length=1, max_length=180)
    content_type: str = Field(max_length=120)
    size_bytes: int = Field(gt=0, le=10 * 1024 * 1024)


class HealthResponse(WebModel):
    status: Literal["ok", "ready", "not_ready"]


class RecommendedQuestion(WebModel):
    category: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=120)
    prompt: str = Field(min_length=1, max_length=2_000)


class SafeEvidenceSource(WebModel):
    citation_id: str = Field(pattern=r"^E[1-9]\d*$")
    chunk_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    title: str = Field(min_length=1)
    official_url: str | None = None
    section_id: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    evidence_type: Literal["text", "figure_summary"]
    storage_class: StorageClass
    excerpt: str = Field(min_length=1, max_length=1_200)
    final_rank: int = Field(ge=1)


class SafeRetrievalHit(WebModel):
    chunk_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    final_rank: int = Field(ge=1)
    evidence_type: Literal["text", "figure_summary"]
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    route_ranks: dict[str, int]


class SafeRetrievalTrace(WebModel):
    original_question: str = Field(min_length=1)
    resolved_question: str = Field(min_length=1)
    standalone_question: str | None = None
    chinese_query: str | None = None
    english_query: str | None = None
    rewrite_status: str = Field(min_length=1)
    degraded: bool
    degraded_reason: str | None = None
    index_id: str = Field(min_length=1)
    audit_persisted: bool
    conversation_memory_hit_count: int = Field(default=0, ge=0)
    selected_history_turn_ids: tuple[str, ...] = ()
    selected_history_questions: tuple[str, ...] = ()
    selected_history_relevances: tuple[float, ...] = ()
    inherited_across_route: bool = False
    rewrite_confidence: float = Field(default=1, ge=0, le=1)
    needs_clarification: bool = False
    recent_context_turn_count: int = Field(default=0, ge=0)
    recalled_candidate_count: int = Field(default=0, ge=0)
    interpretation_source: str = "deterministic"
    hits: tuple[SafeRetrievalHit, ...]


class SafeContextTrace(WebModel):
    estimated_tokens: int = Field(ge=0)
    token_budget: int = Field(gt=0)
    output_reserve_tokens: int = Field(ge=0)
    included_memory_turn_count: int = Field(ge=0)
    omitted_memory_turn_count: int = Field(ge=0)
    included_evidence_count: int = Field(ge=0)
    omitted_evidence_count: int = Field(ge=0)
    evidence_insufficient: bool


class SafeGenerationTrace(WebModel):
    requested_model: str = Field(min_length=1)
    actual_model: str | None = None
    prompt_version: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    attempts: int = Field(ge=0)
    audit_persisted: bool


class AskResponse(WebModel):
    answer: RAGAnswer
    sources: tuple[SafeEvidenceSource, ...]
    retrieval: SafeRetrievalTrace
    context: SafeContextTrace
    generation: SafeGenerationTrace


class SafeToolObservation(WebModel):
    sequence: int = Field(ge=1)
    tool_name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    purpose: str = Field(min_length=1, max_length=500)
    status: Literal["ok", "not_found", "insufficient", "approval_required", "denied"]
    trust: Literal["citation_evidence", "research_context", "computed_result", "side_effect"]
    item_count: int = Field(ge=0, le=100)


class SafePendingToolApproval(WebModel):
    tool_name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    purpose: str = Field(min_length=1, max_length=500)
    arguments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expires_at_epoch: float = Field(gt=0)


class ToolResearchResponse(WebModel):
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    status: Literal["completed", "approval_required"]
    observations: tuple[SafeToolObservation, ...]
    final_summary: str | None = Field(default=None, max_length=2_000)
    termination_reason: str | None = None
    pending_approval: SafePendingToolApproval | None = None


class SafeLongTermMemory(WebModel):
    memory_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    kind: Literal["preference", "project_context", "confirmed_conclusion"]
    content: str = Field(min_length=1, max_length=3_000)
    source_chunk_ids: tuple[str, ...] = Field(max_length=20)
    version: int = Field(ge=1)
    created_at: str = Field(min_length=1, max_length=64)
    updated_at: str = Field(min_length=1, max_length=64)
    expires_at: str | None = Field(default=None, max_length=64)
    supersedes_memory_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class LongTermMemoryListResponse(WebModel):
    memories: tuple[SafeLongTermMemory, ...]
