from __future__ import annotations

import asyncio
import hashlib
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from paper_research_agent.agent.dynamic.models import DynamicResearchResult
from paper_research_agent.agent.models import (
    CompiledEvidenceFact,
    EvidenceAssessment,
    EvidenceCompilationAttemptAudit,
    EvidenceCompilationAudit,
    EvidenceCompilationRepairAudit,
    EvidenceCompilationVisibility,
    EvidenceCoverage,
    EvidenceLedgerCell,
    EvidenceRecord,
    EvidenceRequirement,
    GetEvidenceResult,
    ResearchActionRecord,
    ResearchDimension,
    ResearchObservation,
    ResearchPlan,
    ResearchStep,
    ResearchTarget,
    SearchCorpusHit,
    SearchCorpusResult,
)
from paper_research_agent.agent.runtime import ResearchRuntimeResult
from paper_research_agent.agent.tooling.contracts import ToolExecutionResult
from paper_research_agent.answering.models import AnswerRequest, GenerationResult
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.context.models import ContextEvidence, ContextLongTermMemory
from paper_research_agent.conversation.models import (
    ConversationCandidate,
    ConversationResolution,
)
from paper_research_agent.memory.config import ShortTermMemoryConfig
from paper_research_agent.memory.models import ShortTermMemoryTurn
from paper_research_agent.retrieval.contracts import (
    BilingualRetrievalRun,
    QueryRewriteTrace,
    SearchHit,
)
from paper_research_agent.web.runtime import (
    RAGRuntime,
    RuntimeBusyError,
    RuntimeClosedError,
    RuntimeDependencies,
    SafePaperMetadata,
    _research_policy_from_environment,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _chunk() -> EvidenceChunk:
    text = "Retrieval augmented generation improves grounded factual answering."
    return EvidenceChunk(
        chunk_id="chunk-1",
        asset_id="asset-1",
        corpus_id="C001",
        section_id="results",
        element_ids=("element-1",),
        page_start=3,
        page_end=3,
        token_start=0,
        token_end=8,
        text=text,
        text_sha256=_digest(text),
        config_sha256="a" * 64,
    )


class FakeRetriever:
    def __init__(self, chunk: EvidenceChunk):
        self.chunk = chunk
        self.queries: list[str] = []
        self.closed = 0

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        privacy_ttl_days: int | None = None,
    ) -> BilingualRetrievalRun:
        del privacy_ttl_days
        self.queries.append(query)
        return BilingualRetrievalRun(
            pipeline_id="test-pipeline",
            original_query=query,
            rewrite=QueryRewriteTrace(
                status="success",
                english_query="grounded RAG factual answering",
                requested_model="qwen-test",
                actual_model="qwen-test",
                prompt_version="query-rewrite-v2",
                latency_ms=4.0,
            ),
            degraded=False,
            top_k=top_k or 10,
            hits=(
                SearchHit(
                    chunk_id=self.chunk.chunk_id,
                    corpus_id=self.chunk.corpus_id,
                    asset_id=self.chunk.asset_id,
                    section_id=self.chunk.section_id,
                    page_start=self.chunk.page_start,
                    page_end=self.chunk.page_end,
                    text_sha256=self.chunk.text_sha256,
                    ranks={"en.vector": 1},
                    scores={"en.vector": 0.9},
                    final_score=0.9,
                    final_rank=1,
                ),
            ),
            index_id="idx-test",
            config_sha256="b" * 64,
            storage_classes={"C001": "internal_research_only"},
            rights_status="loaded",
            audit_persisted=True,
        )

    async def aclose(self) -> None:
        self.closed += 1


class FakeGenerator:
    model_id = "qwen-test"
    prompt_version = "rag-answer-json-v1"

    def __init__(self, gate: asyncio.Event | None = None):
        self.calls = 0
        self.closed = 0
        self.gate = gate
        self.requests: list[AnswerRequest] = []

    async def generate(self, request: AnswerRequest) -> GenerationResult:
        self.calls += 1
        self.requests.append(request)
        if self.gate is not None:
            await self.gate.wait()
        citation_id = request.context.citations[0].citation_id
        return GenerationResult(
            content=(
                '{"status":"answered","claims":[{"text":"RAG 可以改善事实性回答。",'
                f'"citation_ids":["{citation_id}"]}}],"insufficient_reason":null}}'
            ),
            requested_model=self.model_id,
            actual_model=self.model_id,
            prompt_version=self.prompt_version,
            input_tokens=120,
            output_tokens=20,
            latency_ms=8.0,
            attempts=1,
        )

    async def aclose(self) -> None:
        self.closed += 1


class FakeComparisonGenerator(FakeGenerator):
    async def generate(self, request: AnswerRequest) -> GenerationResult:
        self.calls += 1
        self.requests.append(request)
        return GenerationResult(
            content=(
                '{"status":"answered","claims":[{"text":"organized",'
                '"citation_ids":["E1","E2"],'
                '"fact_ids":["a-method-f1","b-method-f1"]}],'
                '"insufficient_reason":null}'
            ),
            requested_model=self.model_id,
            actual_model=self.model_id,
            prompt_version=self.prompt_version,
            input_tokens=20,
            output_tokens=10,
            latency_ms=2,
            attempts=1,
        )


class FakeMemoryStore:
    def __init__(self):
        self.turns: list[ShortTermMemoryTurn] = []

    def recent(self, session_id: str, *, now=None) -> tuple[ShortTermMemoryTurn, ...]:
        del now
        return tuple(turn for turn in self.turns if turn.session_id == session_id)

    def append(self, turn: ShortTermMemoryTurn, *, now=None) -> bool:
        del now
        self.turns.append(turn)
        return True


def _research_result(chunk: EvidenceChunk) -> ResearchRuntimeResult:
    record = EvidenceRecord(
        chunk_id=chunk.chunk_id,
        corpus_id=chunk.corpus_id,
        section_id=chunk.section_id,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        text=chunk.text,
        text_sha256=chunk.text_sha256,
        storage_class="internal_research_only",
    )
    search = SearchCorpusResult(
        query="grounded RAG",
        index_id="idx-agent",
        degraded=False,
        hits=(
            SearchCorpusHit(
                chunk_id=chunk.chunk_id,
                corpus_id=chunk.corpus_id,
                section_id=chunk.section_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text_sha256=chunk.text_sha256,
                storage_class="internal_research_only",
                final_rank=1,
            ),
        ),
    )
    return ResearchRuntimeResult(
        question="RAG 如何改善事实性？",
        plan=ResearchPlan(
            steps=(
                ResearchStep(
                    step_id="methods",
                    objective="查找方法",
                    query="grounded RAG",
                    top_k=4,
                ),
            )
        ),
        observations=(
            ResearchObservation(
                step_id="methods",
                objective="查找方法",
                search=search,
                evidence=GetEvidenceResult(records=(record,)),
            ),
        ),
        assessments=(EvidenceAssessment(evidence_sufficient=True, status="sufficient"),),
        action_history=(
            ResearchActionRecord(
                sequence=1,
                action="search_corpus",
                step_id="methods",
                query="grounded RAG",
            ),
            ResearchActionRecord(
                sequence=2,
                action="get_evidence",
                step_id="methods",
                chunk_ids=(chunk.chunk_id,),
            ),
            ResearchActionRecord(
                sequence=3,
                action="assess_evidence",
                step_id="methods",
                outcome="sufficient",
            ),
            ResearchActionRecord(
                sequence=4,
                action="finish",
                outcome="evidence_sufficient",
            ),
        ),
        evidence=(
            ContextEvidence(
                chunk_id=chunk.chunk_id,
                corpus_id=chunk.corpus_id,
                asset_id=chunk.asset_id,
                section_id=chunk.section_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
                text_sha256=chunk.text_sha256,
                storage_class="internal_research_only",
                final_score=1.0,
                final_rank=1,
            ),
        ),
        step_budget=3,
        tool_call_count=2,
        tool_call_budget=6,
        replan_count=0,
        evidence_sufficient=True,
        termination_reason="evidence_sufficient",
        task_state='{"kind":"untrusted_research_task_state","step_id":"methods"}',
    )


def _comparison_research_result(
    first: EvidenceChunk,
    second: EvidenceChunk,
) -> ResearchRuntimeResult:
    targets = (
        ResearchTarget(target_id="a", label="Paper A", corpus_id=first.corpus_id),
        ResearchTarget(target_id="b", label="Paper B", corpus_id=second.corpus_id),
    )
    dimension = ResearchDimension(dimension_id="method", label="方法")
    requirements = tuple(
        EvidenceRequirement(
            requirement_id=f"{target.target_id}-method",
            target_id=target.target_id,
            dimension_id="method",
            description=f"{target.label} method",
        )
        for target in targets
    )
    steps = tuple(
        ResearchStep(
            step_id=item.requirement_id,
            objective=item.description,
            query=item.description,
            corpus_id=target.corpus_id,
            target_ids=(target.target_id,),
            dimension_ids=("method",),
        )
        for item, target in zip(requirements, targets, strict=True)
    )
    observations = []
    for step, chunk in zip(steps, (first, second), strict=True):
        record = EvidenceRecord(
            chunk_id=chunk.chunk_id,
            corpus_id=chunk.corpus_id,
            section_id=chunk.section_id,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            text=chunk.text,
            text_sha256=chunk.text_sha256,
            storage_class="internal_research_only",
        )
        observations.append(
            ResearchObservation(
                step_id=step.step_id,
                objective=step.objective,
                search=SearchCorpusResult(
                    query=step.query,
                    corpus_id=chunk.corpus_id,
                    index_id="idx-agent",
                    degraded=False,
                    hits=(
                        SearchCorpusHit(
                            chunk_id=chunk.chunk_id,
                            corpus_id=chunk.corpus_id,
                            section_id=chunk.section_id,
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                            text_sha256=chunk.text_sha256,
                            storage_class="internal_research_only",
                            final_rank=1,
                        ),
                    ),
                ),
                evidence=GetEvidenceResult(records=(record,)),
            )
        )
    coverage = tuple(
        EvidenceCoverage(
            requirement_id=requirement.requirement_id,
            covered=True,
            chunk_ids=(chunk.chunk_id,),
        )
        for requirement, chunk in zip(requirements, (first, second), strict=True)
    )
    ledger = tuple(
        EvidenceLedgerCell(
            requirement_id=requirement.requirement_id,
            status="sufficient",
            facts=(
                CompiledEvidenceFact(
                    fact_id=f"{requirement.requirement_id}-f1",
                    statement=f"{target.label} uses a verified method.",
                    chunk_ids=(chunk.chunk_id,),
                    fact_requirement_ids=(
                        requirement.fact_requirements[0].fact_requirement_id,
                    ),
                ),
            ),
        )
        for requirement, target, chunk in zip(
            requirements, targets, (first, second), strict=True
        )
    )
    assessment = EvidenceAssessment(
        evidence_sufficient=True,
        status="sufficient",
        coverage=coverage,
        ledger=ledger,
        compilation_visibility=tuple(
            EvidenceCompilationVisibility(
                requirement_id=requirement.requirement_id,
                available_chunk_ids=(chunk.chunk_id,),
                visible_chunk_ids=(chunk.chunk_id,),
            )
            for requirement, chunk in zip(
                requirements, (first, second), strict=True
            )
        ),
        compilation_audit=EvidenceCompilationAudit(
            attempts=(
                EvidenceCompilationAttemptAudit(
                    attempt=1,
                    outcome="contract_invalid",
                    failure_code="ledger_ids_mismatch",
                    raw_ledger_cell_count=1,
                    raw_fact_count=1,
                    requested_requirement_ids=("a-method", "b-method"),
                    accepted_requirement_ids=("a-method",),
                    failed_requirement_ids=("b-method",),
                ),
                EvidenceCompilationAttemptAudit(
                    attempt=2,
                    outcome="validated",
                    raw_ledger_cell_count=2,
                    raw_fact_count=2,
                    requested_requirement_ids=("b-method",),
                    accepted_requirement_ids=("b-method",),
                ),
            ),
            repair=EvidenceCompilationRepairAudit(
                applied=False,
                source_assessment_available=True,
                input_fact_count=2,
                retained_fact_count=2,
            ),
        ),
    )
    context_evidence = tuple(
        ContextEvidence(
            chunk_id=chunk.chunk_id,
            corpus_id=chunk.corpus_id,
            asset_id=chunk.asset_id,
            section_id=chunk.section_id,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            text=chunk.text,
            text_sha256=chunk.text_sha256,
            storage_class="internal_research_only",
            final_score=1 / index,
            final_rank=index,
        )
        for index, chunk in enumerate((first, second), 1)
    )
    return ResearchRuntimeResult(
        question="比较 A 和 B 的方法",
        plan=ResearchPlan(
            task_type="comparison",
            targets=targets,
            dimensions=(dimension,),
            requirements=requirements,
            steps=steps,
        ),
        observations=tuple(observations),
        assessments=(assessment,),
        action_history=(
            ResearchActionRecord(
                sequence=1,
                action="finish",
                outcome="evidence_sufficient",
            ),
        ),
        evidence=context_evidence,
        step_budget=2,
        tool_call_count=4,
        tool_call_budget=4,
        replan_count=0,
        evidence_sufficient=True,
        termination_reason="evidence_sufficient",
        task_state='{"kind":"untrusted_research_task_state"}',
    )


class FakeResearchAgent:
    def __init__(self, result: ResearchRuntimeResult):
        self.result = result
        self.calls: list[tuple[str, str]] = []
        self.clear_calls: list[str] = []
        self.closed = 0
        self.dynamic_calls: list[tuple[str, str]] = []
        self.dynamic_resumes: list[tuple[str, bool]] = []
        self.planning_requirements: list[bool] = []

    @property
    def dynamic_tools_enabled(self) -> bool:
        return True

    @property
    def extended_tools_enabled(self) -> bool:
        return True

    async def run(
        self,
        question: str,
        *,
        thread_id: str,
        planning_required: bool = False,
    ) -> ResearchRuntimeResult:
        self.calls.append((question, thread_id))
        self.planning_requirements.append(planning_required)
        return self.result.model_copy(update={"question": question})

    async def clear(self, thread_id: str) -> None:
        self.clear_calls.append(thread_id)

    async def run_dynamic_tools(
        self,
        question: str,
        *,
        thread_id: str,
    ) -> DynamicResearchResult:
        self.dynamic_calls.append((question, thread_id))
        return DynamicResearchResult(
            run_id="e" * 32,
            thread_id=thread_id,
            status="completed",
            final_summary="Done.",
            termination_reason="router_finished",
        )

    async def resume_dynamic_tools(
        self,
        *,
        thread_id: str,
        approved: bool,
    ) -> DynamicResearchResult:
        self.dynamic_resumes.append((thread_id, approved))
        return DynamicResearchResult(
            run_id="e" * 32,
            thread_id=thread_id,
            status="completed",
            final_summary="Denied." if not approved else "Approved.",
            termination_reason="approval_denied" if not approved else "router_finished",
        )

    async def list_long_term_memories(self, *, limit: int = 20) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name="manage_long_term_memory",
            items=(
                {
                    "memory_id": "a" * 32,
                    "kind": "preference",
                    "content": "Concise answers",
                    "source_chunk_ids": (),
                    "version": 1,
                    "created_at": "2026-08-01T00:00:00+00:00",
                    "updated_at": "2026-08-01T00:00:00+00:00",
                    "expires_at": None,
                    "supersedes_memory_id": None,
                },
            )[:limit],
        )

    async def aclose(self) -> None:
        self.closed += 1


def _runtime(
    *,
    gate: asyncio.Event | None = None,
    research_agent: FakeResearchAgent | None = None,
    research_agent_mode: str = "always",
) -> tuple[RAGRuntime, FakeRetriever, FakeGenerator]:
    chunk = _chunk()
    retriever = FakeRetriever(chunk)
    generator = FakeGenerator(gate)
    runtime = RAGRuntime(
        RuntimeDependencies(
            chunks=(chunk,),
            papers={
                "C001": SafePaperMetadata(
                    corpus_id="C001",
                    title="A Private Local Paper",
                    official_url="https://example.test/paper",
                    storage_class="internal_research_only",
                )
            },
            retriever=retriever,
            generator=generator,
            memory_store=FakeMemoryStore(),
            memory_config=ShortTermMemoryConfig(),
            research_agent=research_agent,
            paper_candidate_retriever=AsyncMock(),
        ),
        excerpt_chars=48,
        research_agent_mode=research_agent_mode,
    )
    return runtime, retriever, generator


class RAGRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def test_research_policy_defaults_cover_bounded_dynamic_budget(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            policy = _research_policy_from_environment()

        self.assertEqual(policy.max_steps, 24)
        self.assertEqual(policy.max_followup_steps, 4)
        self.assertEqual(policy.comparison_search_concurrency, 6)
        self.assertFalse(policy.adaptive_evidence_hydration_enabled)
        self.assertEqual(policy.evidence_per_step, 4)
        self.assertEqual(policy.first_followup_evidence_per_step, 6)
        self.assertEqual(policy.later_followup_evidence_per_step, 10)
        self.assertEqual(policy.max_tool_calls, 48)
        self.assertEqual(policy.timeout_seconds, 180)

    def test_from_environment_requires_corpus_and_forwards_optional_paths(self) -> None:
        with (
            patch.dict("os.environ", {"PRA_PROJECT_ROOT": "project-root"}, clear=True),
            self.assertRaisesRegex(RuntimeError, "PRA_CORPUS_DIR"),
        ):
            RAGRuntime.from_environment()

        sentinel, _, _ = _runtime()
        environment = {
            "PRA_PROJECT_ROOT": "project-root",
            "PRA_CORPUS_DIR": "corpus",
            "PRA_CHUNKS_PATH": "private/chunks.jsonl",
            "PRA_PAPER_CARDS_PATH": "private/paper-cards.jsonl",
            "PRA_RETRIEVAL_CONFIG": "private/retrieval.json",
            "PRA_BILINGUAL_CONFIG": "private/bilingual.json",
            "PRA_ANSWER_CONFIG": "private/answer.json",
            "PRA_MEMORY_CONFIG": "private/memory.json",
            "PRA_ANSWER_AUDIT_PATH": "private/answer-audit.sqlite3",
            "PRA_SECTIONS_PATH": "private/sections.jsonl",
            "PRA_ELEMENTS_PATH": "private/elements.jsonl",
            "PRA_LOCAL_RETRIEVAL_WORKERS": "3",
        }
        with (
            patch.dict("os.environ", environment, clear=True),
            patch.object(RAGRuntime, "load", return_value=sentinel) as load,
        ):
            self.assertIs(RAGRuntime.from_environment(), sentinel)
        kwargs = load.call_args.kwargs
        self.assertEqual(kwargs["project_root"], Path("project-root"))
        self.assertEqual(kwargs["corpus_dir"], Path("project-root/corpus"))
        self.assertEqual(kwargs["chunks_path"], Path("private/chunks.jsonl"))
        self.assertEqual(kwargs["paper_cards_path"], Path("private/paper-cards.jsonl"))
        self.assertEqual(kwargs["answer_audit_path"], Path("private/answer-audit.sqlite3"))
        self.assertEqual(kwargs["sections_path"], Path("private/sections.jsonl"))
        self.assertEqual(kwargs["elements_path"], Path("private/elements.jsonl"))
        self.assertEqual(kwargs["local_retrieval_workers"], 3)

        with (
            patch.dict(
                "os.environ",
                {"PRA_PROJECT_ROOT": "project-root", "PRA_CORPUS_DIR": "corpus"},
                clear=True,
            ),
            patch.object(RAGRuntime, "load", return_value=sentinel) as default_load,
        ):
            self.assertIs(RAGRuntime.from_environment(), sentinel)
        self.assertEqual(default_load.call_args.kwargs["local_retrieval_workers"], 6)

    def test_agent_environment_flag_is_explicit_and_fail_closed(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(RAGRuntime.research_agent_enabled_from_environment())
        with patch.dict("os.environ", {"PRA_RESEARCH_AGENT_ENABLED": "true"}, clear=True):
            self.assertTrue(RAGRuntime.research_agent_enabled_from_environment())
        with (
            patch.dict(
                "os.environ",
                {"PRA_RESEARCH_AGENT_ENABLED": "sometimes"},
                clear=True,
            ),
            self.assertRaisesRegex(ValueError, "PRA_RESEARCH_AGENT_ENABLED"),
        ):
            RAGRuntime.research_agent_enabled_from_environment()

    def test_agent_mode_defaults_to_auto_and_rejects_unknown_values(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(RAGRuntime.research_agent_mode_from_environment(), "auto")
        with patch.dict("os.environ", {"PRA_RESEARCH_AGENT_MODE": "always"}, clear=True):
            self.assertEqual(RAGRuntime.research_agent_mode_from_environment(), "always")
        with (
            patch.dict("os.environ", {"PRA_RESEARCH_AGENT_MODE": "sometimes"}, clear=True),
            self.assertRaisesRegex(ValueError, "PRA_RESEARCH_AGENT_MODE"),
        ):
            RAGRuntime.research_agent_mode_from_environment()

    async def test_from_environment_with_agent_maps_policy_and_checkpoint(self) -> None:
        runtime, _, _ = _runtime()
        answer_config = Mock(model="qwen-planner-2026-01-01")
        environment = {
            "PRA_PROJECT_ROOT": "project-root",
            "PRA_RESEARCH_AGENT_CHECKPOINT_PATH": "private/agent.sqlite3",
            "PRA_COMPARISON_SEARCH_CONCURRENCY": "3",
            "PRA_RESEARCH_AGENT_EVIDENCE_PER_STEP": "3",
            "PRA_RESEARCH_AGENT_TIMEOUT_SECONDS": "45",
        }
        with (
            patch.dict("os.environ", environment, clear=True),
            patch.object(RAGRuntime, "from_environment", return_value=runtime),
            patch(
                "paper_research_agent.web.runtime.load_answering_config",
                return_value=answer_config,
            ),
            patch.object(runtime, "enable_research_agent", new=AsyncMock()) as enable,
        ):
            result = await RAGRuntime.from_environment_with_agent()

        self.assertIs(result, runtime)
        kwargs = enable.await_args.kwargs
        self.assertEqual(kwargs["model_id"], "qwen-planner-2026-01-01")
        self.assertEqual(
            kwargs["checkpoint_path"],
            Path("project-root/private/agent.sqlite3"),
        )
        self.assertEqual(kwargs["policy"].max_steps, 24)
        self.assertEqual(kwargs["policy"].comparison_search_concurrency, 3)
        self.assertEqual(kwargs["policy"].evidence_per_step, 3)
        self.assertEqual(kwargs["policy"].first_followup_evidence_per_step, 3)
        self.assertEqual(kwargs["policy"].later_followup_evidence_per_step, 3)
        self.assertEqual(kwargs["policy"].max_tool_calls, 48)
        self.assertEqual(kwargs["policy"].timeout_seconds, 45)
        self.assertEqual(kwargs["mode"], "auto")

    def test_adaptive_evidence_cutoffs_are_explicit_and_legacy_mix_fails(self) -> None:
        environment = {
            "PRA_RESEARCH_AGENT_ADAPTIVE_EVIDENCE_HYDRATION_ENABLED": "true",
            "PRA_RESEARCH_AGENT_INITIAL_EVIDENCE_PER_STEP": "5",
            "PRA_RESEARCH_AGENT_FIRST_FOLLOWUP_EVIDENCE_PER_STEP": "7",
            "PRA_RESEARCH_AGENT_LATER_FOLLOWUP_EVIDENCE_PER_STEP": "11",
        }
        with patch.dict("os.environ", environment, clear=True):
            policy = _research_policy_from_environment()

        self.assertEqual(policy.initial_evidence_per_step, 5)
        self.assertTrue(policy.adaptive_evidence_hydration_enabled)
        self.assertEqual(policy.first_followup_evidence_per_step, 7)
        self.assertEqual(policy.later_followup_evidence_per_step, 11)

        environment["PRA_RESEARCH_AGENT_EVIDENCE_PER_STEP"] = "5"
        with (
            patch.dict("os.environ", environment, clear=True),
            self.assertRaisesRegex(ValueError, "cannot be combined"),
        ):
            _research_policy_from_environment()

    async def test_enable_agent_forwards_paper_candidate_retriever(self) -> None:
        runtime, _, _ = _runtime()
        attached = FakeResearchAgent(_chunk())
        with patch(
            "paper_research_agent.agent.factory.create_research_agent_runtime",
            new=AsyncMock(return_value=attached),
        ) as create:
            await runtime.enable_research_agent(
                model_id="qwen-test",
                checkpoint_path=Path("agent.sqlite3"),
                policy={"max_steps": 2, "max_tool_calls": 4},
            )

        self.assertIs(
            create.await_args.kwargs["paper_candidate_retriever"],
            runtime._paper_candidate_retriever,
        )

    async def test_reuses_dependencies_and_returns_only_safe_trace(self) -> None:
        runtime, retriever, generator = _runtime()

        first = await runtime.ask("RAG 如何改善事实性？", session_id="a" * 32)
        second = await runtime.ask("它的依据是什么？", session_id="a" * 32)

        self.assertEqual(generator.calls, 2)
        self.assertEqual(len(retriever.queries), 2)
        self.assertIn("上一轮研究问题", retriever.queries[1])
        self.assertEqual(first.retrieval.english_query, "grounded RAG factual answering")
        self.assertEqual(first.retrieval.index_id, "idx-test")
        self.assertTrue(first.retrieval.audit_persisted)
        self.assertEqual(second.context.included_memory_turn_count, 1)
        self.assertEqual(first.context.included_evidence_count, 1)
        self.assertGreater(first.context.estimated_tokens, 0)
        self.assertEqual(first.sources[0].title, "A Private Local Paper")
        self.assertEqual(first.sources[0].storage_class, "internal_research_only")
        self.assertLessEqual(len(first.sources[0].excerpt), 49)

        rendered = first.model_dump_json()
        self.assertNotIn("local_pdf_path", rendered)
        self.assertNotIn("image_path", rendered)
        self.assertNotIn("scores", rendered)

    async def test_selected_long_term_memory_reaches_prompt_but_not_citations(self) -> None:
        runtime, _, generator = _runtime()
        memory = ContextLongTermMemory(
            memory_id="f" * 32,
            kind="project_context",
            content="项目正在比较本地 RAG 的证据边界",
            relevance=0.9,
        )

        result = await runtime.ask(
            "RAG 如何改善事实性？",
            session_id="f" * 32,
            long_term_memory=(memory,),
        )

        prompt = "\n".join(
            message.content for message in generator.requests[0].context.messages
        )
        self.assertIn("UNTRUSTED LONG-TERM MEMORY", prompt)
        self.assertIn(memory.content, prompt)
        self.assertEqual(result.context.included_long_term_memory_count, 1)
        self.assertEqual([item.chunk_id for item in result.answer.citations], ["chunk-1"])
        self.assertNotIn(memory.memory_id, {item.chunk_id for item in result.answer.citations})

    async def test_uses_agent_evidence_with_existing_answer_validation(self) -> None:
        chunk = _chunk()
        agent = FakeResearchAgent(_research_result(chunk))
        runtime, retriever, generator = _runtime(research_agent=agent)

        result = await runtime.ask("RAG 如何改善事实性？", session_id="d" * 32)

        self.assertEqual(retriever.queries, [])
        self.assertEqual(agent.calls, [("RAG 如何改善事实性？", "d" * 32)])
        self.assertEqual(result.retrieval.rewrite_status, "agent")
        self.assertEqual(result.retrieval.index_id, "idx-agent")
        self.assertEqual(result.retrieval.hits[0].route_ranks, {"agent": 1})
        self.assertEqual(result.answer.citations[0].chunk_id, "chunk-1")
        prompt = "\n".join(message.content for message in generator.requests[0].context.messages)
        self.assertIn("untrusted_research_task_state", prompt)

        await runtime.clear_conversation("d" * 32)
        self.assertEqual(agent.clear_calls, ["d" * 32])
        await runtime.aclose()
        self.assertEqual(agent.closed, 1)

    async def test_comparison_uses_compiled_ledger_without_raw_evidence_text(self) -> None:
        first = _chunk()
        second_text = "SECOND_RAW_EVIDENCE_BODY"
        second = first.model_copy(
            update={
                "chunk_id": "chunk-2",
                "asset_id": "asset-2",
                "corpus_id": "T001",
                "text": second_text,
                "text_sha256": _digest(second_text),
            }
        )
        agent = FakeResearchAgent(_comparison_research_result(first, second))
        generator = FakeComparisonGenerator()
        runtime = RAGRuntime(
            RuntimeDependencies(
                chunks=(first, second),
                papers={
                    "C001": SafePaperMetadata(
                        corpus_id="C001",
                        title="Paper A",
                        official_url="https://example.test/a",
                        storage_class="internal_research_only",
                    ),
                    "T001": SafePaperMetadata(
                        corpus_id="T001",
                        title="Paper B",
                        official_url="https://example.test/b",
                        storage_class="internal_research_only",
                    ),
                },
                retriever=FakeRetriever(first),
                generator=generator,
                memory_store=FakeMemoryStore(),
                memory_config=ShortTermMemoryConfig(),
                research_agent=agent,
                paper_candidate_retriever=AsyncMock(),
            ),
            research_agent_mode="always",
        )

        result = await runtime.ask("比较 A 和 B 的方法", session_id="ledger-test")

        prompt = "\n".join(
            message.content for request in generator.requests for message in request.context.messages
        )
        self.assertNotIn(first.text, prompt)
        self.assertNotIn(second.text, prompt)
        self.assertIn("Paper A uses a verified method", result.answer.answer_markdown)
        self.assertEqual(
            result.comparison.expressed_fact_ids,
            ("a-method-f1", "b-method-f1"),
        )
        self.assertEqual(result.comparison.fact_requirement_count, 2)
        self.assertEqual(
            result.comparison.satisfied_fact_requirement_ids,
            ("a-method-primary", "b-method-primary"),
        )
        self.assertEqual(result.comparison.visible_compiler_chunk_count, 2)
        self.assertEqual(
            result.comparison.compilation_attempt_outcomes,
            ("contract_invalid", "validated"),
        )
        self.assertEqual(
            result.comparison.compilation_failure_codes,
            ("ledger_ids_mismatch",),
        )
        self.assertEqual(result.comparison.compilation_retained_fact_count, 2)
        self.assertEqual(
            result.comparison.compilation_accepted_requirement_ids,
            ("a-method", "b-method"),
        )
        self.assertFalse(result.comparison.compilation_failed_requirement_ids)
        self.assertEqual(result.comparison.compilation_failed_unit_count, 0)
        self.assertEqual(len(result.sources), 2)

    async def test_zero_fact_compiler_failure_is_not_reported_as_insufficient(self) -> None:
        first = _chunk()
        second_text = "SECOND_RAW_EVIDENCE_BODY"
        second = first.model_copy(
            update={
                "chunk_id": "chunk-2",
                "asset_id": "asset-2",
                "corpus_id": "T001",
                "text": second_text,
                "text_sha256": _digest(second_text),
            }
        )
        research = _comparison_research_result(first, second)
        requirements = research.plan.requirements
        failed_ids = tuple(item.requirement_id for item in requirements)
        failed_assessment = EvidenceAssessment(
            evidence_sufficient=False,
            status="compiler_failed",
            coverage=tuple(
                EvidenceCoverage(requirement_id=item.requirement_id, covered=False)
                for item in requirements
            ),
            ledger=tuple(
                EvidenceLedgerCell(
                    requirement_id=item.requirement_id,
                    status="missing",
                    missing_fact_requirement_ids=tuple(
                        fact.fact_requirement_id for fact in item.fact_requirements
                    ),
                )
                for item in requirements
            ),
            compilation_audit=EvidenceCompilationAudit(
                attempts=(
                    EvidenceCompilationAttemptAudit(
                        attempt=1,
                        outcome="schema_invalid",
                        failure_code="compilation_units_invalid",
                        requested_requirement_ids=failed_ids,
                        failed_requirement_ids=failed_ids,
                    ),
                    EvidenceCompilationAttemptAudit(
                        attempt=2,
                        outcome="schema_invalid",
                        failure_code="compilation_units_invalid",
                        requested_requirement_ids=failed_ids,
                        failed_requirement_ids=failed_ids,
                    ),
                ),
                repair=EvidenceCompilationRepairAudit(
                    applied=False,
                    source_assessment_available=True,
                    missing_ledger_cell_count=2,
                ),
            ),
        )
        research = research.model_copy(
            update={
                "assessments": (failed_assessment,),
                "action_history": (
                    ResearchActionRecord(
                        sequence=1,
                        action="finish",
                        outcome="compiler_failed",
                    ),
                ),
                "evidence_sufficient": False,
                "termination_reason": "compiler_failed",
            }
        )
        agent = FakeResearchAgent(research)
        generator = FakeComparisonGenerator()
        runtime = RAGRuntime(
            RuntimeDependencies(
                chunks=(first, second),
                papers={
                    "C001": SafePaperMetadata(
                        corpus_id="C001",
                        title="Paper A",
                        official_url="https://example.test/a",
                        storage_class="internal_research_only",
                    ),
                    "T001": SafePaperMetadata(
                        corpus_id="T001",
                        title="Paper B",
                        official_url="https://example.test/b",
                        storage_class="internal_research_only",
                    ),
                },
                retriever=FakeRetriever(first),
                generator=generator,
                memory_store=FakeMemoryStore(),
                memory_config=ShortTermMemoryConfig(),
                research_agent=agent,
                paper_candidate_retriever=AsyncMock(),
            ),
            research_agent_mode="always",
        )

        result = await runtime.ask("比较 A 和 B 的方法", session_id="compiler-failed")

        self.assertEqual(result.answer.status, "compiler_failed")
        self.assertIn("不表示论文证据不足", result.answer.answer_markdown)
        self.assertEqual(generator.calls, 0)
        self.assertEqual(result.comparison.compilation_failed_requirement_ids, failed_ids)
        self.assertEqual(result.comparison.compilation_failed_unit_count, 2)
        self.assertFalse(result.sources)

    async def test_shared_conversation_context_drives_retrieval_and_answer_prompt(self) -> None:
        runtime, retriever, generator = _runtime()
        turn_id = "a" * 32
        candidate = ConversationCandidate(
            turn_id=turn_id,
            sequence=1,
            user_question="大模型测评",
            standalone_question="大模型测评",
            route="normal_chat",
            assistant_summary="讨论了评测指标与基准。",
            status="completed",
            episode_id="b" * 16,
            relevance=0.96,
        )
        resolution = ConversationResolution(
            original_question="结合一下知识库",
            standalone_question="请基于本地论文知识库，继续分析大模型测评。",
            chinese_query="大模型测评 方法 指标 基准 安全性 人工评审",
            candidates=(candidate,),
            selected_turn_ids=(turn_id,),
            confidence=0.96,
            inherited_across_route=True,
            episode_id="b" * 16,
        )

        result = await runtime.ask(
            "结合一下知识库",
            session_id="conversation-a",
            conversation_context=resolution,
        )

        self.assertEqual(retriever.queries, [resolution.standalone_question])
        prompt = "\n".join(message.content for message in generator.requests[0].context.messages)
        self.assertIn(resolution.standalone_question, prompt)
        self.assertIn("大模型测评", prompt)
        self.assertEqual(result.retrieval.selected_history_turn_ids, (turn_id,))
        self.assertEqual(result.retrieval.selected_history_relevances, (0.96,))
        self.assertTrue(result.retrieval.inherited_across_route)
        await runtime.aclose()

    async def test_auto_mode_uses_agent_only_for_scholarly_comparison(self) -> None:
        chunk = _chunk()
        agent = FakeResearchAgent(_research_result(chunk))
        runtime, retriever, _ = _runtime(
            research_agent=agent,
            research_agent_mode="auto",
        )

        direct = await runtime.ask("RAG 如何改善事实性？", session_id="1" * 32)
        comparison = await runtime.ask(
            "比较这两篇论文的方法和实验指标",
            session_id="2" * 32,
        )

        self.assertEqual(len(retriever.queries), 1)
        self.assertEqual(agent.calls, [("比较这两篇论文的方法和实验指标", "2" * 32)])
        self.assertEqual(direct.retrieval.rewrite_status, "success")
        self.assertEqual(comparison.retrieval.rewrite_status, "agent")
        self.assertEqual(agent.planning_requirements, [True])

    async def test_dynamic_tool_lane_uses_agent_checkpoint_thread(self) -> None:
        agent = FakeResearchAgent(_research_result(_chunk()))
        runtime, _, _ = _runtime(research_agent=agent)

        started = await runtime.run_tool_research(
            "  Compare the papers  ",
            session_id="f" * 32,
        )
        resumed = await runtime.resume_tool_research(
            session_id="f" * 32,
            approved=False,
        )

        self.assertEqual(started.status, "completed")
        self.assertEqual(resumed.termination_reason, "approval_denied")
        self.assertEqual(agent.dynamic_calls, [("Compare the papers", "f" * 32)])
        self.assertEqual(agent.dynamic_resumes, [("f" * 32, False)])
        memories = await runtime.list_long_term_memories()
        self.assertEqual(memories.items[0]["content"], "Concise answers")
        await runtime.aclose()

    async def test_rejects_concurrent_question_as_busy(self) -> None:
        gate = asyncio.Event()
        runtime, _, _ = _runtime(gate=gate)
        first = asyncio.create_task(runtime.ask("第一个问题", session_id="b" * 32))
        while not runtime.is_busy:
            await asyncio.sleep(0)

        with self.assertRaises(RuntimeBusyError):
            await runtime.ask("第二个问题", session_id="b" * 32)

        gate.set()
        await first
        self.assertFalse(runtime.is_busy)

    async def test_close_is_idempotent_and_prevents_new_work(self) -> None:
        runtime, retriever, generator = _runtime()

        await runtime.aclose()
        await runtime.aclose()

        self.assertEqual(retriever.closed, 1)
        self.assertEqual(generator.closed, 1)
        self.assertFalse(runtime.is_ready)
        with self.assertRaises(RuntimeClosedError):
            await runtime.ask("问题", session_id="c" * 32)


if __name__ == "__main__":
    unittest.main()
