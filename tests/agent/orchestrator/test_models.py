from __future__ import annotations

import unittest
from datetime import UTC, datetime

from pydantic import ValidationError

from paper_research_agent.agent.orchestrator.artifacts import (
    AttachmentArtifact,
    ChatArtifact,
    DynamicToolArtifact,
    FileArtifact,
    LocalRAGArtifact,
    LocalRAGTrace,
)
from paper_research_agent.agent.orchestrator.models import (
    AcceptanceCriterion,
    AgentContextEnvelope,
    AgentTask,
    ChildTaskResult,
    ContextMessage,
    ConversationWorkspace,
    GoalDecision,
    GoalState,
    MainAgentResult,
    RecalledContext,
    TaskPlan,
    TaskPlanDecision,
    TurnInterpretationV2,
)
from paper_research_agent.answering.models import RAGAnswer


def _utc() -> datetime:
    return datetime(2026, 8, 7, tzinfo=UTC)


def _goal(**overrides: object) -> GoalState:
    values: dict[str, object] = {
        "goal_id": "a" * 32,
        "objective": "比较 RAG 与 GraphRAG 并给出选型建议",
        "status": "active",
        "acceptance_criteria": (),
        "constraints": ("只依据本地论文库",),
        "origin_turn_id": "b" * 32,
        "created_at": _utc(),
        "updated_at": _utc(),
    }
    values.update(overrides)
    return GoalState(**values)


def _task(**overrides: object) -> AgentTask:
    values: dict[str, object] = {
        "task_id": "collect-local-evidence",
        "goal_id": "a" * 32,
        "title": "收集本地论文证据",
        "objective": "检索本地语料中关于 RAG 评测的论文",
        "success_criteria": ("找到至少两篇论文证据",),
        "capability": "local_rag",
        "status": "pending",
        "depends_on": (),
        "attempt_count": 0,
        "result_ref": None,
        "blocked_reason": None,
    }
    values.update(overrides)
    return AgentTask(**values)


def _plan(**overrides: object) -> TaskPlan:
    values: dict[str, object] = {
        "plan_id": "c" * 32,
        "goal_id": "a" * 32,
        "revision": 1,
        "tasks": (_task(),),
        "created_at": _utc(),
        "updated_at": _utc(),
    }
    values.update(overrides)
    return TaskPlan(**values)


def _workspace(**overrides: object) -> ConversationWorkspace:
    values: dict[str, object] = {
        "schema_version": "conversation-workspace-v1",
        "conversation_id": "conversation-1",
        "version": 0,
        "summary": "",
        "active_goal": _goal(),
        "task_plan": None,
        "unresolved_questions": (),
        "stable_constraints": (),
        "updated_at": _utc(),
    }
    values.update(overrides)
    return ConversationWorkspace(**values)


class MainAgentModelTests(unittest.TestCase):
    def test_child_artifacts_are_strict_discriminated_contracts(self) -> None:
        chat = ChatArtifact(text="普通回答")
        attachment = AttachmentArtifact(text="附件摘要", source_attachment_ids=("a" * 32,))
        edited = FileArtifact(text="已完成修改", output_attachment_ids=("b" * 32,))
        dynamic = DynamicToolArtifact(
            text="已核验论文状态",
            tool_names=("check_paper_status",),
        )

        self.assertEqual(chat.kind, "chat")
        self.assertEqual(attachment.kind, "attachment_qa")
        self.assertEqual(edited.kind, "file_edit")
        self.assertEqual(dynamic.kind, "dynamic_tools")
        with self.assertRaises(ValidationError):
            ChatArtifact(text="回答", unknown="value")
        with self.assertRaises(ValidationError):
            FileArtifact(text="未产生文件", output_attachment_ids=())

    def test_local_rag_artifact_preserves_validated_answer(self) -> None:
        answer = RAGAnswer(
            status="insufficient_evidence",
            answer_markdown="当前证据不足。",
            claims=(),
            citations=(),
            requested_model="test-model",
            prompt_version="test-prompt",
            latency_ms=0,
            attempts=0,
        )
        artifact = LocalRAGArtifact(
            text=answer.answer_markdown,
            answer=answer,
            retrieval=LocalRAGTrace(
                index_id="index-v1",
                resolved_question_sha256="a" * 64,
                hit_count=0,
            ),
        )

        self.assertEqual(artifact.kind, "local_rag")
        self.assertIs(artifact.answer, answer)
        self.assertEqual(artifact.retrieval.index_id, "index-v1")

    def test_goal_state_enforces_id_pattern_and_criteria_limit(self) -> None:
        with self.assertRaises(ValidationError):
            _goal(goal_id="short")
        with self.assertRaises(ValidationError):
            _goal(origin_turn_id="not-a-turn")
        with self.assertRaises(ValidationError):
            _goal(objective="   ")
        with self.assertRaises(ValidationError):
            _goal(
                acceptance_criteria=tuple(
                    AcceptanceCriterion(criterion_id=f"criterion-{i}", description=f"标准 {i}")
                    for i in range(13)
                )
            )
        with self.assertRaises(ValidationError):
            _goal(status="finished")

    def test_acceptance_criterion_rejects_blank_text(self) -> None:
        with self.assertRaises(ValidationError):
            AcceptanceCriterion(criterion_id="criterion-1", description="   ")
        with self.assertRaises(ValidationError):
            AcceptanceCriterion(criterion_id="bad id!", description="标准")

    def test_task_enforces_capability_and_attempt_limits(self) -> None:
        with self.assertRaises(ValidationError):
            _task(capability="run_shell")
        with self.assertRaises(ValidationError):
            _task(status="awaiting")
        with self.assertRaises(ValidationError):
            _task(attempt_count=6)
        with self.assertRaises(ValidationError):
            _task(success_criteria=())

    def test_task_plan_rejects_duplicate_task_ids(self) -> None:
        with self.assertRaisesRegex(ValidationError, "unique"):
            _plan(tasks=(_task(), _task()))

    def test_task_plan_rejects_goal_id_mismatch(self) -> None:
        with self.assertRaisesRegex(ValidationError, "goal ID"):
            _plan(tasks=(_task(goal_id="d" * 32),))

    def test_task_plan_rejects_unknown_dependency(self) -> None:
        with self.assertRaisesRegex(ValidationError, "unknown task"):
            _plan(tasks=(_task(depends_on=("missing-task",)),))

    def test_task_plan_rejects_self_dependency(self) -> None:
        with self.assertRaisesRegex(ValidationError, "itself"):
            _plan(tasks=(_task(depends_on=("collect-local-evidence",)),))

    def test_task_plan_rejects_dependency_cycle(self) -> None:
        with self.assertRaisesRegex(ValidationError, "cycle"):
            _plan(
                tasks=(
                    _task(task_id="compare-evidence", depends_on=("verify-cost",)),
                    _task(task_id="verify-cost", depends_on=("compare-evidence",)),
                )
            )

    def test_task_plan_accepts_valid_dependency_chain(self) -> None:
        plan = _plan(
            tasks=(
                _task(task_id="collect-local-evidence"),
                _task(task_id="verify-cost", depends_on=("collect-local-evidence",)),
            )
        )
        self.assertEqual(len(plan.tasks), 2)
        with self.assertRaises(ValidationError):
            _plan(revision=0)

    def test_task_plan_accepts_local_and_scholarly_parallel_group(self) -> None:
        plan = _plan(
            tasks=(
                _task(
                    task_id="local-research",
                    parallel_group_id="hybrid-research",
                ),
                _task(
                    task_id="external-research",
                    capability="dynamic_tools",
                    parallel_group_id="hybrid-research",
                    allowed_tool_risks=("network_read",),
                    allowed_tool_names=(
                        "search_scholarly_sources",
                        "resolve_paper_identifier",
                        "get_citation_graph",
                        "check_paper_status",
                    ),
                ),
            )
        )

        self.assertEqual(plan.tasks[0].parallel_group_id, "hybrid-research")
        self.assertEqual(plan.tasks[1].allowed_tool_risks, ("network_read",))

    def test_task_plan_rejects_incomplete_or_wrong_parallel_groups(self) -> None:
        with self.assertRaisesRegex(ValidationError, "exactly two"):
            _plan(tasks=(_task(parallel_group_id="hybrid-research"),))
        with self.assertRaisesRegex(ValidationError, "local_rag and dynamic_tools"):
            _plan(
                tasks=(
                    _task(task_id="local-a", parallel_group_id="hybrid-research"),
                    _task(task_id="local-b", parallel_group_id="hybrid-research"),
                )
            )

    def test_task_plan_rejects_parallel_dependency_or_permission_escalation(self) -> None:
        scholarly_names = (
            "search_scholarly_sources",
            "resolve_paper_identifier",
            "get_citation_graph",
            "check_paper_status",
        )
        with self.assertRaisesRegex(ValidationError, "must not depend on each other"):
            _plan(
                tasks=(
                    _task(
                        task_id="local-research",
                        parallel_group_id="hybrid-research",
                        depends_on=("external-research",),
                    ),
                    _task(
                        task_id="external-research",
                        capability="dynamic_tools",
                        parallel_group_id="hybrid-research",
                        allowed_tool_risks=("network_read",),
                        allowed_tool_names=scholarly_names,
                    ),
                )
            )
        with self.assertRaisesRegex(ValidationError, "network_read only"):
            _plan(
                tasks=(
                    _task(task_id="local-research", parallel_group_id="hybrid-research"),
                    _task(
                        task_id="external-research",
                        capability="dynamic_tools",
                        parallel_group_id="hybrid-research",
                        allowed_tool_risks=("network_read", "write"),
                        allowed_tool_names=scholarly_names,
                    ),
                )
            )
        with self.assertRaisesRegex(ValidationError, "scholarly tool allowlist"):
            _plan(
                tasks=(
                    _task(task_id="local-research", parallel_group_id="hybrid-research"),
                    _task(
                        task_id="external-research",
                        capability="dynamic_tools",
                        parallel_group_id="hybrid-research",
                        allowed_tool_risks=("network_read",),
                        allowed_tool_names=("github__search_code",),
                    ),
                )
            )

    def test_new_task_fields_preserve_old_single_task_serialization(self) -> None:
        restored = AgentTask.model_validate(_task().model_dump(exclude={
            "parallel_group_id",
            "allowed_tool_risks",
            "allowed_tool_names",
        }))

        self.assertIsNone(restored.parallel_group_id)
        self.assertEqual(restored.allowed_tool_risks, ())
        self.assertEqual(restored.allowed_tool_names, ())

    def test_workspace_requires_schema_version_and_enforces_enum(self) -> None:
        with self.assertRaises(ValidationError):
            _workspace(schema_version="conversation-workspace-v2")

    def test_context_message_enforces_role_and_trust(self) -> None:
        message = ContextMessage(
            turn_id="turn-1",
            sequence=1,
            role="user",
            content=" 帮我比较 RAG 和 GraphRAG  ",
        )
        self.assertEqual(message.content, "帮我比较 RAG 和 GraphRAG")
        with self.assertRaises(ValidationError):
            ContextMessage(turn_id="turn-1", sequence=1, role="system", content="hi")
        with self.assertRaises(ValidationError):
            ContextMessage(turn_id="turn-1", sequence=1, role="user", content="hi", trust="evidence")
        with self.assertRaises(ValidationError):
            ContextMessage(turn_id="turn-1", sequence=1, role="user", content="")

    def test_recalled_context_validates_kind_trust_and_relevance(self) -> None:
        with self.assertRaises(ValidationError):
            RecalledContext(
                source_id="source-1",
                kind="chat_log",
                content="旧对话",
                relevance=1.0,
                trust="non_evidence",
            )
        with self.assertRaises(ValidationError):
            RecalledContext(
                source_id="source-1",
                kind="conversation_turn",
                content="旧对话",
                relevance=1.5,
                trust="non_evidence",
            )

    def test_envelope_requires_valid_rag_mode(self) -> None:
        base = {
            "conversation_id": "conversation-1",
            "request_id": "request-1",
            "turn_id": "turn-1",
            "current_message": "帮我比较 RAG 和 GraphRAG",
            "workspace": _workspace(),
            "prepared_at": _utc(),
        }
        envelope = AgentContextEnvelope(rag_mode="required", **base)
        self.assertEqual(envelope.rag_mode, "required")
        with self.assertRaises(ValidationError):
            AgentContextEnvelope(rag_mode="always", **base)

    def test_turn_interpretation_v2_validates_relation_and_clarification(self) -> None:
        interpretation = TurnInterpretationV2(
            relation="continue_goal",
            resolved_request="继续比较评测指标",
            confidence=0.9,
        )
        self.assertEqual(interpretation.relation, "continue_goal")
        with self.assertRaises(ValidationError):
            TurnInterpretationV2(
                relation="keep_going",
                resolved_request="继续",
                confidence=0.9,
            )
        with self.assertRaises(ValidationError):
            TurnInterpretationV2(
                relation="new_goal",
                resolved_request="继续",
                needs_clarification=True,
                confidence=0.4,
            )
        with self.assertRaises(ValidationError):
            TurnInterpretationV2(
                relation="new_goal",
                resolved_request="继续",
                needs_clarification=False,
                clarification_question="你想做什么？",
                confidence=0.9,
            )
        with self.assertRaises(ValidationError):
            TurnInterpretationV2(
                relation="new_goal",
                resolved_request="继续",
                confidence=1.5,
            )

    def test_goal_and_plan_decisions_validate_action_enums(self) -> None:
        decision = GoalDecision(action="keep", goal=_goal(), rationale="目标未变")
        self.assertEqual(decision.action, "keep")
        with self.assertRaises(ValidationError):
            GoalDecision(action="delete", goal=None, rationale="非法动作")
        plan_decision = TaskPlanDecision(action="keep", plan=_plan(), rationale="计划未变")
        self.assertEqual(plan_decision.action, "keep")
        with self.assertRaises(ValidationError):
            TaskPlanDecision(action="drop", plan=None, rationale="非法动作")

    def test_child_task_result_requires_approval_consistency(self) -> None:
        completed = ChildTaskResult(
            child_run_id="child-run-1",
            task_id="collect-local-evidence",
            capability="local_rag",
            status="completed",
            summary="已收集证据",
            citation_kind="local_paper",
        )
        self.assertEqual(completed.citation_kind, "local_paper")
        with self.assertRaisesRegex(ValidationError, "approval"):
            ChildTaskResult(
                child_run_id="child-run-2",
                task_id="save-report",
                capability="dynamic_tools",
                status="waiting_approval",
            )
        with self.assertRaisesRegex(ValidationError, "approval"):
            ChildTaskResult(
                child_run_id="child-run-3",
                task_id="save-report",
                capability="dynamic_tools",
                status="completed",
                pending_approval={"title": "报告"},
            )
        with self.assertRaisesRegex(ValidationError, "error code"):
            ChildTaskResult(
                child_run_id="child-run-4",
                task_id="collect-local-evidence",
                capability="local_rag",
                status="failed",
            )

        pending = ChildTaskResult(
            child_run_id="child-run-5",
            task_id="save-report",
            capability="dynamic_tools",
            status="waiting_approval",
            pending_approval={"title": "报告"},
        )
        self.assertIsNotNone(pending.pending_approval)

    def test_main_agent_result_validates_degradation_codes(self) -> None:
        result = MainAgentResult(
            run_id="run-1",
            request_id="request-1",
            conversation_id="conversation-1",
            status="completed",
            degraded=True,
            degradation_codes=("external_research_unavailable",),
        )
        self.assertTrue(result.degraded)
        with self.assertRaisesRegex(ValidationError, "unique"):
            MainAgentResult(
                run_id="run-2",
                request_id="request-2",
                conversation_id="conversation-1",
                status="completed",
                degraded=True,
                degradation_codes=(
                    "external_research_unavailable",
                    "external_research_unavailable",
                ),
            )
        with self.assertRaisesRegex(ValidationError, "exactly"):
            MainAgentResult(
                run_id="run-3",
                request_id="request-3",
                conversation_id="conversation-1",
                status="completed",
                degraded=False,
                degradation_codes=("external_research_unavailable",),
            )


if __name__ == "__main__":
    unittest.main()
