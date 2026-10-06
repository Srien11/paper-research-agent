import unittest
from unittest.mock import AsyncMock, Mock

from paper_research_agent.agent.models import EvidenceAssessment
from paper_research_agent.agent.reasoner import LangChainEvidenceReasoner
from tests.agent.test_reasoner import _observation, _plan


class ReasonerCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_assessment_exact_hit_and_semantic_input_invalidation(self) -> None:
        model = Mock()
        structured = Mock()
        structured.ainvoke = AsyncMock(
            return_value=EvidenceAssessment(evidence_sufficient=True, status="sufficient")
        )
        model.with_structured_output.return_value = structured
        reasoner = LangChainEvidenceReasoner(model)
        plan = _plan()
        observation = _observation("evidence")
        args = {"plan": plan, "observations": (observation,), "remaining_steps": 1}
        first = await reasoner.assess("question", **args)
        second = await reasoner.assess("question", **args)
        assert first == second and first is not second
        assert structured.ainvoke.await_count == 1
        instructions = structured.ainvoke.await_args.args[0][0].content
        assert "coverage, ledger, followups and next_requirement_ids" in instructions
        assert "For a comparison plan" not in instructions
        # Actual text must participate even if an upstream caller reuses the old text hash.
        await reasoner.assess("question", **{**args, "observations": (_observation("changed"),)})
        await reasoner.assess("question", **{**args, "remaining_steps": 0})
        await reasoner.assess("other question", **args)
        changed_index = observation.model_copy(
            update={"search": observation.search.model_copy(update={"index_id": "new-index"})}
        )
        await reasoner.assess("question", **{**args, "observations": (changed_index,)})
        changed_plan = plan.model_copy(
            update={
                "steps": (plan.steps[0].model_copy(update={"objective": "different objective"}),)
            }
        )
        await reasoner.assess("question", **{**args, "plan": changed_plan})
        assert structured.ainvoke.await_count == 6
        # A second model instance must not inherit another model's assessment.
        await LangChainEvidenceReasoner(model).assess("question", **args)
        assert structured.ainvoke.await_count == 7

    async def test_repaired_and_failed_results_are_not_cached(self) -> None:
        model = Mock()
        structured = Mock()
        structured.ainvoke = AsyncMock(return_value={"invalid": True})
        model.with_structured_output.return_value = structured
        reasoner = LangChainEvidenceReasoner(model)
        for _ in range(2):
            await reasoner.assess(
                "question",
                plan=_plan(),
                observations=(_observation("evidence"),),
                remaining_steps=0,
            )
        assert structured.ainvoke.await_count == 4
        assert reasoner.assessment_cache.hits == 0

    async def test_cache_hits_revalidate_and_do_not_return_poisoned_decisions(self) -> None:
        model = Mock()
        structured = Mock()
        structured.ainvoke = AsyncMock(
            return_value=EvidenceAssessment(evidence_sufficient=True, status="sufficient")
        )
        model.with_structured_output.return_value = structured
        reasoner = LangChainEvidenceReasoner(model)
        args = {"plan": _plan(), "observations": (_observation("evidence"),), "remaining_steps": 0}
        await reasoner.assess("question", **args)
        # Corrupt an internal entry to prove a hit still runs schema validation.
        for key, (_, value) in list(reasoner.assessment_cache._entries.items()):
            reasoner.assessment_cache.put(key, value.model_copy(update={"status": "missing"}))
        result = await reasoner.assess("question", **args)
        assert result.status == "sufficient"
        assert structured.ainvoke.await_count == 2
