from __future__ import annotations

import unittest

from pydantic import ValidationError

from paper_research_agent.agent.models import (
    EvidenceAssessment,
    EvidenceRecord,
    GetEvidenceInput,
    GetEvidenceResult,
    ResearchActionRecord,
    SearchCorpusHit,
    SearchCorpusInput,
    SearchCorpusResult,
)


def _hit(*, rank: int = 1) -> SearchCorpusHit:
    return SearchCorpusHit(
        chunk_id=f"chunk-{rank}",
        corpus_id="C001",
        section_id="results",
        page_start=2,
        page_end=3,
        text_sha256=f"{rank}" * 64,
        evidence_type="text",
        storage_class="internal_research_only",
        final_rank=rank,
    )


class ResearchToolModelTests(unittest.TestCase):
    def test_search_input_normalizes_query_and_bounds_top_k(self) -> None:
        request = SearchCorpusInput(query="  grounded RAG  ", top_k=3)

        self.assertEqual(request.query, "grounded RAG")
        self.assertEqual(request.top_k, 3)
        with self.assertRaises(ValidationError):
            SearchCorpusInput(query=" ")
        with self.assertRaises(ValidationError):
            SearchCorpusInput(query="q", top_k=21)
        with self.assertRaises(ValidationError):
            SearchCorpusInput(query="q", unexpected=True)

    def test_search_result_requires_contiguous_ranks_and_degradation_reason(self) -> None:
        result = SearchCorpusResult(
            query="grounded RAG",
            index_id="idx-test",
            degraded=False,
            hits=(_hit(),),
        )

        self.assertEqual(result.hits[0].storage_class, "internal_research_only")
        with self.assertRaises(ValidationError):
            SearchCorpusResult(
                query="grounded RAG",
                index_id="idx-test",
                degraded=False,
                hits=(_hit(rank=2),),
            )
        with self.assertRaises(ValidationError):
            SearchCorpusResult(
                query="grounded RAG",
                index_id="idx-test",
                degraded=True,
                hits=(),
            )

    def test_get_evidence_rejects_duplicate_ids_and_result_overlap(self) -> None:
        with self.assertRaises(ValidationError):
            GetEvidenceInput(chunk_ids=("chunk-1", "chunk-1"))

        record = EvidenceRecord(
            chunk_id="chunk-1",
            corpus_id="C001",
            section_id=None,
            page_start=1,
            page_end=1,
            text="Evidence text.",
            text_sha256="a" * 64,
            evidence_type="text",
            storage_class="redistributable",
        )
        result = GetEvidenceResult(records=(record,), missing_chunk_ids=("missing",))
        self.assertEqual(result.records[0].text, "Evidence text.")
        with self.assertRaises(ValidationError):
            GetEvidenceResult(records=(record,), missing_chunk_ids=("chunk-1",))

    def test_evidence_assessment_requires_consistent_next_search(self) -> None:
        sufficient = EvidenceAssessment(
            evidence_sufficient=True,
            status="sufficient",
        )

        self.assertIsNone(sufficient.next_query)
        with self.assertRaises(ValidationError):
            EvidenceAssessment(
                evidence_sufficient=True,
                status="sufficient",
                next_query="another query",
                next_objective="find more evidence",
            )
        with self.assertRaises(ValidationError):
            EvidenceAssessment(
                evidence_sufficient=False,
                status="missing_coverage",
                next_query="missing comparison",
            )

        retry = EvidenceAssessment(
            evidence_sufficient=False,
            status="missing_coverage",
            next_query="  missing comparison  ",
            next_objective="  cover the missing dimension  ",
        )
        self.assertEqual(retry.next_query, "missing comparison")

    def test_action_record_enforces_fields_for_each_action(self) -> None:
        search = ResearchActionRecord(
            sequence=1,
            action="search_corpus",
            step_id="methods",
            query="RAG evaluation",
        )
        evidence = ResearchActionRecord(
            sequence=2,
            action="get_evidence",
            step_id="methods",
            chunk_ids=("chunk-1",),
        )
        assessment = ResearchActionRecord(
            sequence=3,
            action="assess_evidence",
            step_id="methods",
            outcome="missing_coverage",
        )
        finish = ResearchActionRecord(
            sequence=4,
            action="finish",
            outcome="plan_exhausted",
        )

        self.assertEqual(
            [search.action, evidence.action, assessment.action, finish.action],
            ["search_corpus", "get_evidence", "assess_evidence", "finish"],
        )
        with self.assertRaises(ValidationError):
            ResearchActionRecord(
                sequence=1,
                action="search_corpus",
                step_id="methods",
            )
        with self.assertRaises(ValidationError):
            ResearchActionRecord(
                sequence=1,
                action="get_evidence",
                step_id="methods",
                chunk_ids=(),
            )
        with self.assertRaises(ValidationError):
            ResearchActionRecord(
                sequence=1,
                action="finish",
                outcome="sufficient",
            )


if __name__ == "__main__":
    unittest.main()
