from __future__ import annotations

import hashlib
import sys
import unittest
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.gold_generation import (
    GeneratedCandidate,
    SourceEvidence,
    build_gold_question,
)
from paper_research_agent.evaluation.gold_selection import CandidateBlueprint


def _blueprint(*, answerable: bool = True) -> CandidateBlueprint:
    return CandidateBlueprint(
        case_id="G001",
        answerable=answerable,
        task_type="definition_scope",
        language="zh",
        difficulty="medium",
        evidence_source="body",
        primary_split="core",
        target_paper_ids=("C001",),
        expected_status="answered" if answerable else "insufficient_evidence",
        unanswerable_reason=None if answerable else "corpus_absent",
        nearest_distractor_paper_ids=() if answerable else ("C001", "C002"),
    )


def _source(*, role: str = "required") -> SourceEvidence:
    quote = "The benchmark contains heterogeneous retrieval tasks."
    return SourceEvidence(
        span_id="S001",
        paper_id="C001",
        evidence_version_id="source-v1",
        page=3,
        element_id="element-1",
        raw_quote=quote,
        span_hash=hashlib.sha256(quote.encode()).hexdigest(),
        support_role=role,
        projected_chunk_ids=("chk_001",),
    )


class GoldGenerationTests(unittest.TestCase):
    def test_builds_answerable_silver_candidate_with_claim_relations(self) -> None:
        draft = GeneratedCandidate.model_validate(
            {
                "question": "该基准覆盖什么类型的检索任务？",
                "must_have_claims": [
                    {"claim_id": "M1", "text": "包含异构检索任务。", "span_ids": ["S001"]}
                ],
                "forbidden_claims": [
                    {"claim_id": "F1", "text": "只包含单一类型任务。"}
                ],
            }
        )
        question = build_gold_question(
            _blueprint(),
            [_source()],
            draft,
            corpus_version="corpus-v1",
            knowledge_cutoff=date(2026, 7, 26),
        )
        self.assertEqual(question.question_id, "GQ001")
        self.assertEqual(question.annotation_status, "silver_generated")
        self.assertEqual(question.citation_relations[0].claim_id, "M1")
        self.assertEqual(question.evidence_spans[0].raw_span_start, 0)
        self.assertEqual(question.evidence_spans[0].raw_span_end, len(_source().raw_quote))

    def test_builds_strict_unanswerable_candidate_without_supporting_claim(self) -> None:
        draft = GeneratedCandidate.model_validate(
            {
                "question": "该基准在 2027 年新增了哪些任务？",
                "must_have_claims": [],
                "forbidden_claims": [],
                "unanswerable_reason": "问题超出冻结语料的时间截止点。",
            }
        )
        question = build_gold_question(
            _blueprint(answerable=False),
            [_source(role="distractor")],
            draft,
            corpus_version="corpus-v1",
            knowledge_cutoff=date(2026, 7, 26),
        )
        self.assertFalse(question.answerable)
        self.assertFalse(question.must_have_claims)
        self.assertTrue(question.unanswerable_reason)
        self.assertTrue(all(span.support_role == "distractor" for span in question.evidence_spans))

    def test_rejects_unknown_span_in_generated_claim(self) -> None:
        draft = GeneratedCandidate.model_validate(
            {
                "question": "问题",
                "must_have_claims": [
                    {"claim_id": "M1", "text": "事实", "span_ids": ["missing"]}
                ],
                "forbidden_claims": [],
            }
        )
        with self.assertRaisesRegex(ValueError, "unknown evidence span"):
            build_gold_question(
                _blueprint(),
                [_source()],
                draft,
                corpus_version="corpus-v1",
                knowledge_cutoff=date(2026, 7, 26),
            )


if __name__ == "__main__":
    unittest.main()
