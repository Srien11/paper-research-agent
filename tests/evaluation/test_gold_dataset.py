from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.gold_dataset import (
    CitationRelation,
    EvidenceSpan,
    GoldClaim,
    GoldQuestion,
    SourceReplayError,
    load_gold_dataset,
    validate_source_replay,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _span(**updates: object) -> EvidenceSpan:
    quote = "supported evidence"
    values: dict[str, object] = {
        "span_id": "S001",
        "paper_id": "C001",
        "evidence_version_id": "source-v1",
        "page": 1,
        "element_id": "element-1",
        "raw_span_start": 7,
        "raw_span_end": 25,
        "raw_quote": quote,
        "span_hash": _sha256(quote),
        "support_role": "required",
        "projected_chunk_ids": ["chunk-1"],
    }
    values.update(updates)
    return EvidenceSpan.model_validate(values)


def _question(**updates: object) -> GoldQuestion:
    values: dict[str, object] = {
        "question_id": "GQ001",
        "split": "sealed_test",
        "language": "zh",
        "task_type": "definition_scope",
        "difficulty": "easy",
        "question": "该方法的定义是什么？",
        "answerable": True,
        "must_have_claims": [{"claim_id": "C1", "text": "一个必要事实"}],
        "forbidden_claims": [],
        "evidence_spans": [_span().model_dump(mode="json")],
        "citation_relations": [
            {"claim_id": "C1", "span_id": "S001", "relation": "supports"}
        ],
        "annotation_status": "gold_adjudicated",
        "reviewer_ids": ["reviewer-a", "reviewer-b"],
        "adjudicator_id": "reviewer-c",
        "corpus_version": "corpus-v1",
        "knowledge_cutoff": "2026-07-26",
    }
    values.update(updates)
    return GoldQuestion.model_validate(values)


class GoldDatasetContractTests(unittest.TestCase):
    def test_duplicate_question_ids_are_rejected(self) -> None:
        question = _question().model_dump_json()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.jsonl"
            path.write_text(f"{question}\n{question}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "question_id"):
                load_gold_dataset(path)

    def test_illegal_corpus_id_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            _span(paper_id="paper-one")

    def test_answerable_question_requires_must_have_claim(self) -> None:
        with self.assertRaisesRegex(ValidationError, "must-have"):
            _question(must_have_claims=[])

    def test_unanswerable_question_rejects_unconditional_must_have_claim(self) -> None:
        with self.assertRaisesRegex(ValidationError, "unanswerable"):
            _question(
                answerable=False,
                unanswerable_reason="冻结语料没有足够证据。",
                evidence_spans=[],
                citation_relations=[],
            )

    def test_unknown_claim_or_span_reference_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "unknown claim"):
            _question(
                citation_relations=[
                    {"claim_id": "missing", "span_id": "S001", "relation": "supports"}
                ]
            )
        with self.assertRaisesRegex(ValidationError, "unknown span"):
            _question(
                citation_relations=[
                    {"claim_id": "C1", "span_id": "missing", "relation": "supports"}
                ]
            )

    def test_gold_status_requires_two_reviewers_and_distinct_adjudicator(self) -> None:
        with self.assertRaisesRegex(ValidationError, "two independent reviewers"):
            _question(reviewer_ids=["reviewer-a"])
        with self.assertRaisesRegex(ValidationError, "adjudicator"):
            _question(adjudicator_id="reviewer-a")

    def test_valid_gold_question_is_frozen_and_strict(self) -> None:
        question = _question()
        self.assertEqual(question.knowledge_cutoff, date(2026, 7, 26))
        with self.assertRaises(ValidationError):
            GoldClaim.model_validate({"claim_id": "C1", "text": "fact", "extra": True})
        with self.assertRaises(ValidationError):
            CitationRelation.model_validate(
                {"claim_id": "C1", "span_id": "S001", "relation": "invented"}
            )


class SourceReplayTests(unittest.TestCase):
    def _write_sources(self, directory: Path) -> tuple[Path, Path]:
        elements = directory / "elements.jsonl"
        chunks = directory / "chunks.jsonl"
        element = {
            "element_id": "element-1",
            "asset_id": "source-v1",
            "corpus_id": "C001",
            "page_number": 1,
            "raw_text": "prefix supported evidence suffix",
        }
        chunk = {
            "chunk_id": "chunk-1",
            "asset_id": "source-v1",
            "corpus_id": "C001",
            "page_start": 1,
            "page_end": 1,
            "element_ids": ["element-1"],
        }
        elements.write_text(json.dumps(element) + "\n", encoding="utf-8")
        chunks.write_text(json.dumps(chunk) + "\n", encoding="utf-8")
        return elements, chunks

    def test_source_replay_accepts_matching_span_and_chunk_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            elements, chunks = self._write_sources(Path(directory))
            report = validate_source_replay([_question()], elements, chunks)
        self.assertEqual(report.question_count, 1)
        self.assertEqual(report.span_count, 1)
        self.assertEqual(report.projected_chunk_count, 1)

    def test_source_replay_rejects_span_hash_mismatch(self) -> None:
        bad_span = _span(span_hash="0" * 64)
        with tempfile.TemporaryDirectory() as directory:
            elements, chunks = self._write_sources(Path(directory))
            with self.assertRaisesRegex(SourceReplayError, "hash"):
                validate_source_replay(
                    [_question(evidence_spans=[bad_span.model_dump(mode="json")])],
                    elements,
                    chunks,
                )

    def test_source_replay_rejects_wrong_page_and_unknown_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            elements, chunks = self._write_sources(Path(directory))
            with self.assertRaisesRegex(SourceReplayError, "page"):
                validate_source_replay(
                    [_question(evidence_spans=[_span(page=2).model_dump(mode="json")])],
                    elements,
                    chunks,
                )
            with self.assertRaisesRegex(SourceReplayError, "unknown chunk"):
                validate_source_replay(
                    [
                        _question(
                            evidence_spans=[
                                _span(projected_chunk_ids=["missing"]).model_dump(mode="json")
                            ]
                        )
                    ],
                    elements,
                    chunks,
                )

    def test_source_replay_rejects_wrong_evidence_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            elements, chunks = self._write_sources(Path(directory))
            with self.assertRaisesRegex(SourceReplayError, "evidence version"):
                validate_source_replay(
                    [
                        _question(
                            evidence_spans=[
                                _span(evidence_version_id="different-asset").model_dump(
                                    mode="json"
                                )
                            ]
                        )
                    ],
                    elements,
                    chunks,
                )


if __name__ == "__main__":
    unittest.main()
