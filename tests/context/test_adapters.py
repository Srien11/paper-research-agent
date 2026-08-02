from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.context.adapters import EvidenceJoinError, join_retrieval_evidence
from paper_research_agent.figures.models import FigureRecord
from paper_research_agent.retrieval.contracts import (
    BilingualRetrievalRun,
    QueryRewriteTrace,
    RetrievalRun,
    SearchHit,
)


def artifacts() -> tuple[RetrievalRun, EvidenceChunk]:
    text = "traceable source"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunk = EvidenceChunk(
        chunk_id="chunk-1",
        asset_id="asset-1",
        corpus_id="C001",
        element_ids=("element-1",),
        page_start=2,
        page_end=2,
        token_start=0,
        token_end=2,
        text=text,
        text_sha256=digest,
        config_sha256="a" * 64,
    )
    hit = SearchHit(
        chunk_id=chunk.chunk_id,
        corpus_id=chunk.corpus_id,
        asset_id=chunk.asset_id,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        text_sha256=digest,
        final_score=0.9,
        final_rank=1,
    )
    run = RetrievalRun(
        query="question",
        variant="C",
        top_k=1,
        hits=(hit,),
        index_id="index",
        config_sha256="b" * 64,
    )
    return run, chunk


class ContextAdapterTests(unittest.TestCase):
    def test_join_preserves_text_and_retrieval_rank(self) -> None:
        run, chunk = artifacts()
        joined = join_retrieval_evidence(run, [chunk])
        self.assertEqual(joined[0].text, chunk.text)
        self.assertEqual(joined[0].final_rank, 1)

    def test_bilingual_run_joins_through_the_same_evidence_boundary(self) -> None:
        baseline, chunk = artifacts()
        run = BilingualRetrievalRun(
            pipeline_id="pipeline",
            original_query="中文问题",
            rewrite=QueryRewriteTrace(
                status="success",
                english_query="English query",
                requested_model="qwen",
                actual_model="qwen",
                prompt_version="v1",
                latency_ms=10,
            ),
            degraded=False,
            top_k=baseline.top_k,
            hits=baseline.hits,
            index_id=baseline.index_id,
            config_sha256=baseline.config_sha256,
            storage_classes={"C001": "redistributable"},
            rights_status="loaded",
        )
        joined = join_retrieval_evidence(run, [chunk])
        self.assertEqual(joined[0].text, chunk.text)

        without_rights = run.model_copy(
            update={"storage_classes": {}, "rights_status": "not_loaded"}
        )
        with self.assertRaisesRegex(EvidenceJoinError, "fails closed"):
            join_retrieval_evidence(without_rights, [chunk])

    def test_missing_or_mismatched_source_fails(self) -> None:
        run, chunk = artifacts()
        with self.assertRaises(EvidenceJoinError):
            join_retrieval_evidence(run, [])
        changed = chunk.model_copy(update={"page_start": 3, "page_end": 3})
        with self.assertRaises(EvidenceJoinError):
            join_retrieval_evidence(run, [changed])

    def test_figure_metadata_is_preserved_from_hit_to_context(self) -> None:
        figure = FigureRecord(
            figure_id="figure-1",
            asset_id="asset-1",
            figure_name="Figure 1",
            page_number=2,
            bbox=(10.0, 20.0, 100.0, 120.0),
            caption="Figure 1. Architecture.",
            image_path="figures/asset-1/p0002.png",
            figure_type="系统架构图",
            summary="展示模块关系。",
            key_findings=("模块 A 连接模块 B",),
            recognition_confidence=0.9,
            model_id="fixture-vision-v1",
            prompt_version="figure-summary-v1",
        )
        text = "图片名称：Figure 1\n视觉摘要：展示模块关系。"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunk = EvidenceChunk(
            chunk_id="figure-chunk",
            asset_id=figure.asset_id,
            corpus_id="C001",
            element_ids=(figure.figure_id,),
            page_start=2,
            page_end=2,
            token_start=0,
            token_end=4,
            text=text,
            text_sha256=digest,
            config_sha256="a" * 64,
            evidence_type="figure_summary",
            content_origin="generated",
            figure=figure,
        )
        hit = SearchHit(
            chunk_id=chunk.chunk_id,
            corpus_id=chunk.corpus_id,
            asset_id=chunk.asset_id,
            page_start=2,
            page_end=2,
            text_sha256=digest,
            evidence_type="figure_summary",
            figure=figure,
            final_score=0.9,
            final_rank=1,
        )
        run = RetrievalRun(
            query="架构图",
            variant="C",
            top_k=1,
            hits=(hit,),
            index_id="index",
            config_sha256="b" * 64,
        )
        joined = join_retrieval_evidence(run, [chunk])
        self.assertEqual(joined[0].figure, figure)
        self.assertEqual(joined[0].evidence_type, "figure_summary")
