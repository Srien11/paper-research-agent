from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.retrieval.contracts import (
    BilingualRetrievalRun,
    QueryRewriteTrace,
    RetrievalRun,
    SearchHit,
)


class RetrievalContractTests(unittest.TestCase):
    def test_non_contiguous_final_ranks_are_rejected(self) -> None:
        hit = SearchHit(
            chunk_id="c",
            corpus_id="C001",
            asset_id="a",
            page_start=1,
            page_end=1,
            text_sha256="0" * 64,
            final_score=1,
            final_rank=2,
        )
        with self.assertRaises(ValidationError):
            RetrievalRun(
                query="q",
                variant="A",
                top_k=1,
                hits=(hit,),
                index_id="i",
                config_sha256="1" * 64,
            )

    def test_bilingual_degradation_must_match_rewrite_failure(self) -> None:
        with self.assertRaises(ValidationError):
            BilingualRetrievalRun(
                pipeline_id="p",
                original_query="中文",
                rewrite=QueryRewriteTrace(
                    status="timeout",
                    requested_model="qwen",
                    prompt_version="v1",
                    latency_ms=1,
                    error_class="TimeoutError",
                ),
                degraded=False,
                top_k=1,
                hits=(),
                index_id="i",
                config_sha256="1" * 64,
                rights_status="not_loaded",
            )
