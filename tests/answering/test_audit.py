from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.answering.audit import SQLiteAnswerAuditLogger
from paper_research_agent.answering.models import AnswerCitation, AnswerClaim, RAGAnswer


class AnswerAuditTests(unittest.TestCase):
    def test_audit_stores_only_hashes_metadata_and_usage(self) -> None:
        sentinel = "PRIVATE_EVIDENCE_MUST_NEVER_BE_STORED"
        answer = RAGAnswer(
            status="answered",
            answer_markdown=f"结论不包含原文 {sentinel}。[E1]",
            claims=(AnswerClaim(text=f"结论不包含原文 {sentinel}。", citation_ids=("E1",)),),
            citations=(
                AnswerCitation(
                    citation_id="E1",
                    chunk_id="chunk-1",
                    corpus_id="C001",
                    asset_id="asset-1",
                    page_start=1,
                    page_end=1,
                    text_sha256="a" * 64,
                    storage_class="internal_research_only",
                ),
            ),
            requested_model="qwen3.7-plus-2026-05-26",
            actual_model="qwen3.7-plus-2026-05-26",
            prompt_version="rag-answer-json-v1",
            input_tokens=100,
            output_tokens=20,
            latency_ms=10,
            attempts=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.sqlite3"
            logger = SQLiteAnswerAuditLogger(path)
            self.assertTrue(logger.log(answer))
            with closing(sqlite3.connect(path)) as connection:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(answer_audit)").fetchall()
                }
                row_count = connection.execute("SELECT COUNT(*) FROM answer_audit").fetchone()[0]
            self.assertEqual(row_count, 1)
            self.assertNotIn("answer", columns)
            self.assertNotIn("context", columns)
            self.assertNotIn("messages", columns)
            self.assertIn("answer_sha256", columns)
            self.assertNotIn(sentinel.encode(), path.read_bytes())


if __name__ == "__main__":
    unittest.main()
