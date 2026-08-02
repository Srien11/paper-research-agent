from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.retrieval.contracts import QueryRewriteTrace
from paper_research_agent.retrieval.query_store import (
    AuditRanking,
    QueryAuditRecord,
    SQLiteQueryAuditLogger,
    SQLiteQueryRewriteCache,
    query_sha256,
)


class QueryRewriteCacheTests(unittest.TestCase):
    def test_lookup_distinguishes_fresh_stale_and_expired_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = SQLiteQueryRewriteCache(Path(directory) / "rewrite.sqlite")
            cache.put(
                "中文问题",
                model="qwen3.7-plus",
                prompt_version="query-rewrite-v1",
                english_query="research question",
                actual_model="qwen3.7-plus-2026-05-26",
                latency_ms=123.5,
                input_tokens=20,
                output_tokens=5,
                stale_days=30,
            )

            fresh = cache.lookup(
                "中文问题",
                model="qwen3.7-plus",
                prompt_version="query-rewrite-v1",
                fresh_days=1,
                stale_days=3,
            )
            self.assertIsNotNone(fresh.fresh)
            self.assertIsNone(fresh.stale)
            assert fresh.fresh is not None
            self.assertEqual(fresh.fresh.english_query, "research question")

            self._set_created_at(cache.path, datetime.now(UTC) - timedelta(days=2))
            stale = cache.lookup(
                "中文问题",
                model="qwen3.7-plus",
                prompt_version="query-rewrite-v1",
                fresh_days=1,
                stale_days=3,
            )
            self.assertIsNone(stale.fresh)
            self.assertIsNotNone(stale.stale)

            self._set_created_at(cache.path, datetime.now(UTC) - timedelta(days=4))
            expired = cache.lookup(
                "中文问题",
                model="qwen3.7-plus",
                prompt_version="query-rewrite-v1",
                fresh_days=1,
                stale_days=3,
            )
            self.assertIsNone(expired.fresh)
            self.assertIsNone(expired.stale)

    @staticmethod
    def _set_created_at(path: Path, created_at: datetime) -> None:
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(
                "UPDATE rewrites SET created_at = ?",
                (created_at.isoformat(),),
            )


class QueryAuditLoggerTests(unittest.TestCase):
    def test_audit_schema_and_rows_do_not_store_evidence_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.sqlite"
            logger = SQLiteQueryAuditLogger(path, plaintext_days=7)
            self.assertTrue(logger.write(self._record(request_id="request-1")))

            with closing(sqlite3.connect(path)) as connection:
                run_columns = {
                    str(row[1]) for row in connection.execute("PRAGMA table_info(runs)").fetchall()
                }
                ranking_columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(rankings)").fetchall()
                }
                ranking = connection.execute(
                    "SELECT stage, chunk_id, rank, score, final_rank FROM rankings"
                ).fetchone()

            forbidden_columns = {
                "text",
                "evidence_text",
                "figure_json",
                "caption",
                "request_body",
                "response_body",
            }
            self.assertTrue(forbidden_columns.isdisjoint(run_columns))
            self.assertTrue(forbidden_columns.isdisjoint(ranking_columns))
            self.assertEqual(ranking, ("final", "chunk-1", 1, 0.75, 1))
            with self.assertRaises(TypeError):
                AuditRanking(
                    stage="final",
                    chunk_id="chunk-2",
                    rank=2,
                    score=0.5,
                    evidence_text="forbidden evidence body",  # type: ignore[call-arg]
                )

    def test_write_clears_expired_plaintext_but_retains_hash_and_rankings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.sqlite"
            logger = SQLiteQueryAuditLogger(path, plaintext_days=7)
            old_record = self._record(
                request_id="old-request",
                created_at=datetime.now(UTC) - timedelta(days=8),
            )
            recent_record = self._record(request_id="recent-request")

            self.assertTrue(logger.write(old_record))
            self.assertTrue(logger.write(recent_record))

            with closing(sqlite3.connect(path)) as connection:
                old_row = connection.execute(
                    """SELECT query_sha256, original_query, rewritten_query
                       FROM runs WHERE request_id = ?""",
                    (old_record.request_id,),
                ).fetchone()
                recent_row = connection.execute(
                    """SELECT original_query, rewritten_query
                       FROM runs WHERE request_id = ?""",
                    (recent_record.request_id,),
                ).fetchone()
                old_ranking_count = connection.execute(
                    "SELECT COUNT(*) FROM rankings WHERE request_id = ?",
                    (old_record.request_id,),
                ).fetchone()

            self.assertEqual(
                old_row,
                (query_sha256(old_record.original_query), None, None),
            )
            self.assertEqual(
                recent_row,
                (recent_record.original_query, recent_record.rewrite.english_query),
            )
            self.assertEqual(old_ranking_count, (1,))

    @staticmethod
    def _record(
        *,
        request_id: str,
        created_at: datetime | None = None,
    ) -> QueryAuditRecord:
        return QueryAuditRecord(
            request_id=request_id,
            created_at=created_at or datetime.now(UTC),
            original_query="原始中文问题",
            rewrite=QueryRewriteTrace(
                status="success",
                english_query="original research question",
                requested_model="qwen3.7-plus",
                actual_model="qwen3.7-plus-2026-05-26",
                prompt_version="query-rewrite-v1",
                latency_ms=150.0,
                input_tokens=22,
                output_tokens=6,
            ),
            pipeline_id="bilingual-v1",
            index_id="idx-test",
            config_sha256="a" * 64,
            degraded_reason=None,
            latency_ms={"total": 200.0},
            rankings=(
                AuditRanking(
                    stage="final",
                    chunk_id="chunk-1",
                    rank=1,
                    score=0.75,
                    final_rank=1,
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
