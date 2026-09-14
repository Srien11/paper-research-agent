from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper_research_agent.web.app import _spool_bounded_request
from paper_research_agent.web.interventions import InterventionStore
from paper_research_agent.web.knowledge import (
    KnowledgeBaseStore,
    KnowledgeItemPatch,
    publish_item,
)


class _ChunkedRequest:
    def __init__(self, *, headers: dict[str, str], chunks: tuple[bytes, ...]):
        self.headers = headers
        self.chunks = chunks
        self.stream_started = False

    async def stream(self):
        self.stream_started = True
        for chunk in self.chunks:
            yield chunk


class KnowledgeBaseStoreTests(unittest.TestCase):
    @staticmethod
    def _ready_item(
        store: KnowledgeBaseStore,
        corpus: Path,
    ):
        item = store.stage(filename="new.pdf", chunks=(b"%PDF-1.4",))
        return store.update(
            item.item_id,
            KnowledgeItemPatch(
                title="A paper",
                authors=("A",),
                year=2026,
                official_url="https://example.test/paper",
            ),
            corpus_id=store.next_corpus_id(corpus),
        )

    def test_staging_requires_review_before_publish_and_keeps_stable_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "core_frozen.jsonl").write_text(
                '{"corpus_id":"C001","corpus_version":"v1"}\n', encoding="utf-8"
            )
            store = KnowledgeBaseStore(root / "runtime")
            item = store.stage(filename="new.pdf", chunks=(b"%PDF-1.4",))
            self.assertEqual(item.status, "staged")
            self.assertEqual(item.sha256, hashlib.sha256(b"%PDF-1.4").hexdigest())
            reviewed = store.update(
                item.item_id,
                KnowledgeItemPatch(
                    title="A paper", authors=("A",), year=2026,
                    official_url="https://example.test/paper",
                ),
                corpus_id=store.next_corpus_id(corpus),
            )
            self.assertEqual(reviewed.status, "ready")
            self.assertEqual(reviewed.corpus_id, "C002")
            self.assertEqual(store.get(item.item_id).corpus_id, "C002")

    def test_queued_item_failure_reaches_failed_and_can_be_requeued(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "core_frozen.jsonl").write_text(
                '{"corpus_id":"C001","corpus_version":"v1"}\n',
                encoding="utf-8",
            )
            store = KnowledgeBaseStore(root / "runtime")
            item = self._ready_item(store, corpus)

            queued = store.queue_for_publish(item.item_id)
            self.assertEqual(queued.status, "queued")
            with (
                patch(
                    "paper_research_agent.web.knowledge._freeze_source",
                    side_effect=RuntimeError("forced publish failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "forced publish failure"),
            ):
                publish_item(
                    store,
                    item.item_id,
                    corpus_dir=corpus,
                    output_root=root / "processed",
                    chunking_config=root / "chunking.json",
                    retrieval_config=root / "retrieval.json",
                )

            self.assertEqual(store.get(item.item_id).status, "failed")
            self.assertEqual(store.queue_for_publish(item.item_id).status, "queued")

    def test_only_ready_or_failed_item_can_be_queued(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            store = KnowledgeBaseStore(root / "runtime")
            item = store.stage(filename="new.pdf", chunks=(b"%PDF-1.4",))

            with self.assertRaisesRegex(ValueError, "不能进入发布队列"):
                store.queue_for_publish(item.item_id)

            ready = self._ready_item(store, corpus)
            store.queue_for_publish(ready.item_id)
            with self.assertRaisesRegex(ValueError, "不能进入发布队列"):
                store.queue_for_publish(ready.item_id)

    def test_interventions_are_durable_and_conversation_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = InterventionStore(Path(directory) / "interventions.sqlite3")
            queued = store.queue(request_id="r" * 16, conversation_id="c1", message="先检查引用")
            self.assertEqual(store.list(request_id="r" * 16, conversation_id="c2"), ())
            paused = store.transition(queued.intervention_id, "awaiting_confirmation")
            self.assertEqual(paused.status, "awaiting_confirmation")


class KnowledgeUploadBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_spool_rejects_chunked_overflow_and_closes_temporary_file(
        self,
    ) -> None:
        request = _ChunkedRequest(headers={}, chunks=(b"123", b"456"))

        with self.assertRaisesRegex(ValueError, "100 MiB"):
            async with _spool_bounded_request(request, max_bytes=5):
                pass

        self.assertTrue(request.stream_started)

    async def test_declared_overflow_is_rejected_before_streaming(self) -> None:
        request = _ChunkedRequest(
            headers={"content-length": "6"},
            chunks=(b"123456",),
        )

        with self.assertRaisesRegex(ValueError, "100 MiB"):
            async with _spool_bounded_request(request, max_bytes=5):
                pass

        self.assertFalse(request.stream_started)

    async def test_small_request_is_spooled_and_rewound(self) -> None:
        request = _ChunkedRequest(headers={}, chunks=(b"123", b"456"))

        async with _spool_bounded_request(request, max_bytes=6) as spool:
            self.assertEqual(spool.read(), b"123456")


if __name__ == "__main__":
    unittest.main()
