from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from paper_research_agent.web.interventions import InterventionStore
from paper_research_agent.web.knowledge import KnowledgeBaseStore, KnowledgeItemPatch


class KnowledgeBaseStoreTests(unittest.TestCase):
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

    def test_interventions_are_durable_and_conversation_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = InterventionStore(Path(directory) / "interventions.sqlite3")
            queued = store.queue(request_id="r" * 16, conversation_id="c1", message="先检查引用")
            self.assertEqual(store.list(request_id="r" * 16, conversation_id="c2"), ())
            paused = store.transition(queued.intervention_id, "awaiting_confirmation")
            self.assertEqual(paused.status, "awaiting_confirmation")


if __name__ == "__main__":
    unittest.main()
