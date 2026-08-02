from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.memory.config import ShortTermMemoryConfig, load_memory_config
from paper_research_agent.memory.models import MemorySourceRef, ShortTermMemoryTurn


class ShortTermMemoryModelTests(unittest.TestCase):
    def test_checked_in_config_is_safe_and_bounded(self) -> None:
        config = load_memory_config(PROJECT_ROOT / "configs/memory/short-term-v1.json")
        self.assertEqual(config.ttl_hours, 24)
        self.assertEqual(config.max_turns_per_session, 20)
        self.assertEqual(config.context_turn_limit, 6)
        self.assertEqual(config.context_token_budget, 1200)
        self.assertEqual(config.protected_evidence_count, 3)
        self.assertEqual(config.store_path.as_posix(), "data/runtime/short-term-memory-v1.sqlite3")

    def test_unsafe_path_and_invalid_limits_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ShortTermMemoryConfig(store_path=Path("C:/memory.sqlite3"))
        with self.assertRaises(ValidationError):
            ShortTermMemoryConfig(context_turn_limit=21, max_turns_per_session=20)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"unknown": True}), encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_memory_config(path)

    def test_turn_rejects_unsafe_session_and_extra_evidence_body(self) -> None:
        now = datetime.now(UTC)
        source = MemorySourceRef(
            chunk_id="chunk-1",
            corpus_id="C001",
            text_sha256="a" * 64,
            storage_class="redistributable",
        )
        turn = ShortTermMemoryTurn(
            turn_id="a" * 32,
            session_id="research-session_1",
            created_at=now,
            expires_at=now + timedelta(hours=24),
            user_question="前一个问题",
            standalone_question="前一个独立问题",
            assistant_claims=("前一个回答。",),
            status="answered",
            source_refs=(source,),
        )
        self.assertEqual(turn.source_refs[0].chunk_id, "chunk-1")
        with self.assertRaises(ValidationError):
            turn.model_copy(update={"session_id": "../escape"}, deep=True).__class__.model_validate(
                {**turn.model_dump(), "session_id": "../escape"}
            )
        with self.assertRaises(ValidationError):
            ShortTermMemoryTurn.model_validate({**turn.model_dump(), "evidence_text": "secret"})
        with self.assertRaises(ValidationError):
            ShortTermMemoryTurn.model_validate(
                {**turn.model_dump(), "assistant_claims": ("x" * 1001,)}
            )
