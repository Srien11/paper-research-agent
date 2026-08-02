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

from paper_research_agent.memory.config import ShortTermMemoryConfig
from paper_research_agent.memory.models import ShortTermMemoryTurn
from paper_research_agent.memory.store import SQLiteShortTermMemory


def turn(session_id: str, number: int, now: datetime, *, age_hours: int = 0) -> ShortTermMemoryTurn:
    created = now - timedelta(hours=age_hours)
    return ShortTermMemoryTurn(
        turn_id=f"{number:032x}",
        session_id=session_id,
        created_at=created,
        expires_at=created + timedelta(hours=24),
        user_question=f"question {number}",
        standalone_question=f"standalone question {number}",
        assistant_claims=(f"answer {number}",),
        status="answered",
    )


class SQLiteShortTermMemoryTests(unittest.TestCase):
    def test_sessions_are_isolated_and_old_rows_are_expired_and_trimmed(self) -> None:
        now = datetime(2026, 8, 2, 12, tzinfo=UTC)
        config = ShortTermMemoryConfig(
            max_turns_per_session=2,
            context_turn_limit=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            store = SQLiteShortTermMemory(path, config=config)
            store.append(turn("session-a", 0, now, age_hours=25), now=now)
            store.append(turn("session-a", 1, now), now=now)
            store.append(turn("session-a", 2, now + timedelta(minutes=1)), now=now)
            store.append(turn("session-a", 3, now + timedelta(minutes=2)), now=now)
            store.append(turn("session-b", 4, now), now=now)

            recent_a = store.recent("session-a", now=now + timedelta(minutes=3))
            recent_b = store.recent("session-b", now=now + timedelta(minutes=3))
            with closing(sqlite3.connect(path)) as connection:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(memory_turns)")}

        self.assertEqual([item.user_question for item in recent_a], ["question 2", "question 3"])
        self.assertEqual([item.user_question for item in recent_b], ["question 4"])
        self.assertNotIn("evidence_text", columns)
        self.assertNotIn("answer_markdown", columns)
        self.assertNotIn("provider_response", columns)
        self.assertNotIn("context_messages", columns)
