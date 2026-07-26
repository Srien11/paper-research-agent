from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.context.budget import conservative_token_count, estimate_messages
from paper_research_agent.context.models import PromptMessage


class ContextBudgetTests(unittest.TestCase):
    def test_estimator_handles_unspaced_chinese_and_dense_punctuation(self) -> None:
        self.assertGreaterEqual(conservative_token_count("证" * 100), 100)
        self.assertGreaterEqual(conservative_token_count("," * 100), 100)

    def test_message_estimate_includes_roles_and_framing(self) -> None:
        content_only = conservative_token_count("question")
        total = estimate_messages((PromptMessage(role="user", content="question"),))
        self.assertGreater(total, content_only)
