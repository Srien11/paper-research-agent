from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.answering.config import AnsweringConfig, load_answering_config


class AnsweringConfigTests(unittest.TestCase):
    def test_production_defaults_are_stable_and_versioned(self) -> None:
        config = AnsweringConfig()
        self.assertEqual(config.model, "qwen3.7-plus-2026-05-26")
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.top_p, 0.7)
        self.assertEqual(config.max_output_tokens, 1200)
        self.assertFalse(config.enable_thinking)
        self.assertEqual(config.timeout_seconds, 30)
        self.assertEqual(config.max_retries, 2)

    def test_unversioned_model_and_unknown_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AnsweringConfig(model="qwen3.7-plus")
        with self.assertRaises(ValidationError):
            AnsweringConfig(unknown=True)  # type: ignore[call-arg]

    def test_parameter_bounds_are_rejected(self) -> None:
        invalid = (
            {"temperature": -0.1},
            {"top_p": 0},
            {"max_output_tokens": 0},
            {"timeout_seconds": 0},
            {"max_retries": 6},
            {"enable_thinking": True},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                AnsweringConfig(**values)  # type: ignore[arg-type]

    def test_loads_versioned_json_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answering.json"
            path.write_text(json.dumps(AnsweringConfig().model_dump()), encoding="utf-8")
            self.assertEqual(load_answering_config(path), AnsweringConfig())


if __name__ == "__main__":
    unittest.main()
