from __future__ import annotations

import unittest
from pathlib import Path

from paper_research_agent.evaluation.mcp_routing_runner import run_fake_mcp_routing_gate

DATASET = (
    Path(__file__).resolve().parents[2] / "evaluation" / "datasets" / "mcp-tool-routing-v1.jsonl"
)


class McpRoutingRunnerTests(unittest.TestCase):
    def test_fake_servers_run_offline_gate_without_recording_prompts_or_payloads(self) -> None:
        report = run_fake_mcp_routing_gate(DATASET)
        self.assertGreaterEqual(report.metrics.route_accuracy, 0.90)
        self.assertEqual(report.metrics.unsafe_tool_call_rate, 0)
        self.assertEqual(report.metrics.offline_graceful_rate, 1)
        self.assertEqual(len(report.records), report.case_count)
        serialized = report.model_dump_json()
        self.assertNotIn('"prompt":', serialized)
        self.assertNotIn('"arguments":', serialized)
        self.assertNotIn('"result":', serialized)


if __name__ == "__main__":
    unittest.main()
