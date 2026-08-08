from __future__ import annotations

import unittest

from paper_research_agent.agent.tooling.catalog import (
    EXTENDED_TOOL_NAMES,
    EXTENDED_TOOL_SPECS,
    TOOL_SPEC_BY_NAME,
    effective_tool_spec,
)
from paper_research_agent.agent.tooling.contracts import TOOL_INPUT_SCHEMAS


class ToolCatalogTests(unittest.TestCase):
    def test_catalog_contains_exactly_eighteen_strict_tools(self) -> None:
        self.assertEqual(len(EXTENDED_TOOL_NAMES), 18)
        self.assertNotIn("compare_papers", EXTENDED_TOOL_NAMES)
        self.assertNotIn("compare_papers", TOOL_INPUT_SCHEMAS)
        self.assertEqual(set(TOOL_INPUT_SCHEMAS), set(EXTENDED_TOOL_NAMES))
        self.assertEqual(set(TOOL_SPEC_BY_NAME), set(EXTENDED_TOOL_NAMES))
        self.assertEqual(
            {spec.name for spec in EXTENDED_TOOL_SPECS if spec.approval_required},
            {
                "save_research_note",
                "export_research_report",
                "manage_long_term_memory",
            },
        )

    def test_long_term_memory_search_resolves_to_read_only_context(self) -> None:
        spec = TOOL_SPEC_BY_NAME["manage_long_term_memory"]

        effective = effective_tool_spec(spec, {"action": "search", "query": "RAG"})

        self.assertEqual(effective.risk, "local_read")
        self.assertEqual(effective.trust, "research_context")
        self.assertFalse(effective.approval_required)
        self.assertEqual(effective_tool_spec(spec, {"action": "delete"}).risk, "write")


if __name__ == "__main__":
    unittest.main()
