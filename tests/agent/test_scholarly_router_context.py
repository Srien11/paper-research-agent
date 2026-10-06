from __future__ import annotations

import json
import unittest

from paper_research_agent.agent.dynamic.models import ToolObservation
from paper_research_agent.agent.dynamic.router import _bounded_observation_json
from paper_research_agent.agent.tooling.contracts import ToolExecutionResult


class ScholarlyRouterContextTests(unittest.TestCase):
    def test_large_abstracts_do_not_erase_titles_ids_or_official_links(self):
        items = tuple(
            {
                "title": f"Public paper {index}",
                "paper_id": f"doi:10.1234/{index}",
                "url": f"https://doi.org/10.1234/{index}",
                "external_ids": {"DOI": f"10.1234/{index}"},
                "abstract": "long abstract " * 2000,
            }
            for index in range(10)
        )
        observation = ToolObservation(
            sequence=1,
            decision_fingerprint="a" * 64,
            tool_name="search_scholarly_sources",
            purpose="查询公开书目",
            result=ToolExecutionResult(
                tool_name="search_scholarly_sources",
                items=items,
                summary={"provider": "crossref", "returned_count": 10},
            ),
        )
        encoded = _bounded_observation_json((observation, observation))
        self.assertLessEqual(len(encoded), 12_000)
        projected = json.loads(encoded)
        self.assertEqual(projected[0]["summary"]["provider"], "crossref")
        self.assertEqual(projected[0]["items"][0]["title"], "Public paper 0")
        self.assertEqual(projected[0]["items"][0]["url"], "https://doi.org/10.1234/0")
        self.assertEqual(projected[0]["items"][0]["external_ids"]["DOI"], "10.1234/0")

    def test_bounded_summary_does_not_introduce_scholarly_citation_evidence(self):
        observation = ToolObservation(
            sequence=1,
            decision_fingerprint="a" * 64,
            tool_name="search_scholarly_sources",
            purpose="查询书目",
            result=ToolExecutionResult(
                tool_name="search_scholarly_sources",
                items=({"title": "Paper", "abstract": "data" * 10000},),
            ),
        )
        projected = json.loads(_bounded_observation_json((observation,)))
        self.assertEqual(projected[0]["trust"], "research_context")
