from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from paper_research_agent.agent.orchestrator.factory import (
    build_main_agent_runtime_from_model,
)


class MainAgentFactoryTests(unittest.TestCase):
    def test_model_factory_passes_memory_provider_to_hydrator(self) -> None:
        provider = Mock()
        event_sink = Mock()
        with patch(
            "paper_research_agent.agent.orchestrator.factory.build_main_agent_runtime",
            return_value=Mock(),
        ) as build:
            build_main_agent_runtime_from_model(
                store=Mock(),
                model=Mock(),
                dispatcher=Mock(),
                memory_provider=provider,
                event_sink=event_sink,
            )

        self.assertIs(build.call_args.kwargs["hydrator"].memory_provider, provider)
        self.assertIs(build.call_args.kwargs["hydrator"].event_sink, event_sink)

    def test_model_factory_passes_fast_path_flag_to_graph_factory(self) -> None:
        with patch(
            "paper_research_agent.agent.orchestrator.factory.build_main_agent_runtime",
            return_value=Mock(),
        ) as build:
            build_main_agent_runtime_from_model(
                store=Mock(),
                model=Mock(),
                dispatcher=Mock(),
                fast_path_enabled=True,
            )

        self.assertTrue(build.call_args.kwargs["fast_path_enabled"])


if __name__ == "__main__":
    unittest.main()
