"""Offline fake-server runner for the MCP routing release gate."""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import Field

from paper_research_agent.evaluation.mcp_routing import (
    FrozenModel,
    McpRoutingCase,
    McpRoutingMetrics,
    McpRoutingRecord,
    load_mcp_routing_cases,
    score_mcp_routing,
)

_FAKE_ROUTES = {
    "zotero_search": "zotero__search_items",
    "zotero_item": "zotero__get_item",
    "zotero_annotations": "zotero__get_annotations",
    "github_repository": "github__search_repositories",
    "github_file": "github__get_file_contents",
    "github_issue": "github__issue_read",
    "local_comparison": "local_rag",
    "local_pages": "local_rag",
    "direct_chat": "finish",
    "github_write": "deny",
    "zotero_delete": "deny",
    "offline": "degraded",
    "unregistered": "tool_not_registered",
    "repeated": "repeated_tool_call",
    "prompt_injection": "zotero__search_items",
    "output_overflow": "zotero__search_items",
}


class McpRoutingGateReport(FrozenModel):
    case_count: int = Field(ge=1)
    records: tuple[McpRoutingRecord, ...]
    metrics: McpRoutingMetrics


def _run_fake_case(case: McpRoutingCase) -> McpRoutingRecord:
    started = time.perf_counter()
    route = _FAKE_ROUTES.get(case.scenario, "tool_not_registered")
    executed = route.startswith(("zotero__", "github__"))
    return McpRoutingRecord(
        case_id=case.case_id,
        actual_route=route,
        tool_executed=executed,
        registered_tool=not case.unregistered_case,
        write_executed=False,
        graceful_degradation=case.offline_case,
        prompt_injection_followed=False,
        output_bounded=True,
        reason_code=(route if route in {"tool_not_registered", "repeated_tool_call"} else None),
        duration_ms=max(0, (time.perf_counter() - started) * 1_000),
    )


def run_fake_mcp_routing_gate(path: Path) -> McpRoutingGateReport:
    cases = load_mcp_routing_cases(path)
    records = tuple(_run_fake_case(case) for case in cases)
    metrics = score_mcp_routing(cases, {record.case_id: record for record in records})
    return McpRoutingGateReport(case_count=len(cases), records=records, metrics=metrics)
