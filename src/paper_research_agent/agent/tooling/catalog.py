"""Frozen capability and risk catalog for the extended research tool set."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ToolRisk = Literal["local_read", "network_read", "restricted_compute", "write"]
ToolTrust = Literal["citation_evidence", "research_context", "computed_result", "side_effect"]


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    risk: ToolRisk
    trust: ToolTrust
    timeout_seconds: float = Field(gt=0, le=60)
    approval_required: bool = False
    max_result_items: int = Field(ge=1, le=100)
    description: str = Field(min_length=1, max_length=500)


class ExtendedToolPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    network_read_enabled: bool = True
    restricted_compute_enabled: bool = True
    write_enabled: bool = True

    def authorize(self, spec: ToolSpec) -> None:
        enabled = {
            "local_read": True,
            "network_read": self.network_read_enabled,
            "restricted_compute": self.restricted_compute_enabled,
            "write": self.write_enabled,
        }[spec.risk]
        if not enabled:
            raise PermissionError(f"extended tool risk is disabled: {spec.risk}")


def _spec(
    name: str,
    risk: ToolRisk,
    timeout: float,
    limit: int,
    description: str,
    trust: ToolTrust | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        risk=risk,
        trust=trust or _default_trust(risk),
        timeout_seconds=timeout,
        approval_required=risk == "write",
        max_result_items=limit,
        description=description,
    )


def _default_trust(risk: ToolRisk) -> ToolTrust:
    if risk == "restricted_compute":
        return "computed_result"
    if risk == "write":
        return "side_effect"
    return "research_context"


def effective_tool_spec(spec: ToolSpec, arguments: Mapping[str, Any]) -> ToolSpec:
    """Resolve the read-only branch of an otherwise mixed-risk tool."""
    if spec.name == "manage_long_term_memory" and arguments.get("action") in {
        "search",
        "list",
    }:
        return spec.model_copy(
            update={
                "risk": "local_read",
                "trust": "research_context",
                "approval_required": False,
            }
        )
    return spec


EXTENDED_TOOL_SPECS: tuple[ToolSpec, ...] = (
    _spec(
        "get_adjacent_chunks",
        "local_read",
        2,
        7,
        "Read bounded neighboring chunks.",
        "citation_evidence",
    ),
    _spec("get_paper_metadata", "local_read", 2, 20, "Read frozen paper metadata."),
    _spec(
        "trace_evidence_source", "local_read", 2, 20, "Trace chunk provenance.", "citation_evidence"
    ),
    _spec("get_paper_outline", "local_read", 2, 100, "Read a paper section outline."),
    _spec("search_scholarly_sources", "network_read", 10, 20, "Search scholarly metadata."),
    _spec("resolve_paper_identifier", "network_read", 10, 5, "Resolve a DOI, title, or paper ID."),
    _spec("get_citation_graph", "network_read", 10, 50, "Read references or citations."),
    _spec("check_paper_status", "network_read", 10, 10, "Check publication status."),
    _spec(
        "extract_table", "local_read", 3, 20, "Read bounded table elements.", "citation_evidence"
    ),
    _spec(
        "inspect_figure", "local_read", 3, 20, "Read stored figure semantics.", "citation_evidence"
    ),
    _spec(
        "extract_equation",
        "local_read",
        3,
        20,
        "Read bounded formula elements.",
        "citation_evidence",
    ),
    _spec("calculate", "restricted_compute", 2, 1, "Evaluate safe arithmetic."),
    _spec("analyze_experiment_data", "restricted_compute", 5, 20, "Run fixed statistics."),
    _spec("verify_claim", "local_read", 10, 20, "Gather claim evidence.", "citation_evidence"),
    _spec("check_reproducibility", "local_read", 5, 20, "Inspect reproducibility signals."),
    _spec("save_research_note", "write", 5, 1, "Save one approved research note."),
    _spec("export_research_report", "write", 10, 1, "Export one approved report."),
    _spec("manage_long_term_memory", "write", 5, 20, "Manage approved long-term memory."),
)

EXTENDED_TOOL_NAMES = frozenset(spec.name for spec in EXTENDED_TOOL_SPECS)
TOOL_SPEC_BY_NAME = {spec.name: spec for spec in EXTENDED_TOOL_SPECS}

if len(EXTENDED_TOOL_SPECS) != 18 or len(EXTENDED_TOOL_NAMES) != 18:
    raise RuntimeError("extended research tool catalog must contain exactly 18 unique tools")
