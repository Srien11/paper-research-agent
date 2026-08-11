"""Provider-aware immutable registry for built-in and admitted MCP tools."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast

from paper_research_agent.agent.tooling.catalog import EXTENDED_TOOL_SPECS, ToolSpec
from paper_research_agent.agent.tooling.contracts import TOOL_INPUT_SCHEMAS, ToolExecutionResult

ToolProviderKind = Literal["builtin", "mcp"]
_PUBLIC_NAME = re.compile(r"^[a-z][a-z0-9_]{1,127}$")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return copy.deepcopy(value)


@dataclass(frozen=True)
class RegisteredTool:
    public_name: str
    provider_id: str
    provider_kind: ToolProviderKind
    remote_name: str
    spec: ToolSpec
    input_schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        if _PUBLIC_NAME.fullmatch(self.public_name) is None:
            raise ValueError("invalid registered tool public name")
        if not self.provider_id:
            raise ValueError("registered tool provider ID is required")
        if self.spec.name != self.public_name:
            raise ValueError("registered tool spec name must match its public name")
        if self.provider_kind == "builtin":
            if self.provider_id != "builtin" or self.remote_name != self.public_name:
                raise ValueError("built-in tools must retain their original public names")
        else:
            if not self.public_name.startswith(f"{self.provider_id}__"):
                raise ValueError("MCP public name must use the exact server namespace")
            if self.spec.trust == "citation_evidence":
                raise ValueError("MCP tools cannot be citation evidence")
            if self.spec.risk not in {"local_read", "network_read"}:
                raise ValueError("MCP tools must be read-only")
            if self.spec.approval_required:
                raise ValueError("read-only MCP tools cannot require approval")
        object.__setattr__(self, "input_schema", cast(Mapping[str, Any], _deep_freeze(self.input_schema)))


class ToolProvider(Protocol):
    provider_id: str

    async def execute(
        self,
        tool: RegisteredTool,
        arguments: dict[str, Any],
        *,
        run_id: str,
    ) -> ToolExecutionResult: ...


class ToolRegistrySnapshot:
    def __init__(
        self,
        tools: Mapping[str, RegisteredTool],
        providers: Mapping[str, ToolProvider],
    ) -> None:
        copied_tools = dict(tools)
        copied_providers = dict(providers)
        for provider_id, provider in copied_providers.items():
            if provider_id != provider.provider_id:
                raise ValueError("provider mapping key must match provider ID")
        for public_name, tool in copied_tools.items():
            if public_name != tool.public_name:
                raise ValueError("tool mapping key must match public name")
            if tool.provider_id not in copied_providers:
                raise ValueError(f"registered tool provider is unavailable: {tool.provider_id}")
        self._tools = MappingProxyType(copied_tools)
        self._providers = MappingProxyType(copied_providers)

    @property
    def tools(self) -> Mapping[str, RegisteredTool]:
        return self._tools

    @property
    def providers(self) -> Mapping[str, ToolProvider]:
        return self._providers

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def list_tools(self) -> tuple[RegisteredTool, ...]:
        return tuple(self._tools[name] for name in sorted(self._tools))

    def resolve(self, name: str) -> tuple[RegisteredTool, ToolProvider]:
        tool = self._tools.get(name)
        if tool is None:
            raise PermissionError(f"tool is not registered: {name}")
        return tool, self._providers[tool.provider_id]

    def validate_arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool, _provider = self.resolve(name)
        if tool.provider_kind == "builtin":
            schema = TOOL_INPUT_SCHEMAS[tool.remote_name]
            return schema.model_validate(arguments).model_dump(mode="python")
        # jsonschema remains an optional dependency and is imported only for an admitted MCP call.
        from paper_research_agent.agent.mcp.provider import validate_mcp_arguments

        return validate_mcp_arguments(tool, arguments)


def builtin_registry_snapshot(provider: ToolProvider) -> ToolRegistrySnapshot:
    if provider.provider_id != "builtin":
        raise ValueError("built-in provider must use provider ID 'builtin'")
    tools = {
        spec.name: RegisteredTool(
            public_name=spec.name,
            provider_id="builtin",
            provider_kind="builtin",
            remote_name=spec.name,
            spec=spec,
            input_schema=TOOL_INPUT_SCHEMAS[spec.name].model_json_schema(),
        )
        for spec in EXTENDED_TOOL_SPECS
    }
    return ToolRegistrySnapshot(tools, {"builtin": provider})
