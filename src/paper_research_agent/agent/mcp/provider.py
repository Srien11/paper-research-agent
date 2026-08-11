"""Allowlist admission and execution provider for one MCP server."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from paper_research_agent.agent.mcp.config import McpStdioServerConfig, McpToolAdmission
from paper_research_agent.agent.mcp.normalizer import normalize_mcp_result
from paper_research_agent.agent.tooling.catalog import ToolSpec
from paper_research_agent.agent.tooling.contracts import ToolExecutionResult
from paper_research_agent.agent.tooling.registry import RegisteredTool


class McpManagerGateway(Protocol):
    def tools_for(self, server_id: str) -> tuple[Any, ...]: ...

    def degrade(self, server_id: str, reason_code: str) -> None: ...

    async def call_tool(
        self,
        server_id: str,
        remote_name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> Any: ...


def _remote_field(remote: Any, name: str, default: Any = None) -> Any:
    if isinstance(remote, Mapping):
        return remote.get(name, default)
    return getattr(remote, name, default)


def _inspect_schema(value: Any, *, depth: int = 0, counter: list[int] | None = None) -> None:
    if depth > 12:
        raise ValueError("MCP schema exceeds maximum depth")
    nodes = counter if counter is not None else [0]
    nodes[0] += 1
    if nodes[0] > 500:
        raise ValueError("MCP schema exceeds maximum node count")
    if isinstance(value, Mapping):
        if "$ref" in value:
            raise ValueError("MCP schema references are forbidden")
        properties = value.get("properties")
        if isinstance(properties, Mapping) and len(properties) > 50:
            raise ValueError("MCP schema exceeds maximum property count")
        for key, item in value.items():
            if len(str(key)) > 500:
                raise ValueError("MCP schema string is too long")
            _inspect_schema(item, depth=depth + 1, counter=nodes)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _inspect_schema(item, depth=depth + 1, counter=nodes)
    elif isinstance(value, str) and len(value) > 10_000:
        raise ValueError("MCP schema string is too long")


def _close_object_schemas(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object" or "properties" in value:
            value.setdefault("additionalProperties", False)
        for item in value.values():
            _close_object_schemas(item)
    elif isinstance(value, list):
        for item in value:
            _close_object_schemas(item)


def admit_mcp_schema(raw_schema: Any) -> dict[str, Any]:
    if not isinstance(raw_schema, Mapping):
        raise TypeError("MCP input schema must be an object")
    schema = copy.deepcopy(dict(raw_schema))
    if len(json.dumps(schema, ensure_ascii=False).encode("utf-8")) > 65_536:
        raise ValueError("MCP schema exceeds maximum byte size")
    _inspect_schema(schema)
    if schema.get("type") != "object":
        raise ValueError("MCP input schema root must be an object")
    _close_object_schemas(schema)
    Draft202012Validator.check_schema(schema)
    return schema


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def validate_mcp_arguments(tool: RegisteredTool, arguments: dict[str, Any]) -> dict[str, Any]:
    schema = _thaw(tool.input_schema)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(arguments), key=lambda item: list(item.path))
    if errors:
        raise ValueError("MCP tool arguments failed local schema validation")
    return copy.deepcopy(arguments)


class McpToolProvider:
    def __init__(self, config: McpStdioServerConfig, manager: McpManagerGateway) -> None:
        self.config = config
        self.manager = manager
        self.provider_id = config.server_id
        self._admissions = {tool.public_name: tool for tool in config.tools}
        self._discovered: tuple[RegisteredTool, ...] = ()

    def discover(self) -> tuple[RegisteredTool, ...]:
        remote_by_name = {
            str(_remote_field(remote, "name")): remote
            for remote in self.manager.tools_for(self.provider_id)
        }
        if any(tool.remote_name not in remote_by_name for tool in self.config.tools):
            self.manager.degrade(self.provider_id, "mcp_tool_missing")
            self._discovered = ()
            return ()
        registered: list[RegisteredTool] = []
        try:
            for admission in self.config.tools:
                remote = remote_by_name[admission.remote_name]
                schema = admit_mcp_schema(_remote_field(remote, "inputSchema"))
                registered.append(self._registered_tool(admission, schema))
        except Exception:  # noqa: BLE001 - remote schemas are an untrusted boundary
            self.manager.degrade(self.provider_id, "mcp_schema_rejected")
            self._discovered = ()
            return ()
        self._discovered = tuple(sorted(registered, key=lambda item: item.public_name))
        return self._discovered

    def _registered_tool(
        self, admission: McpToolAdmission, schema: dict[str, Any]
    ) -> RegisteredTool:
        spec = ToolSpec(
            name=admission.public_name,
            risk=admission.risk,
            trust=admission.trust,
            timeout_seconds=admission.timeout_seconds,
            approval_required=False,
            max_result_items=admission.max_result_items,
            description=admission.description,
        )
        return RegisteredTool(
            public_name=admission.public_name,
            provider_id=self.provider_id,
            provider_kind="mcp",
            remote_name=admission.remote_name,
            spec=spec,
            input_schema=schema,
        )

    async def execute(
        self,
        tool: RegisteredTool,
        arguments: dict[str, Any],
        *,
        run_id: str,
    ) -> ToolExecutionResult:
        del run_id
        if tool.provider_id != self.provider_id or tool.public_name not in self._admissions:
            raise PermissionError("MCP tool is not admitted by this provider")
        validated = validate_mcp_arguments(tool, arguments)
        admission = self._admissions[tool.public_name]
        raw = await self.manager.call_tool(
            self.provider_id,
            tool.remote_name,
            validated,
            timeout_seconds=admission.timeout_seconds,
        )
        return normalize_mcp_result(
            raw,
            tool_name=tool.public_name,
            trust=tool.spec.trust,
            max_result_items=admission.max_result_items,
            max_output_bytes=admission.max_output_bytes,
        )
