from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHELL_LAUNCHERS = {
    "bash",
    "bash.exe",
    "cmd",
    "cmd.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
    "sh",
    "sh.exe",
}


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class McpToolAdmission(FrozenModel):
    remote_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    public_name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,127}$")
    description: str = Field(min_length=1, max_length=500)
    risk: Literal["local_read", "network_read"]
    trust: Literal["research_context", "computed_result"] = "research_context"
    timeout_seconds: float = Field(gt=0, le=30)
    max_result_items: int = Field(ge=1, le=50)
    max_output_bytes: int = Field(default=131_072, ge=1_024, le=262_144)


class McpStdioServerConfig(FrozenModel):
    server_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    enabled: bool = False
    transport: Literal["stdio"] = "stdio"
    command: str = Field(min_length=1, max_length=500)
    args: tuple[str, ...] = Field(default=(), max_length=32)
    inherit_env: tuple[str, ...] = Field(default=(), max_length=16)
    startup_timeout_seconds: float = Field(default=10, gt=0, le=30)
    tools: tuple[McpToolAdmission, ...] = Field(min_length=1, max_length=50)

    @field_validator("command")
    @classmethod
    def _validate_command(cls, value: str) -> str:
        launcher_names = {
            PurePosixPath(value).name.lower(),
            PureWindowsPath(value).name.lower(),
        }
        if launcher_names & _SHELL_LAUNCHERS:
            raise ValueError("shell launchers are forbidden")
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("MCP command must be an absolute path")
        return str(path)

    @field_validator("args")
    @classmethod
    def _validate_args(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(len(value) > 500 or "\n" in value or "\r" in value or "\0" in value for value in values):
            raise ValueError("MCP arguments must be bounded single-line strings")
        return values

    @field_validator("inherit_env")
    @classmethod
    def _validate_inherited_environment(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("inherited environment names must be unique")
        for name in values:
            if _ENV_NAME.fullmatch(name) is None:
                raise ValueError("inherit_env accepts environment variable names only")
            if name not in os.environ:
                raise ValueError("inherited environment variable is not available")
        return values

    @model_validator(mode="after")
    def _validate_tool_namespace(self) -> Self:
        public_names: set[str] = set()
        remote_names: set[str] = set()
        for tool in self.tools:
            normalized = re.sub(r"[.-]", "_", tool.remote_name.lower())
            expected = f"{self.server_id}__{normalized}"
            if tool.public_name != expected:
                raise ValueError("MCP public name must use the exact server namespace")
            if tool.public_name in public_names or tool.remote_name in remote_names:
                raise ValueError("MCP tool names must be unique within a server")
            public_names.add(tool.public_name)
            remote_names.add(tool.remote_name)
        return self


class McpHostConfig(FrozenModel):
    schema_version: Literal["mcp-host-v1"]
    servers: tuple[McpStdioServerConfig, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def _validate_unique_names(self) -> Self:
        server_ids: set[str] = set()
        public_names: set[str] = set()
        for server in self.servers:
            if server.server_id in server_ids:
                raise ValueError("MCP server IDs must be unique")
            server_ids.add(server.server_id)
            for tool in server.tools:
                if tool.public_name in public_names:
                    raise ValueError("MCP public tool names must be globally unique")
                public_names.add(tool.public_name)
        return self


def load_mcp_host_config(path: Path) -> McpHostConfig:
    try:
        source = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError("MCP config file not found") from exc
    try:
        raw = json.loads(source)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("invalid MCP config JSON") from exc
    try:
        return McpHostConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError("invalid MCP config schema") from exc
