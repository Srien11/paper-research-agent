from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from paper_research_agent.agent.mcp.config import McpHostConfig, load_mcp_host_config


def _valid_payload() -> dict[str, object]:
    return {
        "schema_version": "mcp-host-v1",
        "servers": [
            {
                "server_id": "zotero",
                "enabled": True,
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "paper_research_agent.mcp_servers.zotero"],
                "inherit_env": [],
                "startup_timeout_seconds": 10,
                "tools": [
                    {
                        "remote_name": "search_items",
                        "public_name": "zotero__search_items",
                        "description": "在本机 Zotero 文献库中搜索条目。",
                        "risk": "local_read",
                        "trust": "research_context",
                        "timeout_seconds": 5,
                        "max_result_items": 20,
                        "max_output_bytes": 131_072,
                    }
                ],
            }
        ],
    }


class McpConfigTests(unittest.TestCase):
    def test_accepts_valid_frozen_stdio_config(self) -> None:
        config = McpHostConfig.model_validate(_valid_payload())
        self.assertEqual(config.servers[0].server_id, "zotero")
        with self.assertRaises(ValidationError):
            config.servers[0].enabled = False

    def test_rejects_extra_fields(self) -> None:
        payload = _valid_payload()
        payload["secret"] = "must-not-be-accepted"
        with self.assertRaises(ValidationError):
            McpHostConfig.model_validate(payload)

    def test_rejects_relative_command(self) -> None:
        payload = _valid_payload()
        payload["servers"][0]["command"] = "python"
        with self.assertRaisesRegex(ValidationError, "absolute"):
            McpHostConfig.model_validate(payload)

    def test_rejects_shell_launcher(self) -> None:
        payload = _valid_payload()
        payload["servers"][0]["command"] = "C:\\Windows\\System32\\cmd.exe"
        with self.assertRaisesRegex(ValidationError, "shell launchers are forbidden"):
            McpHostConfig.model_validate(payload)

    def test_rejects_duplicate_server_and_public_names(self) -> None:
        payload = _valid_payload()
        duplicate = dict(payload["servers"][0])
        duplicate["tools"] = [dict(payload["servers"][0]["tools"][0])]
        payload["servers"].append(duplicate)
        with self.assertRaises(ValidationError):
            McpHostConfig.model_validate(payload)

    def test_rejects_unqualified_public_name(self) -> None:
        payload = _valid_payload()
        payload["servers"][0]["tools"][0]["public_name"] = "search_items"
        with self.assertRaisesRegex(ValidationError, "server namespace"):
            McpHostConfig.model_validate(payload)

    def test_rejects_citation_trust_and_write_risk(self) -> None:
        for field, value in (("trust", "citation_evidence"), ("risk", "network_write")):
            payload = _valid_payload()
            payload["servers"][0]["tools"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                McpHostConfig.model_validate(payload)

    def test_rejects_environment_values_and_invalid_arguments(self) -> None:
        payload = _valid_payload()
        payload["servers"][0]["inherit_env"] = ["TOKEN=secret"]
        with self.assertRaises(ValidationError):
            McpHostConfig.model_validate(payload)
        payload = _valid_payload()
        payload["servers"][0]["args"] = ["line1\nline2"]
        with self.assertRaises(ValidationError):
            McpHostConfig.model_validate(payload)

    def test_loader_distinguishes_missing_json_and_schema_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "MCP config file not found"):
                load_mcp_host_config(root / "missing.json")
            invalid_json = root / "invalid.json"
            invalid_json.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid MCP config JSON"):
                load_mcp_host_config(invalid_json)
            invalid_schema = root / "invalid-schema.json"
            invalid_schema.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid MCP config schema"):
                load_mcp_host_config(invalid_schema)


if __name__ == "__main__":
    unittest.main()
