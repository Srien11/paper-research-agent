from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import serve_web

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ServeWebScriptTests(unittest.TestCase):
    def test_rejects_non_loopback_bind(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "serve_web.py"),
                "--host",
                "0.0.0.0",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("loopback", result.stderr)

    def test_passes_resolved_mode_to_application_factory(self) -> None:
        with (
            patch.object(sys, "argv", ["serve_web.py", "--port", "8093"]),
            patch.dict(
                "os.environ",
                {"PRA_MAIN_AGENT_MODE": "primary"},
                clear=False,
            ),
            patch("uvicorn.run") as run,
        ):
            serve_web.main()

        self.assertEqual(run.call_args.kwargs["factory"], True)
        self.assertEqual(run.call_args.kwargs["env_file"], None)

    def test_defaults_production_cli_to_primary_and_keeps_explicit_rollback(self) -> None:
        original_mode = os.environ.pop("PRA_MAIN_AGENT_MODE", None)
        try:
            with (
                patch.object(sys, "argv", ["serve_web.py"]),
                patch.object(serve_web, "_load_local_env"),
                patch("uvicorn.run"),
                patch("builtins.print") as printed,
            ):
                serve_web.main()
            self.assertEqual(os.environ["PRA_MAIN_AGENT_MODE"], "primary")
            printed.assert_any_call("主 Agent Web 启动模式：primary")

            with (
                patch.object(sys, "argv", ["serve_web.py"]),
                patch.object(serve_web, "_load_local_env"),
                patch.dict(os.environ, {"PRA_MAIN_AGENT_MODE": "legacy"}),
                patch("uvicorn.run"),
                patch("builtins.print") as rollback_printed,
            ):
                serve_web.main()
            rollback_printed.assert_any_call("主 Agent Web 启动模式：legacy")
        finally:
            if original_mode is None:
                os.environ.pop("PRA_MAIN_AGENT_MODE", None)
            else:
                os.environ["PRA_MAIN_AGENT_MODE"] = original_mode


if __name__ == "__main__":
    unittest.main()
