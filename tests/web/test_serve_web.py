from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
