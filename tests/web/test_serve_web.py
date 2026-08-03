from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
