from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_local_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    _load_local_env(PROJECT_ROOT / ".env")
    # The local CLI enters the unified main graph by default and retains an
    # explicit one-variable fallback with PRA_MAIN_AGENT_MODE=legacy.
    os.environ.setdefault("PRA_MAIN_AGENT_MODE", "primary")
    parser = argparse.ArgumentParser(
        description="Serve the owner-only paper research Web interface."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        parser.error("the private Web service may only listen on a loopback address")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    try:
        import uvicorn
    except ImportError:
        parser.exit(1, 'web dependencies are missing; install ".[retrieval,web]"\n')

    from paper_research_agent.web.bootstrap import main_agent_mode_from_environment

    mode = main_agent_mode_from_environment()
    print(f"主 Agent Web 启动模式：{mode}")
    # One process owns the local indexes and model sessions. In-process retrieval
    # concurrency is configured separately, avoiding one full model copy per worker.
    uvicorn.run(
        "paper_research_agent.web.app:create_app",
        factory=True,
        env_file=None,
        host=args.host,
        port=args.port,
        workers=1,
        access_log=False,
        server_header=False,
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1,::1",
    )


if __name__ == "__main__":
    main()
