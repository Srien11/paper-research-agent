from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.chunking.runner import run_chunking


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic evidence chunks.")
    parser.add_argument("--elements", type=Path, required=True)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/chunking/baseline-v1.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for path in run_chunking(args.elements, args.sections, args.config, args.output):
        print(path)


if __name__ == "__main__":
    main()
