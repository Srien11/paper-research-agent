"""根据 PDF 图注定位并裁剪论文图片。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.figures.runner import run_figure_cropping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "figure-semantics-v1",
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest_path, crops = run_figure_cropping(
        args.elements,
        args.corpus_dir,
        args.output,
        dpi=args.dpi,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "figure_count": len(crops),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
