from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.ingestion.continuous import (
    activate_current_knowledge_base,
    update_current_ingestion_build,
)
from paper_research_agent.ingestion.runner import IngestionRunError
from tests.ingestion.helpers import fake_parse, write_manifests


class ContinuousIngestionTests(unittest.TestCase):
    def test_new_source_creates_child_build_without_changing_old_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_dir = root / "corpus"
            output_root = root / "processed"
            corpus_dir.mkdir()
            first_pdf = corpus_dir / "first.pdf"
            first_pdf.write_bytes(b"%PDF first")
            write_manifests(corpus_dir, first_pdf)

            with patch(
                "paper_research_agent.ingestion.runner.parse_pdf_asset",
                side_effect=fake_parse,
            ) as parse:
                first = update_current_ingestion_build(corpus_dir, output_root)
                self.assertTrue(first.changed)
                self.assertEqual(parse.call_count, 1)

                second_pdf = corpus_dir / "second.pdf"
                second_pdf.write_bytes(b"%PDF second")
                self._append_paper(corpus_dir, second_pdf, "C002", "test:second")
                second = update_current_ingestion_build(corpus_dir, output_root)

            self.assertTrue(second.changed)
            self.assertEqual(parse.call_count, 2)
            self.assertEqual(second.result.manifest.parent_build_id, first.result.manifest.build_id)
            self.assertEqual(second.result.manifest.added_asset_count, 1)
            first_ids = self._ids(first.result.output_dir / "elements.jsonl")
            second_ids = self._ids(second.result.output_dir / "elements.jsonl")
            self.assertTrue(first_ids <= second_ids)
            self.assertEqual(
                (first.result.output_dir / "elements.jsonl").read_text(encoding="utf-8"),
                "".join(
                    line + "\n"
                    for line in (second.result.output_dir / "elements.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if json.loads(line)["element_id"] in first_ids
                ),
            )

            with patch(
                "paper_research_agent.ingestion.runner.parse_pdf_asset",
                side_effect=AssertionError("no source should be reparsed"),
            ):
                repeated = update_current_ingestion_build(corpus_dir, output_root)
            self.assertFalse(repeated.changed)
            self.assertEqual(repeated.result.output_dir, second.result.output_dir)

            pointer = activate_current_knowledge_base(output_root, second.result)
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            self.assertEqual(payload["build_id"], second.result.manifest.build_id)
            self.assertEqual(payload["build_path"], "corpus-v1/" + second.result.manifest.build_id)

    def test_removing_a_parent_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_dir = root / "corpus"
            corpus_dir.mkdir()
            pdf = corpus_dir / "first.pdf"
            pdf.write_bytes(b"%PDF first")
            write_manifests(corpus_dir, pdf)
            with patch(
                "paper_research_agent.ingestion.runner.parse_pdf_asset",
                side_effect=fake_parse,
            ):
                update_current_ingestion_build(corpus_dir, root / "processed")
            (corpus_dir / "core_frozen.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(IngestionRunError, "缺少既有来源"):
                update_current_ingestion_build(corpus_dir, root / "processed")

    @staticmethod
    def _append_paper(corpus_dir: Path, pdf: Path, corpus_id: str, key: str) -> None:
        source = json.loads((corpus_dir / "core_frozen.jsonl").read_text(encoding="utf-8"))
        source.update(
            {
                "corpus_id": corpus_id,
                "canonical_key": key,
                "title": "Second paper",
                "local_pdf_path": str(pdf),
                "download_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                "download_bytes": pdf.stat().st_size,
            }
        )
        with (corpus_dir / "core_frozen.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(source) + "\n")

    @staticmethod
    def _ids(path: Path) -> set[str]:
        return {
            json.loads(line)["element_id"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }


if __name__ == "__main__":
    unittest.main()
