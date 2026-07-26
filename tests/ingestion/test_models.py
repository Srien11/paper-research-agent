from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.ingestion.models import (
    DocumentAsset,
    DocumentElement,
    IngestionManifest,
    PageRecord,
    SectionRecord,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


class IngestionContractTests(unittest.TestCase):
    def test_document_asset_round_trip_and_extra_field_rejection(self) -> None:
        asset = DocumentAsset(
            asset_id="asset-a",
            corpus_id="C001",
            corpus_version="corpus-v1",
            source_sha256=SHA_A,
            source_bytes=100,
            expected_page_count=2,
            storage_class="internal_research_only",
        )

        self.assertEqual(
            DocumentAsset.model_validate(asset.model_dump()),
            asset,
        )
        with self.assertRaises(ValidationError):
            DocumentAsset.model_validate({**asset.model_dump(), "local_path": "D:/secret.pdf"})

    def test_parsed_page_requires_text_and_hashes(self) -> None:
        with self.assertRaisesRegex(ValidationError, "parsed 页面必须包含"):
            self._page(status="parsed")

        page = self._page(
            status="parsed",
            raw_text="Raw",
            normalized_text="Raw",
            raw_text_sha256=SHA_A,
            normalized_text_sha256=SHA_A,
        )
        self.assertEqual(page.page_number, 1)

    def test_failed_page_requires_error_and_rejects_text(self) -> None:
        with self.assertRaisesRegex(ValidationError, "必须包含错误代码"):
            self._page(status="failed")
        with self.assertRaisesRegex(ValidationError, "不能保存正文"):
            self._page(
                status="failed",
                raw_text="leak",
                error_code="parse_error",
                error_message="failed",
            )

    def test_empty_page_rejects_errors(self) -> None:
        with self.assertRaisesRegex(ValidationError, "empty 页面不能包含错误"):
            self._page(
                status="empty",
                error_code="unexpected",
                error_message="unexpected",
            )

    def test_section_rejects_invalid_range_and_self_parent(self) -> None:
        values = {
            "section_id": "section-1",
            "asset_id": "asset-a",
            "corpus_id": "C001",
            "level": 1,
            "ordinal": 0,
            "title_raw": "Introduction",
            "title_normalized": "Introduction",
            "start_page": 2,
            "end_page": 1,
            "source_sha256": SHA_A,
            "parser_name": "test",
            "parser_version": "1",
        }
        with self.assertRaisesRegex(ValidationError, "结束页不能早于"):
            SectionRecord(**values)
        with self.assertRaisesRegex(ValidationError, "自身作为父章节"):
            SectionRecord(
                **{
                    **values,
                    "start_page": 1,
                    "parent_section_id": "section-1",
                }
            )

    def test_element_rejects_invalid_bbox_and_incomplete_span(self) -> None:
        with self.assertRaisesRegex(ValidationError, "坐标框终点"):
            self._element(bbox=(10.0, 0.0, 5.0, 10.0))
        with self.assertRaisesRegex(ValidationError, "同时提供起点"):
            self._element(raw_start=0)

    def test_generated_element_requires_complete_lineage(self) -> None:
        with self.assertRaisesRegex(ValidationError, "完整生成血缘"):
            self._element(content_origin="generated", generation_method="ocr")
        element = self._element(
            content_origin="generated",
            generation_method="ocr",
            generation_model="tesseract",
            generation_version="5",
        )
        self.assertEqual(element.content_origin, "generated")

    def test_manifest_page_counts_must_balance(self) -> None:
        values = {
            "build_id": "build-1",
            "corpus_version": "corpus-v1",
            "parser_name": "test",
            "parser_version": "1",
            "parser_config_sha256": SHA_A,
            "asset_count": 1,
            "expected_page_count": 2,
            "parsed_page_count": 1,
            "empty_page_count": 0,
            "failed_page_count": 0,
            "quarantined_page_count": 0,
            "section_count": 0,
            "element_count": 1,
            "artifact_sha256": {"pages.jsonl": SHA_B},
        }
        with self.assertRaisesRegex(ValidationError, "状态计数之和"):
            IngestionManifest(**values)
        manifest = IngestionManifest(**{**values, "empty_page_count": 1})
        self.assertEqual(manifest.expected_page_count, 2)

    @staticmethod
    def _page(**overrides: object) -> PageRecord:
        values: dict[str, object] = {
            "page_id": "page-1",
            "asset_id": "asset-a",
            "corpus_id": "C001",
            "page_number": 1,
            "status": "empty",
            "width_points": 612.0,
            "height_points": 792.0,
            "source_sha256": SHA_A,
            "parser_name": "test",
            "parser_version": "1",
        }
        return PageRecord(**{**values, **overrides})

    @staticmethod
    def _element(**overrides: object) -> DocumentElement:
        values: dict[str, object] = {
            "element_id": "element-1",
            "asset_id": "asset-a",
            "page_id": "page-1",
            "corpus_id": "C001",
            "page_number": 1,
            "element_type": "paragraph",
            "reading_order": 0,
            "raw_text": "Evidence",
            "normalized_text": "Evidence",
            "normalized_text_sha256": SHA_A,
            "source_sha256": SHA_B,
            "parser_name": "test",
            "parser_version": "1",
        }
        return DocumentElement(**{**values, **overrides})


if __name__ == "__main__":
    unittest.main()

