from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.ingestion.ocr import TesseractOcrBackend


class FakeImage:
    width = 1200
    height = 1600

    def save(self, fp, format: str) -> None:
        self.format = format
        fp.write(b"fake-png")


class TesseractOcrBackendTests(unittest.TestCase):
    def test_tsv_words_are_grouped_and_scaled_to_pdf_points(self) -> None:
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t100\t200\t100\t40\t95\tScanned\n"
            "5\t1\t1\t1\t1\t2\t220\t200\t120\t40\t94\tevidence\n"
        )
        completed = subprocess.CompletedProcess(
            args=["tesseract"],
            returncode=0,
            stdout=tsv.encode(),
            stderr=b"",
        )
        backend = TesseractOcrBackend()

        with (
            patch.object(TesseractOcrBackend, "_render_page", return_value=FakeImage()),
            patch.object(
                TesseractOcrBackend,
                "_version",
                return_value="tesseract 5.5.0",
            ),
            patch("paper_research_agent.ingestion.ocr.subprocess.run", return_value=completed),
        ):
            result = backend.extract_page(
                Path("paper.pdf"),
                1,
                width_points=600,
                height_points=800,
            )

        self.assertEqual(result.lines[0].text, "Scanned evidence")
        self.assertEqual(result.lines[0].x0, 50)
        self.assertEqual(result.lines[0].top, 100)
        self.assertEqual(result.lines[0].x1, 170)
        self.assertEqual(result.lines[0].bottom, 120)
        self.assertEqual(result.engine_version, "tesseract 5.5.0")
        self.assertEqual(result.model, "tesseract:eng+chi_sim")


if __name__ == "__main__":
    unittest.main()
