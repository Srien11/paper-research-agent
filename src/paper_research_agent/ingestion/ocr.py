"""为无文本层的 PDF 页面提供可审计的 Tesseract OCR 降级。"""

from __future__ import annotations

import csv
import io
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, cast


class OcrError(RuntimeError):
    """OCR 无法完成。"""


class OcrUnavailableError(OcrError):
    """OCR 运行时或 PDF 渲染依赖不可用。"""


@dataclass(frozen=True)
class OcrLine:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float


@dataclass(frozen=True)
class OcrPageResult:
    lines: tuple[OcrLine, ...]
    engine: str
    engine_version: str
    model: str


class OcrBackend(Protocol):
    def config(self) -> dict[str, object]: ...

    def extract_page(
        self,
        pdf_path: Path,
        page_number: int,
        *,
        width_points: float,
        height_points: float,
    ) -> OcrPageResult: ...


class RenderedImage(Protocol):
    width: int
    height: int

    def save(self, fp: BinaryIO, format: str) -> None: ...


@dataclass(frozen=True)
class TesseractOcrBackend:
    """将单页渲染为 PNG，并使用 Tesseract TSV 输出恢复行和坐标。"""

    command: str = "tesseract"
    languages: str = "eng+chi_sim"
    dpi: int = 300
    timeout_seconds: float = 120.0

    def config(self) -> dict[str, object]:
        return {
            "engine": "tesseract",
            "engine_version": self._version(),
            "languages": self.languages,
            "dpi": self.dpi,
            "timeout_seconds": self.timeout_seconds,
        }

    def extract_page(
        self,
        pdf_path: Path,
        page_number: int,
        *,
        width_points: float,
        height_points: float,
    ) -> OcrPageResult:
        image = self._render_page(pdf_path, page_number)
        payload = io.BytesIO()
        image.save(payload, format="PNG")
        try:
            completed = subprocess.run(
                [
                    self.command,
                    "stdin",
                    "stdout",
                    "-l",
                    self.languages,
                    "--dpi",
                    str(self.dpi),
                    "tsv",
                ],
                input=payload.getvalue(),
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise OcrUnavailableError(f"找不到 Tesseract: {self.command}") from exc
        except subprocess.TimeoutExpired as exc:
            raise OcrError(f"Tesseract OCR 超时（{self.timeout_seconds:g} 秒）") from exc
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise OcrError(f"Tesseract OCR 失败（退出码 {completed.returncode}）: {detail}")
        lines = _parse_tsv_lines(
            completed.stdout.decode("utf-8", errors="replace"),
            image_width=image.width,
            image_height=image.height,
            width_points=width_points,
            height_points=height_points,
        )
        return OcrPageResult(
            lines=lines,
            engine="tesseract",
            engine_version=self._version(),
            model=f"tesseract:{self.languages}",
        )

    def _render_page(self, pdf_path: Path, page_number: int) -> RenderedImage:
        try:
            import pypdfium2 as pdfium  # type: ignore[import-untyped]
        except ImportError as exc:
            raise OcrUnavailableError(
                '缺少 pypdfium2；请安装项目的 "ingestion" 可选依赖'
            ) from exc
        try:
            document = pdfium.PdfDocument(str(pdf_path))
            page = document[page_number - 1]
            return cast(RenderedImage, page.render(scale=self.dpi / 72).to_pil())
        except Exception as exc:
            raise OcrError(f"PDF 第 {page_number} 页渲染失败: {exc}") from exc

    def _version(self) -> str:
        try:
            completed = subprocess.run(
                [self.command, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return "unavailable"
        first_line = completed.stdout.decode("utf-8", errors="replace").splitlines()
        if completed.returncode != 0 or not first_line:
            return "unavailable"
        return first_line[0].strip()


def create_default_ocr_backend() -> OcrBackend | None:
    """按环境变量创建默认 OCR；默认开启，可显式关闭。"""

    if os.getenv("PRA_OCR_ENABLED", "true").strip().casefold() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return None
    return TesseractOcrBackend(
        command=os.getenv("PRA_TESSERACT_CMD", "tesseract").strip() or "tesseract",
        languages=os.getenv("PRA_OCR_LANGUAGES", "eng+chi_sim").strip() or "eng+chi_sim",
        dpi=_positive_int_env("PRA_OCR_DPI", 300),
        timeout_seconds=_positive_float_env("PRA_OCR_TIMEOUT_SECONDS", 120.0),
    )


def _parse_tsv_lines(
    payload: str,
    *,
    image_width: int,
    image_height: int,
    width_points: float,
    height_points: float,
) -> tuple[OcrLine, ...]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in csv.DictReader(io.StringIO(payload), delimiter="\t"):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        grouped.setdefault(
            (row.get("block_num", ""), row.get("par_num", ""), row.get("line_num", "")),
            [],
        ).append(row)
    x_scale = width_points / image_width
    y_scale = height_points / image_height
    lines: list[OcrLine] = []
    for words in grouped.values():
        left = min(int(word["left"]) for word in words)
        top = min(int(word["top"]) for word in words)
        right = max(int(word["left"]) + int(word["width"]) for word in words)
        bottom = max(int(word["top"]) + int(word["height"]) for word in words)
        lines.append(
            OcrLine(
                text=" ".join(word["text"].strip() for word in words),
                x0=left * x_scale,
                top=top * y_scale,
                x1=right * x_scale,
                bottom=bottom * y_scale,
            )
        )
    return tuple(sorted(lines, key=lambda line: (line.top, line.x0, line.bottom)))


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return value


def _positive_float_env(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return value
