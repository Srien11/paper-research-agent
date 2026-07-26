"""论文检索文本的确定性归一化。"""

from __future__ import annotations

import re
import unicodedata

_HYPHENATED_LINE_BREAK = re.compile(r"(?<=\w)-\s*\r?\n\s*(?=\w)")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """统一 Unicode、断词换行和空白，不改变原始证据文本。"""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u00a0", " ")
    normalized = _HYPHENATED_LINE_BREAK.sub("", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()

