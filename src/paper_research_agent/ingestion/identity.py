"""为解析产物生成与运行环境无关的确定性标识。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def sha256_text(text: str) -> str:
    """计算 UTF-8 文本的 SHA-256。"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: object) -> str:
    """使用带分隔符的规范输入生成稳定标识。"""

    if not prefix or not parts:
        raise ValueError("稳定标识必须包含前缀和输入")
    payload = "\0".join(str(part) for part in parts)
    return f"{prefix}_{sha256_text(payload)[:32]}"


def make_asset_id(source_sha256: str) -> str:
    return stable_id("asset", source_sha256)


def make_page_id(source_sha256: str, page_number: int) -> str:
    if page_number < 1:
        raise ValueError("页码必须从 1 开始")
    return stable_id("page", source_sha256, page_number)


def make_section_id(
    source_sha256: str,
    ordinal: int,
    normalized_title_sha256: str,
) -> str:
    if ordinal < 0:
        raise ValueError("章节序号不能为负数")
    return stable_id("section", source_sha256, ordinal, normalized_title_sha256)


def make_element_id(
    source_sha256: str,
    page_number: int,
    element_type: str,
    reading_order: int,
    normalized_text_sha256: str,
) -> str:
    if page_number < 1:
        raise ValueError("页码必须从 1 开始")
    if reading_order < 0:
        raise ValueError("阅读序号不能为负数")
    return stable_id(
        "element",
        source_sha256,
        page_number,
        element_type,
        reading_order,
        normalized_text_sha256,
    )


def make_build_id(
    corpus_version: str,
    parser_name: str,
    parser_version: str,
    parser_config_sha256: str,
    source_hashes: Iterable[str],
) -> str:
    sorted_hashes = tuple(sorted(source_hashes))
    if not sorted_hashes:
        raise ValueError("构建标识至少需要一个源文件哈希")
    return stable_id(
        "build",
        corpus_version,
        parser_name,
        parser_version,
        parser_config_sha256,
        *sorted_hashes,
    )

