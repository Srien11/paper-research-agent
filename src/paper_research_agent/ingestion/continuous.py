"""不可变增量摄取的当前构建指针与轮询更新入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from paper_research_agent.ingestion.runner import (
    IngestionRunResult,
    run_corpus_ingestion,
    run_incremental_corpus_ingestion,
)


class CurrentIngestionBuild(BaseModel):
    """只保存相对路径，避免将本机绝对路径写入可同步产物。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["current-ingestion-build-v1"] = "current-ingestion-build-v1"
    build_id: str = Field(min_length=1)
    corpus_version: str = Field(min_length=1)
    build_path: str = Field(min_length=1)


class CurrentKnowledgeBase(BaseModel):
    """仅在 chunks 与向量索引都构建成功后才更新的 RAG 当前版本。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["current-knowledge-base-v1"] = "current-knowledge-base-v1"
    build_id: str = Field(min_length=1)
    corpus_version: str = Field(min_length=1)
    build_path: str = Field(min_length=1)


@dataclass(frozen=True)
class ContinuousUpdateResult:
    result: IngestionRunResult
    changed: bool
    pointer_path: Path


def update_current_ingestion_build(
    corpus_dir: Path,
    output_root: Path,
) -> ContinuousUpdateResult:
    """处理冻结清单中新增加的 PDF，并在成功后原子切换当前构建指针。"""

    pointer_path = output_root / "current_ingestion_build.json"
    current = _load_current(pointer_path, output_root)
    if current is None:
        result = run_corpus_ingestion(corpus_dir, output_root)
        _write_current(pointer_path, output_root, result)
        return ContinuousUpdateResult(result=result, changed=True, pointer_path=pointer_path)

    parent_dir = output_root / Path(current.build_path)
    result = run_incremental_corpus_ingestion(
        corpus_dir,
        output_root,
        parent_output_dir=parent_dir,
    )
    changed = result.manifest.build_id != current.build_id
    if changed:
        _write_current(pointer_path, output_root, result)
    return ContinuousUpdateResult(result=result, changed=changed, pointer_path=pointer_path)


def _load_current(pointer_path: Path, output_root: Path) -> CurrentIngestionBuild | None:
    if not pointer_path.is_file():
        return None
    current = CurrentIngestionBuild.model_validate_json(pointer_path.read_text(encoding="utf-8"))
    candidate = (output_root / current.build_path).resolve()
    if output_root.resolve() not in candidate.parents:
        raise ValueError("当前构建指针越出解析产物根目录")
    if not (candidate / "ingestion_manifest.json").is_file():
        raise ValueError("当前构建指针指向不存在或不完整的构建")
    return current


def _write_current(
    pointer_path: Path,
    output_root: Path,
    result: IngestionRunResult,
) -> None:
    relative = result.output_dir.resolve().relative_to(output_root.resolve())
    payload = CurrentIngestionBuild(
        build_id=result.manifest.build_id,
        corpus_version=result.manifest.corpus_version,
        build_path=relative.as_posix(),
    ).model_dump_json(indent=2)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer_path.with_suffix(".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(pointer_path)


def activate_current_knowledge_base(
    output_root: Path,
    result: IngestionRunResult,
) -> Path:
    """原子发布完整 RAG 构建；运行中的服务在下次启动时读取该指针。"""

    pointer_path = output_root / "current_knowledge_base.json"
    relative = result.output_dir.resolve().relative_to(output_root.resolve())
    payload = CurrentKnowledgeBase(
        build_id=result.manifest.build_id,
        corpus_version=result.manifest.corpus_version,
        build_path=relative.as_posix(),
    ).model_dump_json(indent=2)
    temporary = pointer_path.with_suffix(".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(pointer_path)
    return pointer_path
