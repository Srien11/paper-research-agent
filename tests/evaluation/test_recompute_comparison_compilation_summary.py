from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts import recompute_comparison_compilation_summary as recompute_module
from scripts.recompute_comparison_compilation_summary import (
    main,
    recompute_file,
    recompute_payload,
)

PRIVATE_SENTINEL = "PRIVATE-QUESTION-TEXT"


def _case() -> dict[str, object]:
    return {
        "schema_version": "comparison-e2e-diagnostic-v1",
        "question_id": "CPG001",
        "split": "dev",
        "original_question_sha256": "a" * 64,
        "raw_question_preserved": True,
        "rewrite_status": PRIVATE_SENTINEL,
        "candidate_paper_ids_top8": [],
        "final_paper_ids": [],
        "planned_dimensions": [],
        "retrievals": [],
        "compilation_audit": {
            "attempts": [
                {
                    "attempt": 1,
                    "outcome": "contract_invalid",
                    "failure_code": "fact_chunk_scope_invalid",
                    "accepted_fact_count": 1,
                    "rejected_fact_count": 1,
                    "unresolved_fact_requirement_count": 1,
                    "requested_requirement_ids": ["a", "b"],
                    "accepted_requirement_ids": ["a"],
                    "failed_requirement_ids": ["b"],
                },
                {
                    "attempt": 2,
                    "outcome": "validated",
                    "accepted_fact_count": 1,
                    "requested_requirement_ids": ["b"],
                    "accepted_requirement_ids": ["b"],
                    "failed_requirement_ids": [],
                },
            ],
            "repair": {
                "applied": False,
                "source_assessment_available": True,
                "input_fact_count": 2,
                "retained_fact_count": 2,
                "dropped_chunk_scope_count": 0,
                "dropped_fact_mapping_count": 0,
                "missing_ledger_cell_count": 0,
                "fallback_empty_used": False,
            },
        },
        "citations": [],
        "total_latency_ms": 0,
    }


def _payload() -> dict[str, object]:
    return {
        "schema_version": "comparison-e2e-run-v1",
        "question_count": 1,
        "summary": {"private": PRIVATE_SENTINEL},
        "answer_scores": {"score": 1},
        "fact_lineage": {"fact_count": 1},
        "compilation": {"failed_unit_count": 1},
        "split_summaries": {
            "dev": {
                "pipeline": {"run_success_rate": 1},
                "compilation": {"failed_unit_count": 1},
            }
        },
        "cases": [_case()],
    }


def _write_payload(path: Path, payload: object) -> bytes:
    raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return raw


def test_recompute_payload_upgrades_only_compilation_summaries() -> None:
    payload = _payload()
    original = copy.deepcopy(payload)

    updated = recompute_payload(payload, source_run_sha256="f" * 64)

    assert payload == original
    assert updated["schema_version"] == "comparison-e2e-run-v2"
    assert updated["source_run_sha256"] == "f" * 64
    assert updated["compilation"]["attempts"]["failed_unit_count"] == 1
    assert updated["compilation"]["final"]["failed_unit_count"] == 0
    assert updated["split_summaries"]["dev"]["compilation"]["final"][
        "failed_unit_count"
    ] == 0
    for field in ("summary", "answer_scores", "fact_lineage", "cases"):
        assert updated[field] == original[field]


def test_offline_tool_has_no_runtime_or_network_dependencies() -> None:
    module_names = set(recompute_module.__dict__)

    assert "_load_local_env" not in module_names
    assert "RAGRuntime" not in module_names
    assert "httpx" not in module_names


def test_recompute_file_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        recompute_file(tmp_path / "missing.json", tmp_path / "output.json")


def test_recompute_file_rejects_same_input_and_output(tmp_path: Path) -> None:
    input_path = tmp_path / "run.json"
    _write_payload(input_path, _payload())

    with pytest.raises(ValueError, match="different paths"):
        recompute_file(input_path, input_path)


def test_recompute_file_refuses_to_overwrite_output(tmp_path: Path) -> None:
    input_path = tmp_path / "run.json"
    output_path = tmp_path / "corrected.json"
    _write_payload(input_path, _payload())
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        recompute_file(input_path, output_path)

    assert output_path.read_text(encoding="utf-8") == "existing"


def test_recompute_file_rejects_invalid_json(tmp_path: Path) -> None:
    input_path = tmp_path / "run.json"
    input_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        recompute_file(input_path, tmp_path / "corrected.json")


def test_recompute_file_rejects_invalid_case_schema(tmp_path: Path) -> None:
    input_path = tmp_path / "run.json"
    _write_payload(input_path, {"cases": [{"question_id": PRIVATE_SENTINEL}]})

    with pytest.raises(ValidationError):
        recompute_file(input_path, tmp_path / "corrected.json")


def test_cli_saves_new_file_without_changing_input_or_leaking_case_data(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "run.json"
    output_path = tmp_path / "corrected.json"
    raw_input = _write_payload(input_path, _payload())
    input_sha256 = hashlib.sha256(raw_input).hexdigest()

    main(["--input", str(input_path), "--output", str(output_path)])

    stdout = capsys.readouterr().out
    corrected = json.loads(output_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == input_sha256
    assert corrected["source_run_sha256"] == input_sha256
    assert corrected["compilation"]["final"]["failed_unit_count"] == 0
    assert "schema_version=comparison-e2e-run-v2" in stdout
    assert "question_count=1" in stdout
    assert "attempt_count=2" in stdout
    assert "final_failed_unit_count=0" in stdout
    assert PRIVATE_SENTINEL not in stdout
