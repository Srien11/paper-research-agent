"""Sandbox-free deterministic calculation and experiment analysis."""

from __future__ import annotations

import ast
import math
import operator
import statistics
from collections.abc import Callable, Mapping, Sequence

from paper_research_agent.agent.tooling.contracts import (
    AnalyzeExperimentDataInput,
    CalculateInput,
    CorpusInput,
    ToolExecutionResult,
)
from paper_research_agent.chunking.models import EvidenceChunk

_BINARY: Mapping[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: Mapping[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class AnalysisResearchTools:
    def __init__(self, *, chunks: Sequence[EvidenceChunk]):
        self._chunks = tuple(chunks)

    def calculate(self, request: CalculateInput) -> ToolExecutionResult:
        tree = ast.parse(request.expression, mode="eval")
        value = _evaluate(tree.body, depth=0)
        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("calculation result is outside the safe range")
        return ToolExecutionResult(
            tool_name="calculate",
            items=({"value": value},),
            summary={"node_count": sum(1 for _ in ast.walk(tree))},
        )

    def analyze_experiment_data(self, request: AnalyzeExperimentDataInput) -> ToolExecutionResult:
        items = []
        for index, column in enumerate(request.columns):
            values = [row[index] for row in request.rows]
            result: dict[str, object] = {"column": column}
            for operation in request.operations:
                if operation == "count":
                    result[operation] = len(values)
                elif operation == "mean":
                    result[operation] = statistics.fmean(values)
                elif operation == "median":
                    result[operation] = statistics.median(values)
                elif operation == "stdev":
                    result[operation] = statistics.stdev(values) if len(values) > 1 else None
                elif operation == "min":
                    result[operation] = min(values)
                elif operation == "max":
                    result[operation] = max(values)
            items.append(result)
        return ToolExecutionResult(
            tool_name="analyze_experiment_data",
            items=tuple(items),
            summary={"row_count": len(request.rows), "column_count": len(request.columns)},
        )

    def check_reproducibility(self, request: CorpusInput) -> ToolExecutionResult:
        text = "\n".join(
            chunk.text for chunk in self._chunks if chunk.corpus_id == request.corpus_id
        ).casefold()
        signals = {
            "code": any(
                term in text for term in ("github.com", "source code", "code is available")
            ),
            "data": any(term in text for term in ("dataset is available", "data are available")),
            "model": any(term in text for term in ("model weights", "pretrained model")),
            "random_seed": any(term in text for term in ("random seed", "seeds")),
            "hyperparameters": "hyperparameter" in text,
        }
        return ToolExecutionResult(
            tool_name="check_reproducibility",
            status="ok" if text else "not_found",
            items=({"corpus_id": request.corpus_id, **signals},) if text else (),
            summary={"present_count": sum(signals.values()) if text else 0},
        )


def _evaluate(node: ast.AST, *, depth: int) -> float:
    if depth > 20:
        raise ValueError("calculation expression is too deep")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left = _evaluate(node.left, depth=depth + 1)
        right = _evaluate(node.right, depth=depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("calculation exponent is outside the safe range")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand, depth=depth + 1))
    raise ValueError("calculation contains a forbidden expression")
