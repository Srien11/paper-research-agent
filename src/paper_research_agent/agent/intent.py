"""Conservative intent gates for selecting the multi-step research workflow."""

from __future__ import annotations

import re

_COMPARISON_SIGNAL = re.compile(
    r"(?:比较|对比|相比|异同|区别|差异|优劣|优缺点|"
    r"\bcompare\b|\bcomparison\b|\bversus\b|\bvs\.?\b|\bdifferences?\b)",
    re.IGNORECASE,
)
_RESEARCH_OBJECT_SIGNAL = re.compile(
    r"(?:论文|研究|方法|模型|算法|实验|数据集|指标|架构|"
    r"\bpapers?\b|\bstud(?:y|ies)\b|\bresearch\b|\bmethods?\b|"
    r"\bmodels?\b|\balgorithms?\b|\bexperiments?\b|\bdatasets?\b|"
    r"\bmetrics?\b|\barchitectures?\b)",
    re.IGNORECASE,
)
_MULTI_OBJECT_SIGNAL = re.compile(
    r"(?:两篇|多篇|两个|多个|三种|各自|分别|共同|相同|不同|之间|这些论文|上述论文|"
    r"双方|\bboth\b|\btwo\b|\bmultiple\b|\bseveral\b|\beach\b|"
    r"\brespectively\b|\bacross\b|\bbetween\b|\bshared\b|\bcommon\b|"
    r"\bthese papers\b|\bthe papers\b)",
    re.IGNORECASE,
)


def requires_research_planning(question: str) -> bool:
    """Return true only for explicit comparisons over scholarly objects.

    This gate is deliberately narrower than semantic routing. It exists to stop a
    clearly evidence-dependent comparison from silently becoming ordinary chat,
    while leaving casual comparisons and all non-comparison questions untouched.
    """
    normalized = " ".join(question.strip().split())
    if not normalized:
        return False
    return bool(
        _RESEARCH_OBJECT_SIGNAL.search(normalized)
        and (
            _COMPARISON_SIGNAL.search(normalized)
            or _MULTI_OBJECT_SIGNAL.search(normalized)
        )
    )
