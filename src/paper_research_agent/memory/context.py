"""Project bounded session memory into retrieval and model context."""

from __future__ import annotations

import re
from collections.abc import Sequence

from paper_research_agent.context.models import ContextMemoryTurn
from paper_research_agent.memory.models import ShortTermMemoryTurn

_CONTEXT_DEPENDENT = re.compile(
    r"(它|其|这个|这些|这种|该(?:方法|模型|论文|图|结果|指标)|上述|前者|后者|"
    r"上一(?:个|种|篇|张)|刚才|继续|相比呢|区别呢|怎么样呢)"
)


def contextualize_retrieval_query(
    question: str,
    turns: Sequence[ShortTermMemoryTurn],
    *,
    max_question_chars: int = 96,
) -> str:
    """Add the previous user topic only for clearly context-dependent follow-ups."""
    normalized = question.strip()
    if not normalized:
        raise ValueError("question cannot be blank")
    if (
        not turns
        or len(normalized) > max_question_chars
        or _CONTEXT_DEPENDENT.search(normalized) is None
    ):
        return normalized
    previous = turns[-1].standalone_question
    return f"当前问题：{normalized}\n上一轮研究问题：{previous}"


def to_context_memory(
    turns: Sequence[ShortTermMemoryTurn],
) -> tuple[ContextMemoryTurn, ...]:
    """Remove storage metadata and expose only low-trust conversational continuity."""
    return tuple(
        ContextMemoryTurn(
            turn_id=turn.turn_id,
            user_question=turn.user_question,
            status=turn.status,
            assistant_claims=turn.assistant_claims,
        )
        for turn in turns
    )
