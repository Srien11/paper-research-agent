"""Model-independent conservative context budget estimation."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from paper_research_agent.chunking.chunker import tokenize
from paper_research_agent.context.models import PromptMessage

TokenEstimator = Callable[[str], int]


class ContextBudgetExceeded(ValueError):
    """Trusted rules and required request data do not fit the input budget."""


def conservative_token_count(text: str) -> int:
    """Conservatively estimate tokens across English, Chinese, and control text.

    A real model tokenizer can be injected into the assembler. This fallback uses
    UTF-8 bytes so a long unspaced CJK string cannot be mistaken for one token.
    """
    return max(1, len(tokenize(text)), math.ceil(len(text.encode("utf-8")) / 3))


def estimate_messages(
    messages: Sequence[PromptMessage],
    estimator: TokenEstimator = conservative_token_count,
) -> int:
    # Reserve a small, explicit framing overhead for role and message separators.
    return sum(estimator(message.content) + estimator(message.role) + 4 for message in messages)
