"""Conversation RAG over user questions, with deterministic ambiguity gates."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime

from paper_research_agent.conversation.models import (
    ConversationCandidate,
    ConversationContextSnapshot,
    ConversationEpisode,
    ConversationResolution,
    ConversationTurn,
    TurnInterpretation,
)

_REFERENTIAL = re.compile(
    r"(?:它|其|这个|这些|该(?:方法|模型|论文|结果|指标)|上述|前者|后者|刚才|"
    r"继续|接着|回到(?:之前|先前|前面|上次)|(?:再|重新)(?:说|回答|解释|分析|总结|补充))"
)
_KNOWLEDGE_BASE_COMMAND = re.compile(
    r"(?:(?:结合|参考|依据|根据|按照|按|用|查)(?:一下)?(?:本地)?(?:论文)?知识库)"
)
_DEPENDENT = re.compile(
    rf"(?:{_REFERENTIAL.pattern}|{_KNOWLEDGE_BASE_COMMAND.pattern})"
)
_DIRECTIVE = re.compile(
    r"(?:请|麻烦)?(?:继续|接着|再)?"
    r"(?:结合|参考|依据|根据|按照|按|用|查)(?:一下)?"
    r"(?:本地)?(?:论文)?知识库(?:来|中|里|一下)?|"
    r"(?:再|重新)(?:说|回答|解释|分析|总结|补充)(?:一遍|一次|一下)?|"
    r"回到(?:之前|先前|前面|上次)(?:的)?"
)
_PUNCTUATION = re.compile(r"[\s，。！？、；：,.!?;:'\"“”‘’（）()\[\]{}]+")
_ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]{1,}")
_CHINESE_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")


def build_conversation_context(
    question: str,
    history: Sequence[ConversationTurn],
    *,
    episodes: Sequence[ConversationEpisode] = (),
    recent_limit: int = 6,
    recall_limit: int = 5,
) -> ConversationContextSnapshot:
    normalized = question.strip()
    if not normalized:
        raise ValueError("question cannot be blank")
    if recent_limit <= 0 or recent_limit > 12:
        raise ValueError("recent context limit must be between 1 and 12")
    if recall_limit <= 0 or recall_limit > 10:
        raise ValueError("conversation recall limit must be between 1 and 10")
    completed = tuple(turn for turn in history if turn.status != "pending")
    recent_source = completed[-recent_limit:]
    recent = tuple(
        _candidate(turn, max(0.7, 1.0 - (len(recent_source) - index - 1) * 0.05))
        for index, turn in enumerate(recent_source)
    )
    recent_ids = {item.turn_id for item in recent}
    ranked = _rank(normalized, completed, episodes)
    recalled = tuple(
        item
        for item, _score in ranked
        if item.turn_id not in recent_ids
    )[:recall_limit]
    return ConversationContextSnapshot(
        original_question=normalized,
        recent_turns=recent,
        recalled_turns=recalled,
        episodes=tuple(episodes),
        prepared_at=datetime.now(UTC),
    )


def resolution_from_interpretation(
    snapshot: ConversationContextSnapshot,
    interpretation: TurnInterpretation,
) -> ConversationResolution:
    candidates = snapshot.candidates
    candidate_ids = {item.turn_id for item in candidates}
    unknown = tuple(
        turn_id
        for turn_id in interpretation.selected_history_turn_ids
        if turn_id not in candidate_ids
    )
    if unknown:
        raise ValueError("turn interpreter selected an unknown conversation turn")
    selected = set(interpretation.selected_history_turn_ids)
    anchor = next((item for item in candidates if item.turn_id in selected), None)
    return ConversationResolution(
        original_question=snapshot.original_question,
        standalone_question=interpretation.standalone_question,
        chinese_query=interpretation.chinese_query,
        candidates=candidates,
        selected_turn_ids=interpretation.selected_history_turn_ids,
        confidence=interpretation.confidence,
        needs_clarification=interpretation.needs_clarification,
        clarification_question=interpretation.clarification_question,
        episode_id=(
            anchor.episode_id
            if anchor is not None and anchor.episode_id is not None
            else _episode_id(interpretation.standalone_question)
        ),
        recent_context_turn_count=len(snapshot.recent_turns),
        recalled_candidate_count=len(snapshot.recalled_turns),
        interpretation_source="model",
    )


def fallback_resolution_from_context(
    snapshot: ConversationContextSnapshot,
) -> ConversationResolution:
    """Deterministic safety fallback used only when turn interpretation is unavailable."""
    question = snapshot.original_question
    topic_query = _topic_query(question)
    referential = _REFERENTIAL.search(question) is not None
    if _has_explicit_topic(topic_query) and not referential:
        return ConversationResolution(
            original_question=question,
            standalone_question=question,
            chinese_query=question,
            candidates=snapshot.candidates,
            confidence=0.5,
            episode_id=_episode_id(topic_query),
            recent_context_turn_count=len(snapshot.recent_turns),
            recalled_candidate_count=len(snapshot.recalled_turns),
            interpretation_source="fallback",
        )
    if snapshot.recent_turns:
        anchor = snapshot.recent_turns[-1]
    elif snapshot.recalled_turns:
        anchor = snapshot.recalled_turns[0]
    else:
        return ConversationResolution(
            original_question=question,
            standalone_question=question,
            chinese_query=question,
            confidence=0,
            needs_clarification=True,
            clarification_question="你希望继续讨论哪个主题？",
            episode_id=_episode_id(question),
            recent_context_turn_count=0,
            recalled_candidate_count=0,
            interpretation_source="fallback",
        )
    standalone = _standalone(question, topic_query, anchor.standalone_question)
    return ConversationResolution(
        original_question=question,
        standalone_question=standalone,
        chinese_query=standalone,
        candidates=snapshot.candidates,
        selected_turn_ids=(anchor.turn_id,),
        confidence=0.5,
        episode_id=anchor.episode_id or _episode_id(anchor.standalone_question),
        recent_context_turn_count=len(snapshot.recent_turns),
        recalled_candidate_count=len(snapshot.recalled_turns),
        interpretation_source="fallback",
    )
def resolve_conversation_question(
    question: str,
    history: Sequence[ConversationTurn],
    *,
    episodes: Sequence[ConversationEpisode] = (),
    candidate_limit: int = 5,
    ambiguity_margin: float = 0.25,
) -> ConversationResolution:
    normalized = question.strip()
    if not normalized:
        raise ValueError("question cannot be blank")
    completed = tuple(turn for turn in history if turn.status != "pending")
    dependent = _DEPENDENT.search(normalized) is not None
    referential = _REFERENTIAL.search(normalized) is not None
    topic_query = _topic_query(normalized)

    if not completed or (_has_explicit_topic(topic_query) and not referential):
        episode_id = _episode_id(topic_query or normalized)
        return ConversationResolution(
            original_question=normalized,
            standalone_question=normalized,
            chinese_query=normalized,
            confidence=1.0,
            episode_id=episode_id,
        )

    ranked = _rank(topic_query or normalized, completed, episodes)
    if dependent and not _has_explicit_topic(topic_query):
        ranked = [(_candidate(completed[-1], 0.96), 0.96), *ranked]
        ranked = _deduplicate_ranked(ranked)
    ranked = ranked[:candidate_limit]
    candidates = tuple(item for item, _score in ranked)
    if not candidates or candidates[0].relevance < 0.24:
        return ConversationResolution(
            original_question=normalized,
            standalone_question=normalized,
            chinese_query=normalized,
            candidates=candidates,
            confidence=candidates[0].relevance if candidates else 0.0,
            needs_clarification=dependent,
            clarification_question=("你希望结合知识库继续讨论哪个主题？" if dependent else None),
            episode_id=_episode_id(topic_query or normalized),
        )

    best = candidates[0]
    ambiguous = (
        len(candidates) > 1
        and candidates[1].relevance >= 0.35
        and best.relevance - candidates[1].relevance < ambiguity_margin
        and _topic_identity(best) != _topic_identity(candidates[1])
    )
    if ambiguous:
        return ConversationResolution(
            original_question=normalized,
            standalone_question=normalized,
            chinese_query=normalized,
            candidates=candidates,
            confidence=best.relevance,
            needs_clarification=True,
            clarification_question=(
                f"你指的是“{best.user_question}”，还是“{candidates[1].user_question}”？"
            ),
            episode_id=best.episode_id,
        )

    anchor = re.sub(r"[。！？?]+$", "", best.standalone_question)
    standalone = _standalone(normalized, topic_query, anchor)
    return ConversationResolution(
        original_question=normalized,
        standalone_question=standalone,
        chinese_query=standalone,
        candidates=candidates,
        selected_turn_ids=(best.turn_id,),
        confidence=max(best.relevance, 0.75 if dependent else best.relevance),
        episode_id=best.episode_id or _episode_id(anchor),
    )


def _rank(
    query: str,
    history: Sequence[ConversationTurn],
    episodes: Sequence[ConversationEpisode],
) -> list[tuple[ConversationCandidate, float]]:
    total = max(len(history), 1)
    episode_summaries = {item.episode_id: item.summary for item in episodes}
    ranked: list[tuple[ConversationCandidate, float]] = []
    for index, turn in enumerate(history):
        target = turn.standalone_question or turn.user_question
        if turn.episode_id in episode_summaries:
            target = f"{target} {episode_summaries[turn.episode_id]}"
        lexical = _token_overlap(query, target)
        semantic = _ngram_cosine(query, target)
        recency = (index + 1) / total
        score = min(1.0, lexical * 0.50 + semantic * 0.38 + recency * 0.12)
        ranked.append((_candidate(turn, score), score))
    ranked.sort(key=lambda item: (-item[1], -item[0].sequence, item[0].turn_id))
    return ranked


def _candidate(turn: ConversationTurn, score: float) -> ConversationCandidate:
    return ConversationCandidate(
        turn_id=turn.turn_id,
        sequence=turn.sequence,
        user_question=turn.user_question,
        standalone_question=turn.standalone_question or turn.user_question,
        route=turn.route,
        assistant_summary=turn.assistant_summary,
        status=turn.status,
        episode_id=turn.episode_id,
        relevance=round(min(max(score, 0.0), 1.0), 4),
    )


def _deduplicate_ranked(
    ranked: Sequence[tuple[ConversationCandidate, float]],
) -> list[tuple[ConversationCandidate, float]]:
    result: list[tuple[ConversationCandidate, float]] = []
    seen: set[str] = set()
    for item in ranked:
        if item[0].turn_id in seen:
            continue
        seen.add(item[0].turn_id)
        result.append(item)
    return result


def _standalone(question: str, topic_query: str, anchor: str) -> str:
    if not _has_explicit_topic(topic_query):
        if "知识库" in question:
            return f"请基于本地论文知识库，继续分析{anchor}。"
        return f"围绕“{anchor}”，继续回答：{question}"
    if "回到之前" in question:
        return f"围绕“{anchor}”，继续回答：{question}"
    return question


def _topic_query(question: str) -> str:
    stripped = _DIRECTIVE.sub(" ", question)
    stripped = re.sub(
        r"(?:结合|解释|分析|讨论|看看|说说|说|回答|继续|补充)(?:一遍|一次|一下)?",
        " ",
        stripped,
    )
    return " ".join(stripped.split()).strip("，。！？,.!? ")


def _has_explicit_topic(value: str) -> bool:
    compact = _PUNCTUATION.sub("", value)
    return len(compact) >= 2 and compact not in {"一下", "一下吧", "这个", "它"}


def _tokens(value: str) -> set[str]:
    lowered = value.casefold()
    tokens = {match.group(0) for match in _ASCII_WORD.finditer(lowered)}
    for match in _CHINESE_RUN.finditer(lowered):
        run = match.group(0)
        tokens.add(run)
        tokens.update(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return tokens


def _token_overlap(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / math.sqrt(len(left_tokens) * len(right_tokens))


def _ngram_vector(value: str) -> Counter[str]:
    compact = _PUNCTUATION.sub("", value.casefold())
    if len(compact) < 2:
        return Counter({compact: 1}) if compact else Counter()
    return Counter(compact[index : index + 2] for index in range(len(compact) - 1))


def _ngram_cosine(left: str, right: str) -> float:
    lhs = _ngram_vector(left)
    rhs = _ngram_vector(right)
    if not lhs or not rhs:
        return 0.0
    dot = sum(value * rhs.get(key, 0) for key, value in lhs.items())
    lhs_norm = math.sqrt(sum(value * value for value in lhs.values()))
    rhs_norm = math.sqrt(sum(value * value for value in rhs.values()))
    return dot / (lhs_norm * rhs_norm) if lhs_norm and rhs_norm else 0.0


def _topic_identity(candidate: ConversationCandidate) -> str:
    return candidate.episode_id or _episode_id(candidate.standalone_question)


def _episode_id(value: str) -> str:
    normalized = _PUNCTUATION.sub("", value.casefold())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
