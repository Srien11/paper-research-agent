"""Async Chinese/English dual-route retrieval over the existing local index."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, TypeAlias

from paper_research_agent.chunking.chunker import canonical_sha256
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.retrieval.bm25 import BM25Index
from paper_research_agent.retrieval.config import (
    BilingualRetrievalConfig,
    RetrievalConfig,
)
from paper_research_agent.retrieval.contracts import (
    BilingualRetrievalRun,
    QueryRewriteTrace,
    SearchHit,
)
from paper_research_agent.retrieval.hybrid import reciprocal_rank_fusion
from paper_research_agent.retrieval.query_rewrite import (
    AsyncQueryRewriter,
    QueryRewriteResult,
)
from paper_research_agent.retrieval.query_store import (
    AuditRanking,
    CacheLookup,
    QueryAuditRecord,
    QueryRewriteCache,
    SQLiteQueryAuditLogger,
    rewrite_cache_key,
)
from paper_research_agent.retrieval.rerank import Reranker, rerank
from paper_research_agent.retrieval.rights import CorpusRightsMap
from paper_research_agent.retrieval.vector import SearchableVectorIndex

Candidate: TypeAlias = tuple[
    EvidenceChunk,
    float,
    dict[str, int],
    dict[str, float],
]
RankedChunks: TypeAlias = list[tuple[EvidenceChunk, float]]


@dataclass(frozen=True)
class RecallRoute:
    name: str
    sparse: RankedChunks
    vector: RankedChunks
    candidates: list[Candidate]
    latency_ms: float


@dataclass(frozen=True)
class RewriteResolution:
    trace: QueryRewriteTrace
    cache_lookup: CacheLookup


class BilingualRetrievalService:
    """Keep local ML serialized while overlapping Chinese recall with the API call."""

    def __init__(
        self,
        sparse: BM25Index,
        vector: SearchableVectorIndex,
        reranker: Reranker,
        rewriter: AsyncQueryRewriter,
        cache: QueryRewriteCache,
        audit: SQLiteQueryAuditLogger | None,
        retrieval_config: RetrievalConfig,
        bilingual_config: BilingualRetrievalConfig,
        *,
        index_id: str,
        rights: CorpusRightsMap | None = None,
        local_executor: Executor | None = None,
    ):
        if rewriter.model_id != bilingual_config.rewrite_model:
            raise ValueError("rewriter model does not match bilingual configuration")
        if rewriter.prompt_version != bilingual_config.rewrite_prompt_version:
            raise ValueError("rewriter prompt version does not match bilingual configuration")
        self.sparse = sparse
        self.vector = vector
        self.reranker = reranker
        self.rewriter = rewriter
        self.cache = cache
        self.audit = audit
        self.retrieval_config = retrieval_config
        self.bilingual_config = bilingual_config
        self.index_id = index_id
        self.rights = rights
        self._owns_executor = local_executor is None
        self._local_executor = local_executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="paper-retrieval-local-ml",
        )
        self._flight_lock = asyncio.Lock()
        self._rewrite_flights: dict[str, asyncio.Task[RewriteResolution]] = {}
        self._closed = False

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: Mapping[str, str] | None = None,
        privacy_ttl_days: int | None = None,
    ) -> BilingualRetrievalRun:
        original_query = query.strip()
        if self._closed:
            raise RuntimeError("bilingual retrieval service is closed")
        if not original_query:
            raise ValueError("query cannot be blank")
        limit = self.retrieval_config.top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")
        if privacy_ttl_days is not None and privacy_ttl_days <= 0:
            raise ValueError("privacy_ttl_days must be positive")

        request_id = uuid.uuid4().hex
        created_at = datetime.now(UTC)
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        rewrite_task = asyncio.create_task(
            self._resolve_rewrite(original_query, privacy_ttl_days=privacy_ttl_days)
        )
        zh_future = loop.run_in_executor(
            self._local_executor,
            functools.partial(self._recall, "zh", original_query, filters),
        )
        try:
            zh_route = await zh_future
            rewrite = await rewrite_task
        except BaseException:
            rewrite_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await rewrite_task
            raise

        routes = {"zh": zh_route}
        route_latencies: dict[str, float] = {"zh_recall": zh_route.latency_ms}
        degraded = rewrite.trace.status in {"timeout", "error", "stale_cache"}
        if rewrite.trace.status == "stale_cache":
            degraded_reason = f"query_rewrite_{rewrite.trace.fallback_reason}_using_stale_cache"
        elif degraded:
            degraded_reason = f"query_rewrite_{rewrite.trace.status}"
        else:
            degraded_reason = None

        if rewrite.trace.english_query is not None:
            en_route = await loop.run_in_executor(
                self._local_executor,
                functools.partial(
                    self._recall,
                    "en",
                    rewrite.trace.english_query,
                    filters,
                ),
            )
            routes["en"] = en_route
            route_latencies["en_recall"] = en_route.latency_ms

        cross_started = time.perf_counter()
        fused = fuse_language_routes(
            {name: route.candidates for name, route in routes.items()},
            rrf_k=self.bilingual_config.route_rrf_k,
            top_k=self.retrieval_config.rerank_candidates,
        )
        route_latencies["cross_route_rrf"] = _elapsed_ms(cross_started)

        if rewrite.trace.english_query is not None:
            rerank_started = time.perf_counter()
            final_candidates = rerank(
                rewrite.trace.english_query,
                fused,
                self.reranker,
            )
            route_latencies["rerank"] = _elapsed_ms(rerank_started)
        else:
            final_candidates = fused
        hits = _build_hits(final_candidates, limit=limit)
        config_sha256 = canonical_sha256(
            {
                "retrieval": self.retrieval_config.model_dump(mode="json"),
                "bilingual": self.bilingual_config.model_dump(mode="json"),
            }
        )
        storage_classes = self.rights.for_hits(hits) if self.rights is not None else {}
        route_latencies["total"] = _elapsed_ms(started)
        run = BilingualRetrievalRun(
            pipeline_id=self.bilingual_config.pipeline_id,
            original_query=original_query,
            rewrite=rewrite.trace,
            degraded=degraded,
            degraded_reason=degraded_reason,
            top_k=limit,
            hits=hits,
            index_id=self.index_id,
            config_sha256=config_sha256,
            storage_classes=storage_classes,
            rights_status="loaded" if self.rights is not None else "not_loaded",
        )
        persisted = await self._write_audit(
            QueryAuditRecord(
                request_id=request_id,
                created_at=created_at,
                original_query=original_query,
                rewrite=rewrite.trace,
                pipeline_id=self.bilingual_config.pipeline_id,
                index_id=self.index_id,
                config_sha256=config_sha256,
                degraded_reason=degraded_reason,
                latency_ms=route_latencies,
                rankings=_audit_rankings(routes, fused, hits),
                plaintext_days=privacy_ttl_days,
            )
        )
        return run.model_copy(update={"audit_persisted": persisted})

    def close(self) -> None:
        if self._owns_executor and isinstance(self._local_executor, ThreadPoolExecutor):
            self._local_executor.shutdown(wait=True, cancel_futures=True)

    async def aclose(self) -> None:
        """Cancel provider work before shutting down local executors or HTTP clients."""
        self._closed = True
        async with self._flight_lock:
            tasks = tuple(self._rewrite_flights.values())
            self._rewrite_flights.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.to_thread(self.close)

    def _recall(
        self,
        route: str,
        query: str,
        filters: Mapping[str, str] | None,
    ) -> RecallRoute:
        started = time.perf_counter()
        vector = self.vector.search(
            query,
            self.retrieval_config.vector_candidates,
            filters=filters,
        )
        sparse = self.sparse.search(
            query,
            self.retrieval_config.sparse_candidates,
            filters=filters,
        )
        candidates = reciprocal_rank_fusion(
            sparse,
            vector,
            rrf_k=self.retrieval_config.rrf_k,
            top_k=self.retrieval_config.rerank_candidates,
        )
        namespaced: list[Candidate] = []
        for rank, (chunk, score, ranks, scores) in enumerate(candidates, start=1):
            namespaced.append(
                (
                    chunk,
                    score,
                    {
                        **{f"{route}.{name}": value for name, value in ranks.items()},
                        f"{route}.route_rrf": rank,
                    },
                    {
                        **{f"{route}.{name}": value for name, value in scores.items()},
                        f"{route}.route_rrf": score,
                    },
                )
            )
        return RecallRoute(
            name=route,
            sparse=sparse,
            vector=vector,
            candidates=namespaced,
            latency_ms=_elapsed_ms(started),
        )

    async def _resolve_rewrite(
        self, query: str, *, privacy_ttl_days: int | None = None
    ) -> RewriteResolution:
        flight_key = rewrite_cache_key(
            query,
            model=self.rewriter.model_id,
            prompt_version=self.rewriter.prompt_version,
        )
        if privacy_ttl_days is not None:
            flight_key = f"{flight_key}:privacy:{privacy_ttl_days}"
        async with self._flight_lock:
            task = self._rewrite_flights.get(flight_key)
            if task is None:
                task = asyncio.create_task(
                    self._rewrite_once(query, privacy_ttl_days=privacy_ttl_days)
                )
                self._rewrite_flights[flight_key] = task
                task.add_done_callback(functools.partial(self._cleanup_rewrite_flight, flight_key))
        return await asyncio.shield(task)

    def _cleanup_rewrite_flight(
        self,
        flight_key: str,
        task: asyncio.Task[RewriteResolution],
    ) -> None:
        if self._rewrite_flights.get(flight_key) is task:
            self._rewrite_flights.pop(flight_key, None)

    async def _rewrite_once(
        self, query: str, *, privacy_ttl_days: int | None = None
    ) -> RewriteResolution:
        started = time.perf_counter()
        cache_error_class: str | None = None
        fresh_days = self.bilingual_config.rewrite_cache_fresh_days
        stale_days = self.bilingual_config.rewrite_cache_stale_days
        if privacy_ttl_days is not None:
            fresh_days = min(fresh_days, privacy_ttl_days)
            stale_days = min(stale_days, privacy_ttl_days)
        try:
            lookup = await asyncio.to_thread(
                self.cache.lookup,
                query,
                model=self.rewriter.model_id,
                prompt_version=self.rewriter.prompt_version,
                fresh_days=fresh_days,
                stale_days=stale_days,
            )
        except Exception as error:  # noqa: BLE001 - cache is strictly best-effort
            lookup = CacheLookup()
            cache_error_class = type(error).__name__
        if lookup.fresh is not None:
            return RewriteResolution(
                trace=_trace_from_cached(
                    lookup.fresh,
                    status="cache_hit",
                    requested_model=self.rewriter.model_id,
                    prompt_version=self.rewriter.prompt_version,
                    latency_ms=_elapsed_ms(started),
                    cache_error_class=cache_error_class,
                ),
                cache_lookup=lookup,
            )
        try:
            async with asyncio.timeout(self.bilingual_config.rewrite_timeout_seconds):
                result = await self.rewriter.rewrite(query)
        except TimeoutError:
            return self._rewrite_failure(
                lookup,
                status="timeout",
                error_class="TimeoutError",
                started=started,
                cache_error_class=cache_error_class,
            )
        except Exception as error:  # noqa: BLE001 - provider adapters are an isolation boundary
            return self._rewrite_failure(
                lookup,
                status="error",
                error_class=type(error).__name__,
                started=started,
                cache_error_class=cache_error_class,
            )

        latency_ms = _elapsed_ms(started)
        try:
            await asyncio.to_thread(
                self._cache_result,
                query,
                result,
                latency_ms,
                stale_days,
            )
        except Exception as error:  # noqa: BLE001 - cache cannot fail retrieval
            cache_error_class = type(error).__name__
        return RewriteResolution(
            trace=QueryRewriteTrace(
                status="success",
                english_query=result.english_query,
                requested_model=self.rewriter.model_id,
                actual_model=result.actual_model,
                prompt_version=self.rewriter.prompt_version,
                latency_ms=latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cache_error_class=cache_error_class,
            ),
            cache_lookup=lookup,
        )

    def _rewrite_failure(
        self,
        lookup: CacheLookup,
        *,
        status: Literal["timeout", "error"],
        error_class: str,
        started: float,
        cache_error_class: str | None,
    ) -> RewriteResolution:
        if lookup.stale is not None:
            return RewriteResolution(
                trace=_trace_from_cached(
                    lookup.stale,
                    status="stale_cache",
                    requested_model=self.rewriter.model_id,
                    prompt_version=self.rewriter.prompt_version,
                    latency_ms=_elapsed_ms(started),
                    error_class=error_class,
                    fallback_reason=status,
                    cache_error_class=cache_error_class,
                ),
                cache_lookup=lookup,
            )
        return RewriteResolution(
            trace=QueryRewriteTrace(
                status=status,
                requested_model=self.rewriter.model_id,
                prompt_version=self.rewriter.prompt_version,
                latency_ms=_elapsed_ms(started),
                error_class=error_class,
                fallback_reason=status,
                cache_error_class=cache_error_class,
            ),
            cache_lookup=lookup,
        )

    def _cache_result(
        self,
        query: str,
        result: QueryRewriteResult,
        latency_ms: float,
        stale_days: int,
    ) -> None:
        self.cache.put(
            query,
            model=self.rewriter.model_id,
            prompt_version=self.rewriter.prompt_version,
            english_query=result.english_query,
            actual_model=result.actual_model,
            latency_ms=latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            stale_days=stale_days,
        )

    async def _write_audit(self, record: QueryAuditRecord) -> bool:
        if self.audit is None:
            return False
        return await asyncio.to_thread(self.audit.write, record)


def fuse_language_routes(
    routes: Mapping[str, Sequence[Candidate]],
    *,
    rrf_k: int,
    top_k: int | None = None,
) -> list[Candidate]:
    """Second-level RRF gives each language route one vote regardless of recall channels."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    chunks: dict[str, EvidenceChunk] = {}
    route_ranks: dict[str, dict[str, int]] = {}
    stage_ranks: dict[str, dict[str, int]] = {}
    stage_scores: dict[str, dict[str, float]] = {}
    for route_name, candidates in routes.items():
        if not route_name.strip():
            raise ValueError("route name cannot be blank")
        seen: set[str] = set()
        for rank, (chunk, _score, ranks, scores) in enumerate(candidates, start=1):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks[chunk.chunk_id] = chunk
            route_ranks.setdefault(chunk.chunk_id, {})[route_name] = rank
            stage_ranks.setdefault(chunk.chunk_id, {}).update(ranks)
            stage_scores.setdefault(chunk.chunk_id, {}).update(scores)
    fused: list[Candidate] = []
    for chunk_id, ranks in route_ranks.items():
        score = sum(1 / (rrf_k + rank) for rank in ranks.values())
        fused.append(
            (
                chunks[chunk_id],
                score,
                dict(stage_ranks[chunk_id]),
                {**stage_scores[chunk_id], "cross_route_rrf": score},
            )
        )
    fused.sort(key=lambda item: (-item[1], item[0].chunk_id))
    ranked = [
        (chunk, score, {**ranks, "cross_route_rrf": rank}, scores)
        for rank, (chunk, score, ranks, scores) in enumerate(fused, start=1)
    ]
    return ranked[:top_k] if top_k is not None else ranked


def _build_hits(candidates: Sequence[Candidate], *, limit: int) -> tuple[SearchHit, ...]:
    return tuple(
        SearchHit(
            chunk_id=chunk.chunk_id,
            corpus_id=chunk.corpus_id,
            asset_id=chunk.asset_id,
            section_id=chunk.section_id,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            text_sha256=chunk.text_sha256,
            evidence_type=chunk.evidence_type,
            figure=chunk.figure,
            scores=scores,
            ranks={**ranks, "final": rank},
            final_score=score,
            final_rank=rank,
        )
        for rank, (chunk, score, ranks, scores) in enumerate(candidates[:limit], start=1)
    )


def _trace_from_cached(
    entry: object,
    *,
    status: Literal["cache_hit", "stale_cache"],
    requested_model: str,
    prompt_version: str,
    latency_ms: float,
    error_class: str | None = None,
    fallback_reason: Literal["timeout", "error"] | None = None,
    cache_error_class: str | None = None,
) -> QueryRewriteTrace:
    from paper_research_agent.retrieval.query_store import CachedRewrite

    if not isinstance(entry, CachedRewrite):
        raise TypeError("cache returned an invalid rewrite entry")
    return QueryRewriteTrace(
        status=status,
        english_query=entry.english_query,
        requested_model=requested_model,
        actual_model=entry.actual_model,
        prompt_version=prompt_version,
        latency_ms=latency_ms,
        input_tokens=0,
        output_tokens=0,
        error_class=error_class,
        fallback_reason=fallback_reason,
        cache_error_class=cache_error_class,
    )


def _audit_rankings(
    routes: Mapping[str, RecallRoute],
    fused: Sequence[Candidate],
    hits: Sequence[SearchHit],
) -> tuple[AuditRanking, ...]:
    records: list[AuditRanking] = []
    for route_name, route in routes.items():
        for stage_name, ranked in (("bm25", route.sparse), ("vector", route.vector)):
            records.extend(
                AuditRanking(
                    stage=f"{route_name}.{stage_name}",
                    chunk_id=chunk.chunk_id,
                    rank=rank,
                    score=score,
                )
                for rank, (chunk, score) in enumerate(ranked, start=1)
            )
        records.extend(
            AuditRanking(
                stage=f"{route_name}.route_rrf",
                chunk_id=chunk.chunk_id,
                rank=rank,
                score=score,
            )
            for rank, (chunk, score, _ranks, _scores) in enumerate(route.candidates, start=1)
        )
    records.extend(
        AuditRanking(
            stage="cross_route_rrf",
            chunk_id=chunk.chunk_id,
            rank=rank,
            score=score,
        )
        for rank, (chunk, score, _ranks, _scores) in enumerate(fused, start=1)
    )
    records.extend(
        AuditRanking(
            stage="final",
            chunk_id=hit.chunk_id,
            rank=hit.final_rank,
            score=hit.final_score,
            final_rank=hit.final_rank,
        )
        for hit in hits
    )
    return tuple(records)


def _elapsed_ms(started: float) -> float:
    return max((time.perf_counter() - started) * 1000, 0.0)
