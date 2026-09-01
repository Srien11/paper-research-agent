"""Provider contracts and deterministic routing for scholarly metadata tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from paper_research_agent.agent.tooling.contracts import (
    CitationGraphInput,
    IdentifierInput,
    ScholarlySearchInput,
)

ScholarlyOperation = Literal["search", "resolve", "citations", "status"]
SCHOLARLY_OPERATIONS: frozenset[ScholarlyOperation] = frozenset(
    {"search", "resolve", "citations", "status"}
)
ScholarlyProviderRequest: TypeAlias = ScholarlySearchInput | IdentifierInput | CitationGraphInput


class ScholarlyProviderResult(BaseModel):
    """Normalized low-trust result returned by any scholarly provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    status: Literal["ok", "not_found", "insufficient"] = "ok"
    items: tuple[dict[str, Any], ...] = ()
    summary: dict[str, Any] = Field(default_factory=dict)


class ScholarlyProvider(Protocol):
    """Capability-aware provider contract; implementations must not mutate requests."""

    provider_id: str
    capabilities: frozenset[ScholarlyOperation]

    async def execute(
        self,
        operation: ScholarlyOperation,
        request: ScholarlyProviderRequest,
    ) -> ScholarlyProviderResult: ...


class ScholarlyProviderError(RuntimeError):
    """Base class for provider failures that may be handled by another provider."""

    reason_code = "provider_error"


class ScholarlyProviderUnavailable(ScholarlyProviderError):
    reason_code = "provider_unavailable"


class ScholarlyProviderTimeout(ScholarlyProviderUnavailable):
    reason_code = "provider_timeout"


class ScholarlyProviderRateLimited(ScholarlyProviderUnavailable):
    reason_code = "provider_rate_limited"

    def __init__(self, *, retry_after_seconds: float | None = None):
        super().__init__(self.reason_code)
        self.retry_after_seconds = retry_after_seconds


class ScholarlyProviderInvalidResponse(ScholarlyProviderError):
    reason_code = "provider_invalid_response"


class OfflineScholarlyProvider:
    """Production-safe provider used until live network access is explicitly enabled."""

    provider_id = "offline"
    capabilities = SCHOLARLY_OPERATIONS

    async def execute(
        self,
        operation: ScholarlyOperation,
        request: ScholarlyProviderRequest,
    ) -> ScholarlyProviderResult:
        del operation, request
        return ScholarlyProviderResult(
            provider_id=self.provider_id,
            status="insufficient",
            summary={"reason_code": "provider_not_configured", "network_accessed": False},
        )


class ScholarlyProviderRegistry:
    """Immutable ordered provider registry with bounded deterministic fallback."""

    def __init__(self, providers: Sequence[ScholarlyProvider]):
        resolved = tuple(providers)
        if not resolved:
            raise ValueError("scholarly provider registry requires at least one provider")
        provider_ids = tuple(provider.provider_id for provider in resolved)
        if len(set(provider_ids)) != len(provider_ids):
            raise ValueError("scholarly provider IDs must be unique")
        if any(not provider.capabilities for provider in resolved):
            raise ValueError("scholarly providers must declare at least one capability")
        if any(not provider.capabilities <= SCHOLARLY_OPERATIONS for provider in resolved):
            raise ValueError("scholarly provider declared an unknown capability")
        self._providers = resolved

    @property
    def provider_ids(self) -> tuple[str, ...]:
        return tuple(provider.provider_id for provider in self._providers)

    async def execute(
        self,
        operation: ScholarlyOperation,
        request: ScholarlyProviderRequest,
    ) -> ScholarlyProviderResult:
        attempts: list[dict[str, Any]] = []
        last_result: ScholarlyProviderResult | None = None
        for provider in self._providers:
            if operation not in provider.capabilities:
                continue
            try:
                result = await provider.execute(operation, request)
            except ScholarlyProviderError as exc:
                attempt: dict[str, Any] = {
                    "provider": provider.provider_id,
                    "status": "failed",
                    "reason_code": exc.reason_code,
                }
                if isinstance(exc, ScholarlyProviderRateLimited):
                    attempt["retry_after_seconds"] = exc.retry_after_seconds
                attempts.append(attempt)
                continue
            if result.provider_id != provider.provider_id:
                raise ValueError("scholarly provider result identity mismatch")
            last_result = result
            attempts.append({"provider": provider.provider_id, "status": result.status})
            if result.status == "ok":
                return _with_attempts(result, attempts)

        if last_result is not None:
            return _with_attempts(last_result, attempts)
        if attempts:
            return ScholarlyProviderResult(
                provider_id=attempts[-1]["provider"],
                status="insufficient",
                summary={
                    "reason_code": "all_providers_unavailable",
                    "provider_attempts": tuple(attempts),
                },
            )
        return ScholarlyProviderResult(
            provider_id="registry",
            status="insufficient",
            summary={
                "reason_code": "provider_capability_not_configured",
                "provider_attempts": (),
            },
        )


def _with_attempts(
    result: ScholarlyProviderResult,
    attempts: Sequence[Mapping[str, Any]],
) -> ScholarlyProviderResult:
    return result.model_copy(
        update={"summary": {**result.summary, "provider_attempts": tuple(attempts)}}
    )
