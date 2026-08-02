"""Short-deadline Chinese-to-English scientific query rewriting."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

QUERY_REWRITE_PROMPT_VERSION = "query-rewrite-v2"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
DEFAULT_BASE_URL_ENV = "DASHSCOPE_BASE_URL"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

SYSTEM_PROMPT = """You rewrite Chinese research questions into one concise English retrieval query.
Do not answer the question. Return exactly one JSON object with the field english_query.
Preserve model names, dataset names, abbreviations, metrics, numbers, units, signs, inequalities,
negations, scope restrictions, comparison direction, and other limiting conditions exactly.
Translate only descriptive Chinese content. Prefer scientific keywords and noun phrases suitable
for BM25, embedding retrieval, and an English cross-encoder reranker."""


@dataclass(frozen=True)
class QueryRewriteResult:
    english_query: str
    actual_model: str
    input_tokens: int = 0
    output_tokens: int = 0


class AsyncQueryRewriter(Protocol):
    model_id: str
    prompt_version: str

    async def rewrite(self, query: str) -> QueryRewriteResult: ...


class QueryRewriteError(RuntimeError):
    """A sanitized provider or response error with no credentials or request body."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class UnavailableQueryRewriter:
    """Keep local retrieval available when provider credentials are not configured."""

    def __init__(self, model_id: str, *, reason: str = "provider unavailable"):
        self.model_id = model_id
        self.prompt_version = QUERY_REWRITE_PROMPT_VERSION
        self._reason = reason

    async def rewrite(self, query: str) -> QueryRewriteResult:
        del query
        raise QueryRewriteError(self._reason)


class DashScopeQueryRewriter:
    """Use the DashScope OpenAI-compatible API without uploading corpus evidence."""

    def __init__(
        self,
        model_id: str,
        *,
        timeout_seconds: float = 2.0,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        base_url_env: str = DEFAULT_BASE_URL_ENV,
        client: httpx.AsyncClient | None = None,
    ):
        if not model_id.strip():
            raise ValueError("query rewrite model cannot be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        resolved_key = api_key or os.getenv(api_key_env)
        if client is None and not resolved_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未配置")
        resolved_base_url = base_url or os.getenv(base_url_env) or DEFAULT_BASE_URL

        self.model_id = model_id.strip()
        self.prompt_version = QUERY_REWRITE_PROMPT_VERSION
        self._endpoint = resolved_base_url.rstrip("/") + "/chat/completions"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            headers={"Authorization": f"Bearer {resolved_key}"},
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def rewrite(self, query: str) -> QueryRewriteResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be blank")
        payload: dict[str, object] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": normalized_query},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "top_p": 0.7,
            "enable_thinking": False,
            "max_tokens": 128,
        }
        try:
            response = await self._client.post(self._endpoint, json=payload)
        except httpx.TimeoutException as error:
            raise TimeoutError("query rewrite request timed out") from error
        except httpx.RequestError as error:
            raise QueryRewriteError(
                f"query rewrite network error: {type(error).__name__}"
            ) from error
        if response.is_error:
            error_code = _provider_error_code(response)
            raise QueryRewriteError(
                f"query rewrite request failed with HTTP {response.status_code}",
                status_code=response.status_code,
                error_code=error_code,
            )
        try:
            body = response.json()
        except json.JSONDecodeError as error:
            raise QueryRewriteError("query rewrite response is not JSON") from error
        if not isinstance(body, dict):
            raise QueryRewriteError("query rewrite response is not a JSON object")
        return _parse_rewrite_response(body, requested_model=self.model_id)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _parse_rewrite_response(
    response: Mapping[str, object],
    *,
    requested_model: str,
) -> QueryRewriteResult:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise QueryRewriteError("query rewrite response is missing choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise QueryRewriteError("query rewrite response is missing message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise QueryRewriteError("query rewrite response content is empty")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise QueryRewriteError("query rewrite content is not valid JSON") from error
    if not isinstance(parsed, dict) or set(parsed) != {"english_query"}:
        raise QueryRewriteError("query rewrite content has an invalid schema")
    english_query = parsed.get("english_query")
    if not isinstance(english_query, str) or not english_query.strip():
        raise QueryRewriteError("query rewrite produced an empty query")
    cleaned_query = english_query.strip()
    if len(cleaned_query) > 1024:
        raise QueryRewriteError("query rewrite exceeded 1024 characters")

    actual_model_value = response.get("model")
    actual_model = (
        actual_model_value.strip()
        if isinstance(actual_model_value, str) and actual_model_value.strip()
        else requested_model
    )
    input_tokens, output_tokens = _usage(response.get("usage"))
    return QueryRewriteResult(
        english_query=cleaned_query,
        actual_model=actual_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _usage(value: object) -> tuple[int, int]:
    if not isinstance(value, dict):
        return 0, 0
    return (
        _non_negative_int(value.get("prompt_tokens", value.get("input_tokens", 0))),
        _non_negative_int(value.get("completion_tokens", value.get("output_tokens", 0))),
    )


def _non_negative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _provider_error_code(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    nested = payload.get("error")
    error_payload = nested if isinstance(nested, dict) else payload
    code = error_payload.get("code")
    return code if isinstance(code, str) else None
