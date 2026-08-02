"""Sanitized DashScope adapter for structured, evidence-grounded answers."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from paper_research_agent.answering.config import AnsweringConfig
from paper_research_agent.answering.models import (
    AnswerRequest,
    GenerationResult,
    ProviderAnswer,
)

DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
DEFAULT_BASE_URL_ENV = "DASHSCOPE_BASE_URL"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class AsyncAnswerGenerator(Protocol):
    model_id: str
    prompt_version: str

    async def generate(self, request: AnswerRequest) -> GenerationResult: ...


class AnswerGenerationError(RuntimeError):
    """A provider failure that never includes credentials, evidence, or response bodies."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        attempts: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.attempts = attempts
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class UnavailableAnswerGenerator:
    """Allow evidence-free local short-circuiting without provider credentials."""

    def __init__(self, model_id: str, prompt_version: str):
        self.model_id = model_id
        self.prompt_version = prompt_version

    async def generate(self, request: AnswerRequest) -> GenerationResult:
        del request
        raise AnswerGenerationError("answer generation credentials are unavailable")


class DashScopeAnswerGenerator:
    """Call a pinned Qwen model through DashScope's OpenAI-compatible endpoint."""

    def __init__(
        self,
        config: AnsweringConfig,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        base_url_env: str = DEFAULT_BASE_URL_ENV,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        resolved_key = api_key or os.getenv(api_key_env)
        if client is None and not resolved_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未配置")
        resolved_base_url = base_url or os.getenv(base_url_env) or DEFAULT_BASE_URL
        self.model_id = config.model
        self.prompt_version = config.prompt_version
        self._config = config
        self._endpoint = resolved_base_url.rstrip("/") + "/chat/completions"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            headers={"Authorization": f"Bearer {resolved_key}"},
            timeout=httpx.Timeout(config.timeout_seconds),
        )
        self._sleep = sleep

    async def generate(self, request: AnswerRequest) -> GenerationResult:
        if self._config.max_output_tokens > request.context.output_reserve_tokens:
            raise ValueError("answer max_output_tokens exceeds the context output reserve")
        payload: dict[str, object] = {
            "model": self.model_id,
            "messages": [message.model_dump(mode="json") for message in request.context.messages],
            "response_format": {"type": "json_object"},
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "enable_thinking": self._config.enable_thinking,
            "max_tokens": self._config.max_output_tokens,
        }
        started = time.perf_counter()
        input_tokens = 0
        output_tokens = 0
        attempts = 0
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                for attempt in range(1, self._config.max_retries + 2):
                    attempts = attempt
                    try:
                        response = await self._client.post(self._endpoint, json=payload)
                    except httpx.TimeoutException:
                        if attempt <= self._config.max_retries:
                            await self._sleep(_backoff_seconds(attempt))
                            continue
                        raise AnswerGenerationError(
                            "answer generation timed out",
                            attempts=attempts,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                        ) from None
                    except httpx.RequestError as error:
                        if attempt <= self._config.max_retries:
                            await self._sleep(_backoff_seconds(attempt))
                            continue
                        raise AnswerGenerationError(
                            f"answer generation network error: {type(error).__name__}",
                            attempts=attempts,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                        ) from None

                    if response.is_error:
                        error_code = _provider_error_code(response)
                        if (
                            response.status_code in RETRYABLE_STATUS_CODES
                            and attempt <= self._config.max_retries
                        ):
                            await self._sleep(_retry_delay(response, attempt))
                            continue
                        raise AnswerGenerationError(
                            f"answer generation request failed with HTTP {response.status_code}",
                            status_code=response.status_code,
                            error_code=error_code,
                            attempts=attempts,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                        )

                    try:
                        body = response.json()
                    except json.JSONDecodeError:
                        body = None
                    if isinstance(body, dict):
                        used_input, used_output = _usage(body.get("usage"))
                        input_tokens += used_input
                        output_tokens += used_output
                    try:
                        if not isinstance(body, dict):
                            raise TypeError("response is not an object")
                        content = _response_content(body)
                        ProviderAnswer.model_validate_json(content)
                    except (ValueError, TypeError, ValidationError):
                        if attempt <= self._config.max_retries:
                            await self._sleep(_backoff_seconds(attempt))
                            continue
                        raise AnswerGenerationError(
                            "answer generation returned an invalid response",
                            attempts=attempts,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                        ) from None

                    actual_model_value = body.get("model")
                    actual_model = (
                        actual_model_value.strip()
                        if isinstance(actual_model_value, str) and actual_model_value.strip()
                        else self.model_id
                    )
                    return GenerationResult(
                        content=content,
                        requested_model=self.model_id,
                        actual_model=actual_model,
                        prompt_version=self.prompt_version,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        attempts=attempts,
                    )
        except TimeoutError:
            raise AnswerGenerationError(
                "answer generation exceeded its total deadline",
                attempts=attempts,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ) from None
        raise AssertionError("answer generation loop exited unexpectedly")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _response_content(response: Mapping[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("answer response is missing choices")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("answer response did not finish normally")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise TypeError("answer response is missing message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("answer response content is empty")
    return content.strip()


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


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(float(retry_after), 0)
        except ValueError:
            pass
    return _backoff_seconds(attempt)


def _backoff_seconds(attempt: int) -> float:
    return float(min(2 ** (attempt - 1), 4))
