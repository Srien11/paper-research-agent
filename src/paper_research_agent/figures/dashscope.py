"""通过百炼 OpenAI 兼容接口生成论文图片摘要。"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from threading import Lock
from typing import Any

from paper_research_agent.figures.summarizer import (
    PROMPT_VERSION,
    VisionSummaryResult,
    build_summary_prompt,
    parse_summary_response,
)

DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
DEFAULT_BASE_URL_ENV = "DASHSCOPE_BASE_URL"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
FREE_QUOTA_ERROR_CODE = "AllocationQuota.FreeTierOnly"


class DashScopeRequestError(RuntimeError):
    """不包含凭据或完整请求体的百炼调用错误。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds

    @property
    def free_quota_exhausted(self) -> bool:
        return self.error_code == FREE_QUOTA_ERROR_CODE


class NoAvailableVisionModelError(RuntimeError):
    """所有配置模型均因免费额度耗尽而不可用。"""


class ModelUsage:
    """按实际响应累计模型 Token 用量。"""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.requests = 0
        self._lock = Lock()

    def add(self, *, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.requests += 1

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return {
                "requests": self.requests,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
            }


ResponseRequester = Callable[[str, Mapping[str, object]], Mapping[str, object]]


class DashScopeVisionSummarizer:
    """调用一组百炼模型，并在免费额度明确耗尽时依次切换。"""

    def __init__(
        self,
        *,
        model_ids: Sequence[str],
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        base_url_env: str = DEFAULT_BASE_URL_ENV,
        timeout_seconds: int = 180,
        max_output_tokens: int | None = None,
        max_retries: int = 8,
        requester: ResponseRequester | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ):
        cleaned_models = tuple(model.strip() for model in model_ids if model.strip())
        if not cleaned_models:
            raise ValueError("至少需要配置一个百炼视觉模型")
        if len(cleaned_models) != len(set(cleaned_models)):
            raise ValueError("百炼视觉模型列表不能包含重复项")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须为正数")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("max_output_tokens 必须为正数")
        if max_retries < 0:
            raise ValueError("max_retries 不能为负数")

        resolved_key = api_key or os.getenv(api_key_env)
        if requester is None and not resolved_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未配置")
        resolved_base_url = base_url or os.getenv(base_url_env) or DEFAULT_BASE_URL

        self.model_ids = cleaned_models
        self.prompt_version = PROMPT_VERSION
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self._sleep = sleep
        self._jitter = jitter
        self._requester = requester or self._build_requester(
            api_key=resolved_key or "",
            base_url=resolved_base_url,
        )
        self._unavailable_models: set[str] = set()
        self._availability_lock = Lock()
        self.usage_by_model = {model: ModelUsage() for model in cleaned_models}

    def summarize(
        self,
        image_path: Path,
        *,
        figure_name: str,
        caption: str,
    ) -> VisionSummaryResult:
        if not image_path.is_file():
            raise FileNotFoundError(f"图片不存在: {image_path}")
        prompt = build_summary_prompt(figure_name=figure_name, caption=caption)
        image_data_url = _image_data_url(image_path)

        for model_id in self.model_ids:
            with self._availability_lock:
                if model_id in self._unavailable_models:
                    continue
            try:
                result = self._summarize_with_model(
                    model_id,
                    prompt=prompt,
                    image_data_url=image_data_url,
                )
            except DashScopeRequestError as error:
                if error.free_quota_exhausted:
                    with self._availability_lock:
                        self._unavailable_models.add(model_id)
                    continue
                raise
            return result
        raise NoAvailableVisionModelError(
            "所有配置模型均被免费额度限制拦截；请在百炼控制台关闭“仅免费额度”"
            "或完成付费认证后继续，现有 JSONL 可断点续跑"
        )

    def usage_report(self) -> dict[str, dict[str, int]]:
        return {
            model: usage.as_dict()
            for model, usage in self.usage_by_model.items()
            if usage.requests
        }

    def _summarize_with_model(
        self,
        model_id: str,
        *,
        prompt: str,
        image_data_url: str,
    ) -> VisionSummaryResult:
        payload: dict[str, object] = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "enable_thinking": False,
        }
        if self.max_output_tokens is not None:
            payload["max_tokens"] = self.max_output_tokens
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._requester(model_id, payload)
                input_tokens, output_tokens = _response_usage(response)
                self.usage_by_model[model_id].add(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                content = _response_content(response)
                summary = parse_summary_response(content)
                actual_model = response.get("model")
                return VisionSummaryResult(
                    summary=summary,
                    model_id=(
                        actual_model.strip()
                        if isinstance(actual_model, str) and actual_model.strip()
                        else model_id
                    ),
                )
            except DashScopeRequestError as error:
                if error.free_quota_exhausted or not error.retryable:
                    raise
                last_error = error
            except (ValueError, KeyError, TypeError) as error:
                last_error = error
            if attempt < self.max_retries:
                base_delay = (
                    last_error.retry_after_seconds
                    if isinstance(last_error, DashScopeRequestError)
                    and last_error.retry_after_seconds is not None
                    else min(2**attempt, 60)
                )
                self._sleep(base_delay + self._jitter(0, base_delay * 0.25))
        raise RuntimeError(
            f"百炼模型 {model_id} 连续返回无效响应，已重试 {self.max_retries} 次"
        ) from last_error

    def _build_requester(self, *, api_key: str, base_url: str) -> ResponseRequester:
        endpoint = base_url.rstrip("/") + "/chat/completions"

        def request(_model_id: str, payload: Mapping[str, object]) -> Mapping[str, object]:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            http_request = urllib.request.Request(
                endpoint,
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    http_request,
                    timeout=self.timeout_seconds,
                ) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                raise _http_error(error) from None
            except (urllib.error.URLError, TimeoutError) as error:
                raise DashScopeRequestError(
                    f"百炼网络请求失败: {type(error).__name__}",
                    retryable=True,
                ) from error
            if not isinstance(result, dict):
                raise DashScopeRequestError("百炼响应不是 JSON 对象")
            return result

        return request


def _image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _response_content(response: Mapping[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("百炼响应缺少 choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise TypeError("百炼 choice 格式无效")
    if choice.get("finish_reason") == "length":
        raise ValueError("百炼响应因输出长度限制被截断")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise TypeError("百炼响应缺少 message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("百炼响应内容为空")
    return content


def _response_usage(response: Mapping[str, object]) -> tuple[int, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return 0, 0
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    return _non_negative_int(input_tokens), _non_negative_int(output_tokens)


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _http_error(error: urllib.error.HTTPError) -> DashScopeRequestError:
    error_code: str | None = None
    message = "请求被服务端拒绝"
    try:
        payload = json.loads(error.read().decode("utf-8"))
        if isinstance(payload, dict):
            nested_error = payload.get("error")
            error_payload = nested_error if isinstance(nested_error, dict) else payload
            candidate_code = error_payload.get("code")
            candidate_message = error_payload.get("message")
            if isinstance(candidate_code, str):
                error_code = candidate_code
            if isinstance(candidate_message, str) and candidate_message.strip():
                message = candidate_message.strip()[:300]
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    retry_after_seconds: float | None = None
    retry_after = error.headers.get("Retry-After") if error.headers is not None else None
    if retry_after is not None:
        try:
            retry_after_seconds = max(float(retry_after), 0)
        except ValueError:
            retry_after_seconds = None
    return DashScopeRequestError(
        f"百炼请求失败（HTTP {error.code}, code={error_code or 'unknown'}）: {message}",
        status_code=error.code,
        error_code=error_code,
        retryable=error.code in RETRYABLE_STATUS_CODES,
        retry_after_seconds=retry_after_seconds,
    )
