"""In-memory, one-time approval grants bound to exact tool arguments."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass(frozen=True)
class ApprovalRequest:
    request_id: str
    tool_name: str
    arguments_sha256: str
    expires_at_epoch: float


class ApprovalManager:
    def __init__(self, *, ttl_seconds: float = 300, secret: bytes | None = None):
        if ttl_seconds <= 0 or ttl_seconds > 3600:
            raise ValueError("approval TTL must be between 0 and 3600 seconds")
        self._ttl = ttl_seconds
        self._secret = secret or secrets.token_bytes(32)
        self._pending: dict[str, ApprovalRequest] = {}
        self._grants: dict[str, ApprovalRequest] = {}
        self._lock = threading.Lock()

    def request(self, tool_name: str, arguments: BaseModel) -> ApprovalRequest:
        request_id = secrets.token_hex(16)
        request = ApprovalRequest(
            request_id=request_id,
            tool_name=tool_name,
            arguments_sha256=arguments_fingerprint(arguments),
            expires_at_epoch=time.time() + self._ttl,
        )
        with self._lock:
            self._purge()
            self._pending[request_id] = request
        return request

    def approve(self, request_id: str) -> str:
        with self._lock:
            self._purge()
            request = self._pending.pop(request_id, None)
            if request is None:
                raise ValueError("approval request is missing or expired")
            token = hmac.new(
                self._secret,
                f"{request.request_id}:{request.arguments_sha256}".encode(),
                hashlib.sha256,
            ).hexdigest()
            self._grants[token] = request
            return token

    def consume(self, tool_name: str, arguments: BaseModel, token: str | None) -> bool:
        if token is None:
            return False
        with self._lock:
            self._purge()
            request = self._grants.pop(token, None)
        return bool(
            request
            and request.tool_name == tool_name
            and request.arguments_sha256 == arguments_fingerprint(arguments)
        )

    def _purge(self) -> None:
        now = time.time()
        self._pending = {
            key: value for key, value in self._pending.items() if value.expires_at_epoch > now
        }
        self._grants = {
            key: value for key, value in self._grants.items() if value.expires_at_epoch > now
        }


def arguments_fingerprint(arguments: BaseModel) -> str:
    payload = arguments.model_dump(mode="json", exclude={"approval_token"})
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
