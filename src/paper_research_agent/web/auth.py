"""Owner credential verification and revocable signed Web sessions."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from paper_research_agent.web.config import OwnerCredentials


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    if not value or len(value) > 1_024:
        raise ValueError("invalid session token")
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(value + padding, altchars=b"-_", validate=True)


class CredentialVerifier:
    """Constant-time verification for direct or existing Zhimo PBKDF2 credentials."""

    def __init__(self, credentials: OwnerCredentials):
        self._credentials = credentials

    def verify(self, username: str, password: str) -> bool:
        if len(username) > 128 or len(password) > 512:
            return False
        username_ok = hmac.compare_digest(username, self._credentials.username)
        if self._credentials.password is not None:
            password_ok = hmac.compare_digest(password, self._credentials.password)
        else:
            candidate = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                self._credentials.salt or b"",
                self._credentials.pbkdf2_iterations,
            )
            password_ok = hmac.compare_digest(
                candidate,
                self._credentials.password_hash or b"",
            )
        return username_ok and password_ok


@dataclass(frozen=True, slots=True)
class OwnerSession:
    """Server-owned authentication session and current RAG conversation."""

    session_id: str
    conversation_id: str
    expires_at: int


class SessionManager:
    """Keep revocable owner sessions in-process and authenticate cookies with HMAC-SHA256."""

    def __init__(
        self,
        secret: bytes,
        ttl_seconds: int,
        *,
        clock: Callable[[], float] = time.time,
    ):
        if len(secret) < 32:
            raise ValueError("session secret must contain at least 32 bytes")
        if ttl_seconds <= 0:
            raise ValueError("session TTL must be positive")
        self._secret = secret
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._sessions: dict[str, OwnerSession] = {}
        self._lock = threading.Lock()

    def create(self) -> tuple[str, OwnerSession]:
        now = int(self._clock())
        session = OwnerSession(
            session_id=secrets.token_urlsafe(32),
            conversation_id=secrets.token_urlsafe(24),
            expires_at=now + self._ttl_seconds,
        )
        with self._lock:
            self._purge_expired(now)
            self._sessions[session.session_id] = session
        return self._encode(session), session

    def resolve(self, token: str | None) -> OwnerSession | None:
        if token is None:
            return None
        try:
            payload_segment, signature_segment = token.split(".", 1)
            payload = _b64decode(payload_segment)
            supplied_signature = _b64decode(signature_segment)
            expected_signature = hmac.digest(self._secret, payload, "sha256")
            if not hmac.compare_digest(supplied_signature, expected_signature):
                return None
            parsed = json.loads(payload)
            session_id = parsed["sid"]
            expires_at = parsed["exp"]
            if not isinstance(session_id, str) or not isinstance(expires_at, int):
                return None
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
        now = int(self._clock())
        if expires_at <= now:
            self.revoke(token)
            return None
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.expires_at != expires_at:
                return None
            return session

    def rotate_conversation(self, token: str) -> OwnerSession | None:
        current = self.resolve(token)
        if current is None:
            return None
        replacement = OwnerSession(
            session_id=current.session_id,
            conversation_id=secrets.token_urlsafe(24),
            expires_at=current.expires_at,
        )
        with self._lock:
            if current.session_id not in self._sessions:
                return None
            self._sessions[current.session_id] = replacement
        return replacement

    def revoke(self, token: str | None) -> None:
        if token is None:
            return
        try:
            payload = json.loads(_b64decode(token.split(".", 1)[0]))
            session_id = payload.get("sid")
        except (ValueError, TypeError, json.JSONDecodeError):
            return
        if isinstance(session_id, str):
            with self._lock:
                self._sessions.pop(session_id, None)

    def _encode(self, session: OwnerSession) -> str:
        payload = json.dumps(
            {"exp": session.expires_at, "sid": session.session_id},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"{_b64encode(payload)}.{_b64encode(hmac.digest(self._secret, payload, 'sha256'))}"

    def _purge_expired(self, now: int) -> None:
        expired = [key for key, session in self._sessions.items() if session.expires_at <= now]
        for key in expired:
            self._sessions.pop(key, None)
