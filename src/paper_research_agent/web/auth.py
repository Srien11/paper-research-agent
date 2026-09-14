"""Owner credential verification and revocable signed Web sessions."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

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


class SQLiteSessionRevocationStore:
    """Persist only revoked session identifiers until their signed expiry."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with closing(self._connect()) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS revoked_web_sessions (
                    session_id TEXT PRIMARY KEY,
                    expires_at INTEGER NOT NULL
                )"""
            )
            connection.commit()

    def contains(self, session_id: str, *, now: int) -> bool:
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                "DELETE FROM revoked_web_sessions WHERE expires_at <= ?", (now,)
            )
            row = connection.execute(
                "SELECT 1 FROM revoked_web_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            connection.commit()
        return row is not None

    def revoke(self, session_id: str, *, expires_at: int) -> None:
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO revoked_web_sessions(session_id, expires_at) VALUES (?, ?)",
                (session_id, expires_at),
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection


class SessionManager:
    """Keep revocable owner sessions in-process and authenticate cookies with HMAC-SHA256."""

    def __init__(
        self,
        secret: bytes,
        ttl_seconds: int,
        *,
        clock: Callable[[], float] = time.time,
        revocation_store: SQLiteSessionRevocationStore | None = None,
    ):
        if len(secret) < 32:
            raise ValueError("session secret must contain at least 32 bytes")
        if ttl_seconds <= 0:
            raise ValueError("session TTL must be positive")
        self._secret = secret
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._revocation_store = revocation_store
        self._sessions: dict[str, OwnerSession] = {}
        self._revoked: set[str] = set()
        self._lock = threading.Lock()

    def create(self) -> tuple[str, OwnerSession]:
        now = int(self._clock())
        session = OwnerSession(
            session_id=secrets.token_urlsafe(32),
            conversation_id=secrets.token_hex(24),
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
            conversation_id = parsed["cid"]
            expires_at = parsed["exp"]
            if (
                not isinstance(session_id, str)
                or not isinstance(conversation_id, str)
                or not isinstance(expires_at, int)
            ):
                return None
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
        now = int(self._clock())
        if expires_at <= now:
            self.revoke(token)
            return None
        with self._lock:
            if session_id in self._revoked or (
                self._revocation_store is not None
                and self._revocation_store.contains(session_id, now=now)
            ):
                return None
            session = self._sessions.get(session_id)
            if session is None:
                session = OwnerSession(
                    session_id=session_id,
                    conversation_id=conversation_id,
                    expires_at=expires_at,
                )
                self._sessions[session_id] = session
            if (
                session.expires_at != expires_at
                or session.conversation_id != conversation_id
            ):
                return None
            return session

    def rotate_conversation(self, token: str) -> tuple[str, OwnerSession] | None:
        current = self.resolve(token)
        if current is None:
            return None
        replacement = OwnerSession(
            session_id=current.session_id,
            conversation_id=secrets.token_hex(24),
            expires_at=current.expires_at,
        )
        with self._lock:
            if current.session_id not in self._sessions:
                return None
            self._sessions[current.session_id] = replacement
        return self._encode(replacement), replacement

    def select_conversation(
        self, token: str, conversation_id: str
    ) -> tuple[str, OwnerSession] | None:
        current = self.resolve(token)
        normalized = conversation_id.strip()
        if current is None or not normalized or len(normalized) > 256:
            return None
        replacement = OwnerSession(
            session_id=current.session_id,
            conversation_id=normalized,
            expires_at=current.expires_at,
        )
        with self._lock:
            if current.session_id not in self._sessions:
                return None
            self._sessions[current.session_id] = replacement
        return self._encode(replacement), replacement

    def revoke(self, token: str | None) -> None:
        if token is None:
            return
        try:
            payload = json.loads(_b64decode(token.split(".", 1)[0]))
            session_id = payload.get("sid")
            expires_at = payload.get("exp")
        except (ValueError, TypeError, json.JSONDecodeError):
            return
        if isinstance(session_id, str) and isinstance(expires_at, int):
            with self._lock:
                self._sessions.pop(session_id, None)
                self._revoked.add(session_id)
                if self._revocation_store is not None and expires_at > int(self._clock()):
                    self._revocation_store.revoke(
                        session_id,
                        expires_at=expires_at,
                    )

    def _encode(self, session: OwnerSession) -> str:
        payload = json.dumps(
            {
                "cid": session.conversation_id,
                "exp": session.expires_at,
                "sid": session.session_id,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"{_b64encode(payload)}.{_b64encode(hmac.digest(self._secret, payload, 'sha256'))}"

    def _purge_expired(self, now: int) -> None:
        expired = [key for key, session in self._sessions.items() if session.expires_at <= now]
        for key in expired:
            self._sessions.pop(key, None)
