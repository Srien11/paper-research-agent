"""Environment-backed configuration for the private owner Web interface."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

DEFAULT_ALLOWED_ORIGINS = frozenset(
    {"https://zhimoai.online", "https://www.zhimoai.online"}
)


def _required(source: Mapping[str, str], name: str) -> str:
    value = source.get(name, "").strip()
    if not value:
        raise ValueError(f"missing required Web configuration: {name}")
    return value


def _positive_int(
    source: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = source.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"invalid integer Web configuration: {name}") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"Web configuration out of range: {name}")
    return value


def _strict_bool(source: Mapping[str, str], name: str, default: bool) -> bool:
    raw = source.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"invalid boolean Web configuration: {name}")


def _hex_bytes(source: Mapping[str, str], name: str, *, exact_size: int | None = None) -> bytes:
    raw = _required(source, name)
    try:
        value = bytes.fromhex(raw)
    except ValueError as error:
        raise ValueError(f"invalid hexadecimal Web configuration: {name}") from error
    if exact_size is not None and len(value) != exact_size:
        raise ValueError(f"invalid byte length for Web configuration: {name}")
    return value


@dataclass(frozen=True, slots=True)
class OwnerCredentials:
    """One supported owner credential mode, without exposing it to API models."""

    username: str
    password: str | None = None
    salt: bytes | None = None
    password_hash: bytes | None = None
    pbkdf2_iterations: int = 310_000

    def __post_init__(self) -> None:
        if not self.username or len(self.username) > 128:
            raise ValueError("owner username must contain 1 to 128 characters")
        direct = self.password is not None
        derived = self.salt is not None and self.password_hash is not None
        if direct == derived:
            raise ValueError("configure exactly one owner credential mode")
        if direct and not self.password:
            raise ValueError("owner password cannot be blank")
        if derived:
            if len(self.salt or b"") < 16 or len(self.password_hash or b"") != 32:
                raise ValueError("invalid owner PBKDF2 credential lengths")
            if not 100_000 <= self.pbkdf2_iterations <= 2_000_000:
                raise ValueError("owner PBKDF2 iteration count is out of range")


@dataclass(frozen=True, slots=True)
class WebConfig:
    """Validated security and request limits for the private Web app."""

    credentials: OwnerCredentials
    session_secret: bytes
    allowed_origins: frozenset[str] = DEFAULT_ALLOWED_ORIGINS
    session_ttl_seconds: int = 43_200
    max_question_chars: int = 2_000
    cookie_name: str = "paper_research_owner"
    cookie_path: str = "/paper-research"
    cookie_secure: bool = True

    def __post_init__(self) -> None:
        if len(self.session_secret) < 32:
            raise ValueError("PRA_WEB_SESSION_SECRET must contain at least 32 UTF-8 bytes")
        if not self.allowed_origins or any(
            not self._origin_is_allowed(origin) for origin in self.allowed_origins
        ):
            raise ValueError("allowed origins must be exact HTTPS origins or local test origins")
        if not 300 <= self.session_ttl_seconds <= 604_800:
            raise ValueError("session TTL must be between 5 minutes and 7 days")
        if not 100 <= self.max_question_chars <= 10_000:
            raise ValueError("maximum question length is out of range")
        if not self.cookie_name or not self.cookie_path.startswith("/"):
            raise ValueError("invalid session cookie configuration")

    def _origin_is_allowed(self, origin: str) -> bool:
        parsed = urlsplit(origin)
        exact_origin = bool(
            parsed.hostname
            and not parsed.username
            and not parsed.password
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
        )
        if not exact_origin:
            return False
        if parsed.scheme == "https":
            return True
        return bool(
            not self.cookie_secure
            and parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> WebConfig:
        """Load credentials without reading dotenv files or logging secret values."""
        source = os.environ if environ is None else environ
        direct_password = source.get("PRA_WEB_PASSWORD")
        username = (
            (source.get("PRA_WEB_USER") or "").strip()
            or (source.get("ZHIMO_ADMIN_USER") or "").strip()
            or "owner"
        )
        if direct_password is not None and direct_password.strip():
            credentials = OwnerCredentials(username=username, password=direct_password)
        else:
            credentials = OwnerCredentials(
                username=username,
                salt=_hex_bytes(source, "ZHIMO_ADMIN_SALT"),
                password_hash=_hex_bytes(source, "ZHIMO_ADMIN_HASH", exact_size=32),
                pbkdf2_iterations=_positive_int(
                    source,
                    "ZHIMO_PBKDF2_ITERATIONS",
                    310_000,
                    minimum=100_000,
                    maximum=2_000_000,
                ),
            )
        origins_raw = source.get("PRA_ALLOWED_ORIGINS")
        allowed_origins = (
            frozenset(part.strip() for part in origins_raw.split(",") if part.strip())
            if origins_raw is not None
            else DEFAULT_ALLOWED_ORIGINS
        )
        cookie_secure = _strict_bool(source, "PRA_WEB_COOKIE_SECURE", True)
        return cls(
            credentials=credentials,
            session_secret=_required(source, "PRA_WEB_SESSION_SECRET").encode("utf-8"),
            allowed_origins=allowed_origins,
            session_ttl_seconds=_positive_int(
                source,
                "PRA_WEB_SESSION_TTL_SECONDS",
                43_200,
                minimum=300,
                maximum=604_800,
            ),
            max_question_chars=_positive_int(
                source,
                "PRA_WEB_MAX_QUESTION_CHARS",
                2_000,
                minimum=100,
                maximum=10_000,
            ),
            cookie_secure=cookie_secure,
        )
