"""FastAPI application factory for the authenticated paper-research interface."""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol, cast

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from paper_research_agent.web.auth import CredentialVerifier, OwnerSession, SessionManager
from paper_research_agent.web.config import WebConfig
from paper_research_agent.web.models import (
    AnonymousSessionResponse,
    AskResponse,
    HealthResponse,
    LoginRequest,
    OperationResponse,
    QuestionRequest,
    RecommendedQuestion,
    SessionResponse,
)

APP_PREFIX = "/paper-research"
API_PREFIX = f"{APP_PREFIX}/api"


class WebRuntime(Protocol):
    is_ready: bool
    is_busy: bool

    async def ask(self, question: str, *, session_id: str) -> object: ...

    async def clear_conversation(self, session_id: str) -> int: ...

    async def aclose(self) -> None: ...


RuntimeFactory = Callable[[], WebRuntime | Awaitable[WebRuntime]]


def _load_recommended_questions() -> tuple[RecommendedQuestion, ...]:
    path = Path(__file__).resolve().parents[3] / "configs" / "web" / "recommended-questions-v1.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        questions = raw["questions"]
        if not isinstance(questions, list):
            return ()
        return tuple(
            RecommendedQuestion.model_validate(
                {
                    "category": item["category"],
                    "title": item["title"],
                    "prompt": item["prompt"],
                }
            )
            for item in questions
            if isinstance(item, dict)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return ()


async def _default_runtime_factory() -> WebRuntime:
    """Import the local-model runtime only when application startup begins."""
    from paper_research_agent.web.runtime import RAGRuntime

    factory_name = (
        "from_environment_with_agent"
        if RAGRuntime.research_agent_enabled_from_environment()
        else "from_environment"
    )
    factory = getattr(RAGRuntime, factory_name, None)
    if factory is None:
        raise RuntimeError(f"RAGRuntime.{factory_name} is unavailable")
    runtime = await _maybe_await(factory())
    return cast(WebRuntime, runtime)


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _runtime_error_response(error: Exception) -> HTTPException:
    """Map runtime boundaries without returning provider or filesystem details."""
    name = type(error).__name__
    if name == "RuntimeBusyError":
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail="系统正在处理上一条问题")
    if name == "RuntimeClosedError":
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="问答服务暂未就绪",
        )
    if isinstance(error, ValueError):
        return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="问题无效")
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="问答服务暂时不可用",
    )


def create_app(
    *,
    config: WebConfig | None = None,
    runtime: WebRuntime | None = None,
    runtime_factory: RuntimeFactory | None = None,
    serve_static: bool = True,
    recommended_questions: tuple[RecommendedQuestion, ...] | None = None,
) -> FastAPI:
    """Create an isolated app; runtime injection keeps API tests free of local ML loading."""
    settings = config or WebConfig.from_env()
    sessions = SessionManager(settings.session_secret, settings.session_ttl_seconds)
    credentials = CredentialVerifier(settings.credentials)
    safe_questions = (
        _load_recommended_questions()
        if recommended_questions is None
        else recommended_questions
    )
    owns_runtime = runtime is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.runtime is None:
            factory = runtime_factory or _default_runtime_factory
            app.state.runtime = await _maybe_await(factory())
        try:
            yield
        finally:
            active_runtime = cast(WebRuntime | None, app.state.runtime)
            if owns_runtime and active_runtime is not None:
                await active_runtime.aclose()

    app = FastAPI(
        title="Paper Research Agent",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    app.state.config = settings
    app.state.sessions = sessions

    @app.middleware("http")
    async def private_response_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'self'; form-action 'self'; "
            "frame-ancestors 'none'; object-src 'none'; img-src 'self' data:; "
            "style-src 'self'; script-src 'self'; connect-src 'self'"
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        _request: Request,
        _error: RequestValidationError,
    ) -> JSONResponse:
        # FastAPI's default response includes rejected input, which could echo a password.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": "请求格式无效"},
        )

    def require_origin(request: Request) -> None:
        if request.headers.get("origin") not in settings.allowed_origins:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="请求来源不允许")

    def current_session(request: Request) -> OwnerSession:
        session = sessions.resolve(request.cookies.get(settings.cookie_name))
        if session is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="需要站长登录")
        return session

    def active_runtime(request: Request) -> WebRuntime:
        active = cast(WebRuntime | None, request.app.state.runtime)
        if active is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="问答服务暂未就绪",
            )
        return active

    @app.get(f"{APP_PREFIX}/healthz", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get(f"{APP_PREFIX}/readyz", response_model=HealthResponse)
    async def ready(request: Request) -> Response | HealthResponse:
        active = cast(WebRuntime | None, request.app.state.runtime)
        if active is None or not active.is_ready:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return HealthResponse(status="ready")

    @app.get(
        f"{API_PREFIX}/session",
        response_model=SessionResponse | AnonymousSessionResponse,
    )
    async def session_state(request: Request) -> SessionResponse | AnonymousSessionResponse:
        session = sessions.resolve(request.cookies.get(settings.cookie_name))
        if session is None:
            return AnonymousSessionResponse()
        return SessionResponse(
            conversation_id=session.conversation_id,
            expires_at=session.expires_at,
        )

    @app.post(f"{API_PREFIX}/login", response_model=SessionResponse)
    async def login(
        payload: LoginRequest,
        response: Response,
        _origin: None = Depends(require_origin),
    ) -> SessionResponse:
        if not credentials.verify(payload.username, payload.password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="认证失败")
        token, session = sessions.create()
        response.set_cookie(
            key=settings.cookie_name,
            value=token,
            max_age=settings.session_ttl_seconds,
            httponly=True,
            secure=settings.cookie_secure,
            samesite="strict",
            path=settings.cookie_path,
        )
        return SessionResponse(
            conversation_id=session.conversation_id,
            expires_at=session.expires_at,
        )

    @app.get(
        f"{API_PREFIX}/recommended-questions",
        response_model=tuple[RecommendedQuestion, ...],
    )
    async def get_recommended_questions(
        _session: OwnerSession = Depends(current_session),  # noqa: B008
    ) -> tuple[RecommendedQuestion, ...]:
        return safe_questions

    @app.post(f"{API_PREFIX}/logout", response_model=OperationResponse)
    async def logout(
        request: Request,
        response: Response,
        _origin: None = Depends(require_origin),
        _session: OwnerSession = Depends(current_session),  # noqa: B008
    ) -> OperationResponse:
        token = request.cookies.get(settings.cookie_name)
        sessions.revoke(token)
        response.delete_cookie(
            key=settings.cookie_name,
            path=settings.cookie_path,
            secure=settings.cookie_secure,
            httponly=True,
            samesite="strict",
        )
        return OperationResponse()

    @app.delete(f"{API_PREFIX}/conversation", response_model=SessionResponse)
    async def new_conversation(
        request: Request,
        _origin: None = Depends(require_origin),
        session: OwnerSession = Depends(current_session),  # noqa: B008
        rag_runtime: WebRuntime = Depends(active_runtime),  # noqa: B008
    ) -> SessionResponse:
        try:
            await rag_runtime.clear_conversation(session.conversation_id)
        except Exception as error:  # noqa: BLE001 - runtime is an isolation boundary
            raise _runtime_error_response(error) from None
        token = request.cookies.get(settings.cookie_name)
        replacement = sessions.rotate_conversation(token or "")
        if replacement is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="需要站长登录")
        return SessionResponse(
            conversation_id=replacement.conversation_id,
            expires_at=replacement.expires_at,
        )

    @app.post(f"{API_PREFIX}/ask", response_model=AskResponse)
    async def ask(
        payload: QuestionRequest,
        _origin: None = Depends(require_origin),
        session: OwnerSession = Depends(current_session),  # noqa: B008
        rag_runtime: WebRuntime = Depends(active_runtime),  # noqa: B008
    ) -> AskResponse:
        if len(payload.question) > settings.max_question_chars:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="问题长度超过限制",
            )
        try:
            result = await rag_runtime.ask(payload.question, session_id=session.conversation_id)
            return AskResponse.model_validate(result, from_attributes=True)
        except HTTPException:
            raise
        except Exception as error:  # noqa: BLE001 - provider details must not cross this boundary
            raise _runtime_error_response(error) from None

    static_directory = Path(__file__).with_name("static")
    if serve_static and static_directory.is_dir():
        app.mount(APP_PREFIX, StaticFiles(directory=static_directory, html=True), name="web-ui")

    return app
