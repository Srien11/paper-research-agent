"""FastAPI application factory for the authenticated paper-research interface."""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol, cast

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from paper_research_agent.web.auth import CredentialVerifier, OwnerSession, SessionManager
from paper_research_agent.web.chat_runtime import RouteOutputError
from paper_research_agent.web.config import WebConfig
from paper_research_agent.web.files import AttachmentStore
from paper_research_agent.web.models import (
    AnonymousSessionResponse,
    AskResponse,
    AttachmentResponse,
    HealthResponse,
    LoginRequest,
    LongTermMemoryListResponse,
    OperationResponse,
    QuestionRequest,
    RecommendedQuestion,
    SafeLongTermMemory,
    SafePendingToolApproval,
    SafeToolObservation,
    SessionResponse,
    ToolApprovalRequest,
    ToolResearchResponse,
)
from paper_research_agent.web.routing import (
    ROUTE_LABELS,
    RouteContext,
    RouteDecision,
    enforce_route_policy,
)

APP_PREFIX = "/paper-research"
API_PREFIX = f"{APP_PREFIX}/api"


class WebRuntime(Protocol):
    is_ready: bool
    is_busy: bool

    async def ask(self, question: str, *, session_id: str) -> object: ...

    async def run_tool_research(self, question: str, *, session_id: str) -> object: ...

    def stream_chat(
        self, question: str, *, session_id: str
    ) -> AsyncIterator[dict[str, object]]: ...

    async def resume_tool_research(self, *, session_id: str, approved: bool) -> object: ...

    async def list_long_term_memories(self, *, limit: int = 20) -> object: ...

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
    if not os.getenv("PRA_CORPUS_DIR", "").strip():
        from paper_research_agent.web.chat_runtime import ConversationRuntime

        return ConversationRuntime.from_environment()

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
    if name == "RAGUnavailableError":
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="本地论文库尚未配置，请关闭‘仅使用本地论文库（RAG）’后继续交流",
        )
    if isinstance(error, ValueError):
        return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="问题无效")
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="问答服务暂时不可用",
    )


def _value(source: object, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _safe_tool_research_response(result: object) -> ToolResearchResponse:
    observations: list[SafeToolObservation] = []
    for observation in _value(result, "observations", ()):
        execution = _value(observation, "result")
        observations.append(
            SafeToolObservation(
                sequence=_value(observation, "sequence"),
                tool_name=_value(observation, "tool_name"),
                purpose=_value(observation, "purpose"),
                status=_value(execution, "status"),
                trust=_value(execution, "trust"),
                item_count=len(_value(execution, "items", ())),
            )
        )
    raw_pending = _value(result, "pending_approval")
    pending = None
    if raw_pending is not None:
        pending = SafePendingToolApproval(
            tool_name=_value(raw_pending, "tool_name"),
            purpose=_value(raw_pending, "purpose"),
            arguments_sha256=_value(raw_pending, "arguments_sha256"),
            expires_at_epoch=_value(raw_pending, "expires_at_epoch"),
        )
    return ToolResearchResponse(
        run_id=_value(result, "run_id"),
        status=_value(result, "status"),
        observations=tuple(observations),
        final_summary=_value(result, "final_summary"),
        termination_reason=_value(result, "termination_reason"),
        pending_approval=pending,
    )


def _safe_long_term_memory_response(result: object) -> LongTermMemoryListResponse:
    memories = tuple(
        SafeLongTermMemory(
            memory_id=_value(item, "memory_id"),
            kind=_value(item, "kind"),
            content=_value(item, "content"),
            source_chunk_ids=tuple(_value(item, "source_chunk_ids", ())),
            version=_value(item, "version"),
            created_at=_value(item, "created_at"),
            updated_at=_value(item, "updated_at"),
            expires_at=_value(item, "expires_at"),
            supersedes_memory_id=_value(item, "supersedes_memory_id"),
        )
        for item in _value(result, "items", ())
    )
    return LongTermMemoryListResponse(memories=memories)


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
        _load_recommended_questions() if recommended_questions is None else recommended_questions
    )
    owns_runtime = runtime is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.runtime is None:
            factory = runtime_factory or _default_runtime_factory
            app.state.runtime = await _maybe_await(factory())
        active = cast(WebRuntime | None, app.state.runtime)
        if active is not None and hasattr(active, "stream_chat"):
            app.state.chat_runtime = active
        elif owns_runtime:
            from paper_research_agent.web.chat_runtime import ConversationRuntime

            app.state.chat_runtime = ConversationRuntime.from_environment()
        try:
            yield
        finally:
            active_runtime = cast(WebRuntime | None, app.state.runtime)
            chat_runtime = cast(WebRuntime | None, app.state.chat_runtime)
            if owns_runtime and chat_runtime is not None and chat_runtime is not active_runtime:
                await chat_runtime.aclose()
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
    app.state.chat_runtime = runtime if runtime is not None and hasattr(runtime, "stream_chat") else None
    app.state.config = settings
    app.state.sessions = sessions
    app.state.attachments = AttachmentStore(Path(__file__).resolve().parents[3] / "data/runtime/uploads")

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

    def active_chat_runtime(request: Request) -> WebRuntime:
        active = cast(WebRuntime | None, request.app.state.chat_runtime)
        if active is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="普通交流模型暂未就绪",
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

    @app.get(f"{API_PREFIX}/memories", response_model=LongTermMemoryListResponse)
    async def list_long_term_memories(
        _session: OwnerSession = Depends(current_session),  # noqa: B008
        rag_runtime: WebRuntime = Depends(active_runtime),  # noqa: B008
    ) -> LongTermMemoryListResponse:
        try:
            result = await rag_runtime.list_long_term_memories(limit=20)
            return _safe_long_term_memory_response(result)
        except HTTPException:
            raise
        except Exception as error:  # noqa: BLE001
            raise _runtime_error_response(error) from None

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

    @app.post(f"{API_PREFIX}/tools/run", response_model=ToolResearchResponse)
    async def run_tool_research(
        payload: QuestionRequest,
        _origin: None = Depends(require_origin),
        session: OwnerSession = Depends(current_session),  # noqa: B008
        rag_runtime: WebRuntime = Depends(active_runtime),  # noqa: B008
    ) -> ToolResearchResponse:
        if len(payload.question) > settings.max_question_chars:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="问题长度超过限制",
            )
        try:
            result = await rag_runtime.run_tool_research(
                payload.question,
                session_id=session.conversation_id,
            )
            return _safe_tool_research_response(result)
        except HTTPException:
            raise
        except Exception as error:  # noqa: BLE001
            raise _runtime_error_response(error) from None

    @app.post(f"{API_PREFIX}/chat/stream")
    async def stream_chat(
        request: Request,
        payload: QuestionRequest,
        _origin: None = Depends(require_origin),
        session: OwnerSession = Depends(current_session),  # noqa: B008
        chat_runtime: WebRuntime = Depends(active_chat_runtime),  # noqa: B008
    ) -> StreamingResponse:
        if len(payload.question) > settings.max_question_chars:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="问题长度超过限制",
            )
        stream_method = getattr(chat_runtime, "stream_chat", None)
        if stream_method is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="当前运行模式暂不支持流式普通交流",
            )

        async def events() -> AsyncIterator[bytes]:
            try:
                rag_runtime = cast(WebRuntime | None, request.app.state.runtime)
                classifier = getattr(chat_runtime, "classify_route", None)
                if classifier is None:
                    raw_decision = RouteDecision(
                        route="attachment_qa" if payload.attachment_ids else "normal_chat",
                        confidence=0,
                        reason="模型路由器不可用，采用最小权限默认路由",
                    )
                else:
                    try:
                        raw_decision = await classifier(
                            payload.question,
                            has_attachments=bool(payload.attachment_ids),
                            rag_mode=payload.rag_mode,
                        )
                    except RouteOutputError:
                        raw_decision = RouteDecision(
                            route=(
                                "attachment_qa"
                                if payload.attachment_ids
                                else "normal_chat"
                            ),
                            confidence=0,
                            reason="模型路由结果无效，采用最小权限默认路由",
                        )
                decision = enforce_route_policy(
                    raw_decision,
                    RouteContext(
                        has_attachments=bool(payload.attachment_ids),
                        rag_mode=payload.rag_mode,
                        rag_available=bool(rag_runtime and getattr(rag_runtime, "rag_available", True)),
                        web_available=bool(
                            rag_runtime and getattr(rag_runtime, "agent_available", False)
                        ),
                    ),
                )
                yield (
                    json.dumps(
                        {
                            "type": "route",
                            "route": decision.route,
                            "label": ROUTE_LABELS[decision.route],
                            "reason": decision.reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                ).encode()

                if decision.route == "local_rag":
                    if rag_runtime is None:
                        raise RuntimeError("local RAG is unavailable")
                    result = await rag_runtime.ask(
                        payload.question, session_id=session.conversation_id
                    )
                    safe_result = AskResponse.model_validate(result, from_attributes=True)
                    yield (
                        json.dumps(
                            {"type": "rag_result", "payload": safe_result.model_dump(mode="json")},
                            ensure_ascii=False,
                        )
                        + "\n"
                    ).encode()
                    return

                if decision.route == "web_research":
                    if rag_runtime is None:
                        raise RuntimeError("research runtime is unavailable")
                    result = await rag_runtime.run_tool_research(
                        payload.question, session_id=session.conversation_id
                    )
                    safe_result = _safe_tool_research_response(result)
                    if safe_result.final_summary:
                        yield (
                            json.dumps(
                                {"type": "delta", "text": safe_result.final_summary},
                                ensure_ascii=False,
                            )
                            + "\n"
                        ).encode()
                    if safe_result.pending_approval is not None:
                        yield (
                            json.dumps(
                                {
                                    "type": "approval_required",
                                    "payload": safe_result.model_dump(mode="json"),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        ).encode()
                    yield (json.dumps({"type": "done", "metrics": None}) + "\n").encode()
                    return

                if decision.route in {"attachment_qa", "file_edit"}:
                    attachment_texts = request.app.state.attachments.extract(
                        session.conversation_id, payload.attachment_ids
                    )
                    method_name = (
                        "stream_file_edit"
                        if decision.route == "file_edit"
                        else "stream_attachment_chat"
                    )
                    file_method = getattr(chat_runtime, method_name, None)
                    if file_method is None:
                        raise RuntimeError("当前模型暂不支持附件处理")
                    source = file_method(
                        payload.question,
                        attachment_texts=attachment_texts,
                        session_id=session.conversation_id,
                    )
                else:
                    source = stream_method(payload.question, session_id=session.conversation_id)
                async for event in source:
                    yield (json.dumps(event, ensure_ascii=False) + "\n").encode()
            except Exception as error:  # noqa: BLE001
                message = _runtime_error_response(error).detail
                yield (json.dumps({"type": "error", "message": message}, ensure_ascii=False) + "\n").encode()

        return StreamingResponse(
            events(),
            media_type="application/x-ndjson",
            headers={"X-Accel-Buffering": "no"},
        )

    @app.post(f"{API_PREFIX}/files", response_model=AttachmentResponse)
    async def upload_file(
        request: Request,
        filename: str,
        _origin: None = Depends(require_origin),
        session: OwnerSession = Depends(current_session),  # noqa: B008
    ) -> AttachmentResponse:
        try:
            attachment = await request.app.state.attachments.save(
                session_id=session.conversation_id,
                filename=filename,
                content_type=request.headers.get("content-type", "application/octet-stream"),
                chunks=request.stream(),
            )
            return AttachmentResponse.model_validate(attachment, from_attributes=True)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None

    @app.delete(f"{API_PREFIX}/files/{{attachment_id}}", response_model=OperationResponse)
    async def delete_file(
        attachment_id: str,
        _origin: None = Depends(require_origin),
        session: OwnerSession = Depends(current_session),  # noqa: B008
    ) -> OperationResponse:
        app.state.attachments.delete(session.conversation_id, attachment_id)
        return OperationResponse()

    @app.post(f"{API_PREFIX}/tools/approval", response_model=ToolResearchResponse)
    async def resolve_tool_approval(
        payload: ToolApprovalRequest,
        _origin: None = Depends(require_origin),
        session: OwnerSession = Depends(current_session),  # noqa: B008
        rag_runtime: WebRuntime = Depends(active_runtime),  # noqa: B008
    ) -> ToolResearchResponse:
        try:
            result = await rag_runtime.resume_tool_research(
                session_id=session.conversation_id,
                approved=payload.approved,
            )
            return _safe_tool_research_response(result)
        except HTTPException:
            raise
        except Exception as error:  # noqa: BLE001
            raise _runtime_error_response(error) from None

    static_directory = Path(__file__).with_name("static")
    if serve_static and static_directory.is_dir():
        app.mount(APP_PREFIX, StaticFiles(directory=static_directory, html=True), name="web-ui")

    return app
