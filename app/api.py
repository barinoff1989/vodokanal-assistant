"""HTTP-интерфейс шлюза.

Правило 4.4 разрешает ровно два адреса: `POST /v1/generate` и `GET /v1/healthz`.
Эндпоинт `/models` был удалён из контракта сознательно, и добавлять сюда что-то
сверх этих двух нельзя без обновления контракта.

Демо-стенд (`web/`) раздаётся тем же процессом. Это допустимо только на
прототипе: в рабочем контуре виджет живёт внутри личного кабинета абонента
(раздел 5.2), а стенд существует лишь потому, что доступа к настоящему кабинету
на прототипе не будет — он внутри закрытой сети водоканала.

ОШИБКИ. Тело — RFC 7807 (правило 4.5), тип содержимого `application/problem+json`.
Ошибка внутри уже начатого потока приходит событием `error` с тем же телом:
два формата на два пути пришлось бы поддерживать порознь.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.backend.orchestrator import Orchestrator
from app.documents.artifact import ArtifactStore
from app.gateway.llm_gateway import LlmGateway
from app.metrics import prometheus as metrics
from app.models import (
    DoneEvent,
    ErrorEvent,
    GenerateRequest,
    MetadataEvent,
    ProblemDetail,
    TokenEvent,
)

__all__ = ["create_app"]

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
PROBLEM_JSON = "application/problem+json"


def _problem_response(problem: ProblemDetail) -> JSONResponse:
    """Ответ с телом RFC 7807 и правильным типом содержимого.

    `Retry-After` ставится для 429 и 503: правило 4.5 требует его при отказе по
    лимиту, а ADR-008 — при недоступности хранилища счётчиков. Без заголовка
    виджет не знает, когда повторять, и либо давит запросами, либо сдаётся.

    Значение берётся из поля `retry_after`, а не из текста пояснения. Прежняя
    редакция выскабливала первое число из русской прозы: на `429` это работало
    случайно, а на `503` цифр в пояснении нет вовсе — заголовок всегда получал
    зашитую тридцатку, и настройка `quota_retry_after_seconds` не влияла ни на
    что.
    """
    headers: dict[str, str] = {}
    if problem.status in (429, 503) and problem.retry_after is not None:
        headers["Retry-After"] = str(problem.retry_after)
    return JSONResponse(
        status_code=problem.status,
        content=problem.model_dump(exclude_none=True),
        media_type=PROBLEM_JSON,
        headers=headers,
    )


def _sse(event_name: str, payload: dict[str, Any]) -> str:
    """Собрать одно событие потока в формате Server-Sent Events."""
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(
    orchestrator: Orchestrator | None = None,
    *,
    documents: ArtifactStore | None = None,
) -> FastAPI:
    """Собрать приложение.

    Внедряется **оркестратор**, а не шлюз: по разделу 5.2 выбор пути ответа и
    сборка контекста — дело Backend, и HTTP-слою достаточно знать, что кто-то
    отдаёт ему поток событий. Подменяется он, а не сетевой слой, — иначе
    проверки требовали бы Redis и платного внешнего API.

    :param documents: хранилище готовых бланков. Задано — поднимается маршрут
        `/documents/{token}`, отдающий лист заявления для печати. Это часть
        стенда, как `/` и `/static`, а не контракта `/v1` (правило 4.4).
    """
    resolved = orchestrator if orchestrator is not None else Orchestrator(LlmGateway())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Прогреть тяжёлые зависимости до первого запроса абонента.

        Языковая модель обезличивателя грузится лениво. Без прогрева эти секунды
        платит первый абонент: замер на стенде дал 8,6 с до первого куска ответа
        при цели в 500 мс, хотя сам провайдер отвечал за 220 мс.
        """
        warmup = getattr(resolved, "warmup", None)
        if callable(warmup):
            warmup()
        yield

    app = FastAPI(
        title="AI-помощник абонента водоканала", version="1.0.0", lifespan=lifespan
    )

    @app.middleware("http")
    async def count_responses(request: Request, call_next: Any) -> Any:
        """Считать ответы интерфейса по кодам.

        Метрика была объявлена и не записывалась ничем — виджет доли ошибок в
        дашборде остался бы пустым. Это зеркало той же болезни, что виджет без
        метрики, и заметить её так же трудно.

        Берётся шаблон адреса, а не фактический путь: иначе каждый запрос с
        разными параметрами создавал бы свой временной ряд.
        """
        response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        metrics.record_http(path=path, status=response.status_code)
        return response

    @app.get("/v1/healthz")
    async def healthz() -> dict[str, str]:
        """Проверка доступности. Без авторизации — так задано контрактом (7.5)."""
        return {"status": "ok"}

    @app.post("/v1/generate")
    async def generate(request: Request) -> Any:
        """Основной путь абонента. По умолчанию отдаёт поток (правило 4.3)."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 — важен факт непригодного тела, не причина
            # Битый JSON или неверная кодировка — вина запроса, а не сервиса.
            # Без этой ветки клиент получал 500 с трассировкой: живой запрос в
            # неверной кодировке это и вскрыл, а проверки не поймали — они шлют
            # заведомо корректные тела.
            return _problem_response(
                ProblemDetail(
                    type="https://vodokanal.example/errors/invalid-request",
                    title="Invalid request",
                    status=400,
                    detail="Тело запроса не является корректным JSON в кодировке UTF-8.",
                    instance="/v1/generate",
                )
            )

        try:
            parsed = GenerateRequest.model_validate(body)
        except ValidationError as exc:
            return _problem_response(
                ProblemDetail(
                    type="https://vodokanal.example/errors/invalid-request",
                    title="Invalid request",
                    status=400,
                    detail="; ".join(
                        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                        for err in exc.errors()[:5]
                    ),
                    instance="/v1/generate",
                )
            )

        if not parsed.parameters.stream:
            result = await resolved.generate(parsed)
            if isinstance(result, ProblemDetail):
                return _problem_response(result)
            return JSONResponse(content=result.model_dump(exclude_none=True))

        return StreamingResponse(
            _event_stream(resolved, parsed),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Готовый бланк заявления для печати. Маршрут стенда, не контракта: ссылку
    # выдаёт ответчик образцов в поле `document_url`, содержимое — HTML под A4.
    if documents is not None:

        @app.get("/documents/{token}", include_in_schema=False)
        async def document(token: str) -> Response:
            page = documents.get(token)
            if page is None:
                return _problem_response(
                    ProblemDetail(
                        type="https://vodokanal.example/errors/not-found",
                        title="Not found",
                        status=404,
                        detail="Бланк не найден или срок ссылки истёк.",
                        instance="/documents",
                    )
                )
            return HTMLResponse(content=page)

    # Демо-стенд. Только на прототипе — см. примечание в начале модуля.
    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/")
        async def stand() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app


async def _event_stream(source: Orchestrator, request: GenerateRequest) -> AsyncIterator[str]:
    """Перевести события шлюза в формат Server-Sent Events (раздел 7.3)."""
    async for event in source.stream(request):
        if isinstance(event, TokenEvent):
            yield _sse("token", {"delta": event.delta})
        elif isinstance(event, MetadataEvent):
            yield _sse("metadata", event.model_dump(exclude_none=True))
        elif isinstance(event, DoneEvent):
            yield _sse("done", event.model_dump(exclude_none=True))
        elif isinstance(event, ErrorEvent):
            yield _sse("error", event.problem.model_dump(exclude_none=True))
