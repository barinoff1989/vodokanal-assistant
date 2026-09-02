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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.gateway.llm_gateway import LlmGateway
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
    """
    headers: dict[str, str] = {}
    if problem.status in (429, 503):
        headers["Retry-After"] = str(_retry_after_from(problem))
    return JSONResponse(
        status_code=problem.status,
        content=problem.model_dump(exclude_none=True),
        media_type=PROBLEM_JSON,
        headers=headers,
    )


def _retry_after_from(problem: ProblemDetail) -> int:
    """Вытащить время повтора из пояснения либо взять безопасное значение."""
    detail = problem.detail or ""
    for word in detail.replace("с;", " ").split():
        if word.isdigit():
            return int(word)
    return 30


def _sse(event_name: str, payload: dict[str, Any]) -> str:
    """Собрать одно событие потока в формате Server-Sent Events."""
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(gateway: LlmGateway | None = None) -> FastAPI:
    """Собрать приложение.

    Шлюз внедряется, чтобы проверки не требовали ни Redis, ни платного внешнего
    API: подменяется он, а не сетевой слой.
    """
    resolved = gateway if gateway is not None else LlmGateway()

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

    @app.get("/v1/healthz")
    async def healthz() -> dict[str, str]:
        """Проверка доступности. Без авторизации — так задано контрактом (7.5)."""
        return {"status": "ok"}

    @app.post("/v1/generate")
    async def generate(request: Request) -> Any:
        """Основной путь абонента. По умолчанию отдаёт поток (правило 4.3)."""
        body = await request.json()
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

    # Демо-стенд. Только на прототипе — см. примечание в начале модуля.
    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/")
        async def stand() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app


async def _event_stream(gateway: LlmGateway, request: GenerateRequest) -> AsyncIterator[str]:
    """Перевести события шлюза в формат Server-Sent Events (раздел 7.3)."""
    async for event in gateway.stream(request):
        if isinstance(event, TokenEvent):
            yield _sse("token", {"delta": event.delta})
        elif isinstance(event, MetadataEvent):
            yield _sse("metadata", event.model_dump(exclude_none=True))
        elif isinstance(event, DoneEvent):
            yield _sse("done", event.model_dump(exclude_none=True))
        elif isinstance(event, ErrorEvent):
            yield _sse("error", event.problem.model_dump(exclude_none=True))
