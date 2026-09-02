"""Проверки HTTP-интерфейса.

Проверяется контракт: какие адреса есть, какой формат у ошибок, приходит ли
ответ потоком. Шлюз подменяется — иначе каждый прогон стоил бы обращений к
платному внешнему API.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import PROBLEM_JSON, create_app
from app.models import (
    DoneEvent,
    ErrorEvent,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    MetadataEvent,
    ProblemDetail,
    SourceRef,
    TokenEvent,
    Usage,
)


class FakeGateway:
    """Шлюз-заглушка: отдаёт заданные события либо заданную ошибку."""

    def __init__(self, problem: ProblemDetail | None = None) -> None:
        self.problem = problem
        self.seen: list[GenerateRequest] = []

    async def generate(self, request: GenerateRequest) -> Any:
        self.seen.append(request)
        if self.problem is not None:
            return self.problem
        return GenerateResponse(answer="Раз в шесть лет.", model="m", trace_id="trace-1")

    async def stream(self, request: GenerateRequest) -> AsyncIterator[Any]:
        self.seen.append(request)
        if self.problem is not None:
            yield ErrorEvent(problem=self.problem)
            return
        yield TokenEvent(delta="Поверку ")
        yield TokenEvent(delta="раз в шесть лет.")
        yield MetadataEvent(
            sources=[
                SourceRef(chunk_id="c1", source_title="Регламент поверки", relevance_score=0.9)
            ]
        )
        yield DoneEvent(finish_reason=FinishReason.STOP, usage=Usage(), trace_id="trace-1")


def _client(gateway: Any = None) -> TestClient:
    return TestClient(create_app(gateway if gateway is not None else FakeGateway()))


BODY = {
    "system": "Ты помощник абонента водоканала.",
    "query": "Как часто поверять счётчик?",
    "context": [],
    "parameters": {"stream": True},
    "metadata": {"subscriber_id": "sub-1", "session_id": "sess-1", "channel": "lk_web"},
}


# --- состав адресов (правило 4.4) --------------------------------------------- #


def test_только_два_адреса_плюс_стенд():
    """Правило 4.4: `/models` удалён из контракта, лишнего быть не должно."""
    app = create_app(FakeGateway())
    api_paths = {r.path for r in app.routes if getattr(r, "path", "").startswith("/v1")}
    assert api_paths == {"/v1/generate", "/v1/healthz"}


def test_проверка_доступности_отвечает_без_авторизации():
    """Раздел 7.5: `security: []` — заголовков не требуется."""
    response = _client().get("/v1/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- поток (правило 4.3) -------------------------------------------------------- #


def test_ответ_приходит_потоком():
    with _client() as client, client.stream("POST", "/v1/generate", json=BODY) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        text = "".join(response.iter_text())

    assert "event: token" in text
    assert "event: metadata" in text
    assert "event: done" in text


def test_события_потока_разделены_пустой_строкой():
    """Требование формата: иначе потребитель склеит два события в одно."""
    with _client() as client, client.stream("POST", "/v1/generate", json=BODY) as response:
        text = "".join(response.iter_text())
    assert "\n\n" in text
    assert text.endswith("\n\n")


def test_русский_текст_в_потоке_не_экранируется():
    """Иначе стенд показал бы абоненту последовательности вида \\u043f."""
    with _client() as client, client.stream("POST", "/v1/generate", json=BODY) as response:
        text = "".join(response.iter_text())
    assert "Поверку" in text
    assert "\\u04" not in text


def test_ответ_целиком_приходит_обычным_json():
    body = {**BODY, "parameters": {"stream": False}}
    response = _client().post("/v1/generate", json=body)
    assert response.status_code == 200
    assert response.json()["answer"] == "Раз в шесть лет."


# --- ошибки (правило 4.5) --------------------------------------------------------- #


def test_неверный_запрос_даёт_400_в_формате_rfc7807():
    response = _client().post("/v1/generate", json={"query": "без системной части"})
    assert response.status_code == 400
    assert response.headers["content-type"].startswith(PROBLEM_JSON)
    body = response.json()
    assert set(body) >= {"type", "title", "status"}


def test_слишком_длинный_запрос_отклоняется():
    """Ограничение раздела 7.2 — 2000 символов."""
    body = {**BODY, "query": "а" * 2001, "parameters": {"stream": False}}
    assert _client().post("/v1/generate", json=body).status_code == 400


@pytest.mark.parametrize("status", [429, 503])
def test_у_отказов_есть_время_повтора(status: int):
    """Правило 4.5 требует его для 429; ADR-008 — для 503."""
    problem = ProblemDetail(
        type="https://vodokanal.example/errors/x",
        title="X",
        status=status,
        detail="повторите через 17 с",
    )
    body = {**BODY, "parameters": {"stream": False}}
    response = _client(FakeGateway(problem=problem)).post("/v1/generate", json=body)
    assert response.status_code == status
    assert response.headers["Retry-After"] == "17"


def test_ошибка_в_потоке_приходит_событием_с_тем_же_телом():
    """Решение 4 в моделях: один формат ошибки на оба пути."""
    problem = ProblemDetail(
        type="https://vodokanal.example/errors/upstream-unavailable",
        title="LLM provider unavailable",
        status=502,
    )
    gateway = FakeGateway(problem=problem)
    with _client(gateway) as client, client.stream("POST", "/v1/generate", json=BODY) as response:
        # Поток уже начался, поэтому код ответа 200 — ошибка внутри него.
        assert response.status_code == 200
        text = "".join(response.iter_text())
    assert "event: error" in text
    assert '"status": 502' in text


# --- передача запроса шлюзу ---------------------------------------------------------- #


def test_идентификаторы_доходят_до_шлюза():
    """Без них не считаются лимиты и не пишется аудит."""
    gateway = FakeGateway()
    body = {**BODY, "parameters": {"stream": False}}
    _client(gateway).post("/v1/generate", json=body)
    assert gateway.seen[0].metadata.subscriber_id == "sub-1"
    assert gateway.seen[0].metadata.session_id == "sess-1"


# --- демо-стенд ------------------------------------------------------------------------ #


def test_стенд_отдаётся_по_корневому_адресу():
    """Только на прототипе: в рабочем контуре виджет живёт в кабинете абонента."""
    response = _client().get("/")
    assert response.status_code == 200
    assert "AI-помощник" in response.text


def test_стенд_помечен_как_ненастоящий():
    """Иначе на скриншоте он неотличим от рабочей системы."""
    assert "демо-стенд" in _client().get("/").text
