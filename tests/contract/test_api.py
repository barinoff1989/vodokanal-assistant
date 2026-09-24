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
from app.gateway.llm_gateway import LlmGateway
from app.gateway.pii_filter import PiiSanitizer
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


# --- состав адресов --------------------------------------------- #


def test_только_два_адреса_плюс_стенд():
    """Контракт — два адреса: `/models` удалён из контракта, лишнего быть не должно."""
    app = create_app(FakeGateway())
    api_paths = {r.path for r in app.routes if getattr(r, "path", "").startswith("/v1")}
    assert api_paths == {"/v1/generate", "/v1/healthz"}


def test_проверка_доступности_отвечает_без_авторизации():
    """Контракт: `security: []` — заголовков не требуется."""
    response = _client().get("/v1/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- поток -------------------------------------------------------- #


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


# --- ошибки --------------------------------------------------------- #


def test_неверный_запрос_даёт_400_в_формате_rfc7807():
    response = _client().post("/v1/generate", json={"query": "без системной части"})
    assert response.status_code == 400
    assert response.headers["content-type"].startswith(PROBLEM_JSON)
    body = response.json()
    assert set(body) >= {"type", "title", "status"}


def test_слишком_длинный_запрос_отклоняется():
    """Ограничение длины запроса — 2000 символов."""
    body = {**BODY, "query": "а" * 2001, "parameters": {"stream": False}}
    assert _client().post("/v1/generate", json=body).status_code == 400


@pytest.mark.parametrize("status", [429, 503])
def test_у_отказов_есть_время_повтора(status: int):
    """RFC 7807 требует его для 429; ADR-400 — для 503."""
    problem = ProblemDetail(
        type="https://vodokanal.example/errors/x",
        title="X",
        status=status,
        detail="повторите через 17 с",
        retry_after=17,
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


def test_битое_тело_даёт_400_а_не_500():
    """Найдено живым запросом: неверная кодировка давала 500 с трассировкой.

    Проверки этого не ловили — они шлют заведомо корректные тела. Вина за
    непригодное тело лежит на запросе, и ответ должен это отражать.
    """
    response = _client().post(
        "/v1/generate",
        content=b'{"query": "\xe1\xe8\xf2\xfb\xe9 \xf2\xe5\xea\xf1\xf2"}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


def test_не_json_вовсе_тоже_даёт_400():
    response = _client().post(
        "/v1/generate",
        content="это вообще не json".encode(),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400


def test_время_повтора_берётся_из_поля_а_не_из_текста():
    """Настоящий дефект, найденный разбором кода.

    Прежняя редакция выскабливала первое число из русского пояснения. На `429`
    это работало случайно — там в тексте стоит остаток окна. На `503` цифр в
    пояснении нет вовсе, и заголовок всегда получал зашитую тридцатку, а
    настройка `quota_retry_after_seconds` не влияла ни на что.

    Здесь пояснение содержит **другое** число: если заголовок снова начнут
    собирать из текста, тест это увидит.
    """
    problem = ProblemDetail(
        type="https://vodokanal.example/errors/quota-store-unavailable",
        title="Service temporarily unavailable",
        status=503,
        detail="Сервис временно недоступен. Повторите не ранее чем через 999 с.",
        retry_after=45,
    )
    body = {**BODY, "parameters": {"stream": False}}
    response = _client(FakeGateway(problem=problem)).post("/v1/generate", json=body)

    assert response.headers["Retry-After"] == "45"


async def _never_called(*args: object, **kwargs: object) -> object:
    raise AssertionError("провайдер вызван, хотя хранилище лимитов недоступно")


def test_настройка_времени_повтора_доходит_до_заголовка():
    """Сквозная проверка от настройки до заголовка.

    Без неё величина остаётся написанной и неподключённой — ровно так выглядел
    исходный дефект: `store_retry_after` доходил до `QuotaManager` и там
    кончался.
    """
    from app.gateway.quota import QuotaManager, QuotaStoreUnavailableError

    class BrokenRedis:
        def pipeline(self, *args: object, **kwargs: object) -> object:
            raise QuotaStoreUnavailableError("хранилище недоступно")

        def incr(self, *args: object, **kwargs: object) -> object:
            raise QuotaStoreUnavailableError("хранилище недоступно")

    gateway = LlmGateway(
        quota=QuotaManager(
            BrokenRedis(),
            subscriber_limit=100,
            session_limit=100,
            service_limit=100,
            store_retry_after=77,
        ),
        completion=_never_called,
        sanitizer=PiiSanitizer(analyzer=None),
    )
    body = {**BODY, "parameters": {"stream": False}}
    response = _client(gateway).post("/v1/generate", json=body)

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "77"


# --- маршрут готового бланка (стенд, не контракт /v1) ------------------------- #


def _docs_client(tmp_path: Any) -> tuple[TestClient, Any]:
    from app.documents.artifact import ArtifactStore

    store = ArtifactStore(root=tmp_path, ttl_seconds=3600)
    return TestClient(create_app(FakeGateway(), documents=store)), store


def test_бланк_отдаётся_по_ссылке_html_страницей(tmp_path: Any):
    client, store = _docs_client(tmp_path)
    url = store.put("<!doctype html><p>бланк</p>")
    assert url is not None

    response = client.get(url)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "бланк" in response.text


def test_неизвестный_бланк_это_404_в_формате_rfc7807(tmp_path: Any):
    client, _ = _docs_client(tmp_path)
    response = client.get("/documents/нетакогобланка00")
    assert response.status_code == 404
    assert PROBLEM_JSON in response.headers["content-type"]


def test_маршрут_бланка_не_поднимается_без_хранилища():
    """Часть стенда: нет хранилища — нет и адреса."""
    client = TestClient(create_app(FakeGateway()))
    assert client.get("/documents/whatever000000000").status_code == 404


def test_бланка_нет_в_контракте_v1(tmp_path: Any):
    """Контракт — два адреса: `/documents` — доставка стенда, как `/` и `/static`."""
    client, _ = _docs_client(tmp_path)
    schema = client.get("/openapi.json").json()
    assert not any(path.startswith("/documents") for path in schema["paths"])
