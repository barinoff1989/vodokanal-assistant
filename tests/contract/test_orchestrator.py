"""Проверки Backend: выбор пути ответа, поиск контекста, лимиты.

По разделу 5.2 Backend отвечает за оркестрацию и сборку промпта, а шлюз — за
модель, лимиты и защиту. Эти проверки сторожат границу: предметные пути не
должны сползти обратно в шлюз, а поиск — оказаться там, где его не ждут.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from app.backend.orchestrator import DirectAnswer, Orchestrator
from app.config import Settings
from app.gateway.guardrails import SyncGuardrails
from app.gateway.llm_gateway import LlmGateway
from app.gateway.pii_filter import PiiSanitizer
from app.gateway.quota import QuotaManager, QuotaStoreUnavailableError
from app.models import (
    ContextChunk,
    DoneEvent,
    ErrorEvent,
    GenerateRequest,
    GenerateResponse,
    GenerationParameters,
    MetadataEvent,
    ProblemDetail,
    RequestMetadata,
    TokenEvent,
)


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    def incr(self, key: str) -> int:
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    def expire(self, key: str, seconds: int) -> None:
        return None


class BrokenRedis:
    def incr(self, key: str) -> int:
        raise QuotaStoreUnavailableError("хранилище недоступно")

    def expire(self, key: str, seconds: int) -> None:
        raise QuotaStoreUnavailableError("хранилище недоступно")


class AlwaysAnswers:
    """Отвечает на всё. Backend всё равно, какой ответчик подставлен."""

    def __init__(self, text: str = "готовый ответ", disclaimer: str | None = "оговорка"):
        self._answer = DirectAnswer(text=text, disclaimer=disclaimer)

    def answer(self, request: GenerateRequest, *, now: datetime) -> DirectAnswer | None:  # noqa: ARG002
        return self._answer


class NeverAnswers:
    def answer(self, request: GenerateRequest, *, now: datetime) -> DirectAnswer | None:  # noqa: ARG002
        return None


class FakeKnowledgeBase:
    """Отдаёт заранее заданные фрагменты и запоминает, о чём спрашивали."""

    def __init__(self, chunks: list[ContextChunk] | None = None) -> None:
        self.queries: list[str] = []
        self._chunks = chunks or [
            ContextChunk(
                chunk_id="faq-01",
                text="Показания передаются до 25 числа.",
                source_title="FAQ",
                source_url="https://example.test/faq/",
                relevance_score=0.9,
            )
        ]

    def search(self, query: str, **kwargs: Any) -> list[ContextChunk]:
        self.queries.append(query)
        return list(self._chunks)


async def _explode(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("провайдер вызван, хотя ответ был готов без него")


def _completion(text: str = "ответ модели"):
    async def call(*args: Any, **kwargs: Any) -> Any:
        class _M:
            content = text

        class _C:
            message = _M()
            delta = _M()
            finish_reason = "stop"

        class _R:
            choices = [_C()]
            usage = None
            model = "test"

        return _R()

    return call


def _gateway(**overrides: Any) -> LlmGateway:
    sanitizer = PiiSanitizer(analyzer=None)
    defaults: dict[str, Any] = {
        "sanitizer": sanitizer,
        "guardrails": SyncGuardrails(sanitizer=sanitizer),
        "completion": _completion(),
        "settings": Settings(_env_file=None, llm_provider="local-test"),
        "quota": None,
    }
    return LlmGateway(**{**defaults, **overrides})


def _request(**overrides: Any) -> GenerateRequest:
    defaults: dict[str, Any] = {
        "system": "Ты помощник абонента водоканала.",
        "query": "Когда передавать показания?",
        "parameters": GenerationParameters(stream=False),
        "metadata": RequestMetadata(subscriber_id="sub-1", session_id="sess-1"),
    }
    return GenerateRequest(**{**defaults, **overrides})


# --- прямой ответ мимо модели -------------------------------------------------- #


async def test_прямой_ответ_не_вызывает_провайдера():
    """Требование ADR-013: ответ не стоит ни токенов, ни ожидания.

    Проверяется не результатом, а тем, что вызов модели не состоялся, — по
    образцу теста порядка quota-first. Иначе экономия осталась бы заявлением.
    """
    backend = Orchestrator(_gateway(completion=_explode), direct=(AlwaysAnswers(),))

    response = await backend.generate(_request())
    assert isinstance(response, GenerateResponse)
    assert response.answer == "готовый ответ"

    events = [e async for e in backend.stream(_request())]
    assert any(isinstance(e, TokenEvent) and e.delta == "готовый ответ" for e in events)
    assert not any(isinstance(e, ErrorEvent) for e in events)


async def test_оговорка_доходит_обоими_путями():
    """Отметка актуальности бесполезна, если теряется в одном из режимов.

    Проект уже платил за это: фактически ответившая модель была видна только в
    ответе целиком, пока поле не добавили и в поток (раздел 54.3).
    """
    backend = Orchestrator(_gateway(completion=_explode), direct=(AlwaysAnswers(),))

    whole = await backend.generate(_request())
    assert isinstance(whole, GenerateResponse)
    assert whole.disclaimer == "оговорка"

    events = [e async for e in backend.stream(_request())]
    metadata = [e for e in events if isinstance(e, MetadataEvent)]
    assert metadata and metadata[0].disclaimer == "оговорка"


async def test_путь_ответа_виден_в_маршрутизации():
    """Иначе прямой ответ неотличим от сгенерированного при разборе жалобы."""
    backend = Orchestrator(_gateway(completion=_explode), direct=(AlwaysAnswers(),))

    whole = await backend.generate(_request())
    assert isinstance(whole, GenerateResponse)
    assert whole.routing.get("path") == "direct"

    events = [e async for e in backend.stream(_request())]
    done = [e for e in events if isinstance(e, DoneEvent)]
    assert done and done[0].routing.get("path") == "direct"


async def test_молчание_ответчика_уводит_на_обычный_путь():
    """«Не мой случай» не должно превращаться в отказ."""
    backend = Orchestrator(_gateway(), direct=(NeverAnswers(),))
    response = await backend.generate(_request())

    assert isinstance(response, GenerateResponse)
    assert response.routing.get("path") != "direct"


async def test_ответчики_опрашиваются_по_порядку():
    """Первый, кто взялся, и отвечает: иначе порядок стал бы случайным, а с ним
    и ответ на вопрос, попадающий сразу в две темы."""
    backend = Orchestrator(
        _gateway(completion=_explode),
        direct=(NeverAnswers(), AlwaysAnswers("второй"), AlwaysAnswers("третий")),
    )
    response = await backend.generate(_request())

    assert isinstance(response, GenerateResponse)
    assert response.answer == "второй"


# --- лимиты -------------------------------------------------------------------- #


async def test_лимиты_проверяются_и_на_прямом_пути():
    """Запрос, отвечаемый без модели, до шлюза не доходит.

    Ответ без модели ничего не стоит нам, но обработка запроса стоит: путь в
    обход лимитов стал бы способом бесплатно давить сервис.
    """
    quota = QuotaManager(
        FakeRedis(), subscriber_limit=1, session_limit=1, service_limit=1
    )
    backend = Orchestrator(
        _gateway(completion=_explode), direct=(AlwaysAnswers(),), quota=quota
    )

    await backend.generate(_request())
    second = await backend.generate(_request())

    assert isinstance(second, ProblemDetail)
    assert second.status == 429


async def test_отказ_хранилища_на_прямом_пути_даёт_503():
    """ADR-008: недоступность хранилища — это `503`, а не `429`.

    Ответ обязан совпадать с тем, что даёт шлюз: иначе абонент получал бы разные
    коды на одну причину в зависимости от пути.
    """
    quota = QuotaManager(
        BrokenRedis(), subscriber_limit=10, session_limit=10, service_limit=10
    )
    backend = Orchestrator(
        _gateway(completion=_explode), direct=(AlwaysAnswers(),), quota=quota
    )
    response = await backend.generate(_request())

    assert isinstance(response, ProblemDetail)
    assert response.status == 503
    assert response.retry_after is not None


# --- поиск контекста ------------------------------------------------------------ #


async def test_контекст_ищется_и_доходит_до_шлюза():
    """Раздел 5.3: Backend ищет, шлюз получает промпт с уже собранным контекстом."""
    kb = FakeKnowledgeBase()
    captured: dict[str, Any] = {}

    async def capture(*args: Any, **kwargs: Any) -> Any:
        captured["messages"] = kwargs.get("messages") or args[0]
        return await _completion()()

    backend = Orchestrator(_gateway(completion=capture), knowledge_base=kb)
    await backend.generate(_request())

    assert kb.queries == ["Когда передавать показания?"]
    assert "25 числа" in str(captured["messages"])


async def test_переданный_контекст_не_перетирается():
    """Служебные вызовы передают контекст сами.

    Перетереть его значило бы сделать поведение зависимым от того, настроен ли
    поиск, — а это самый неприятный вид зависимости: невидимый.
    """
    kb = FakeKnowledgeBase()
    own = ContextChunk(
        chunk_id="own-1",
        text="переданный снаружи фрагмент",
        source_title="служебный",
        relevance_score=1.0,
    )
    backend = Orchestrator(_gateway(), knowledge_base=kb)
    await backend.generate(_request(context=[own]))

    assert kb.queries == []


async def test_без_базы_знаний_запрос_идёт_как_есть():
    """Отсутствие поиска — не отказ: модель отвечает без опоры на регламенты.

    На прототипе это допустимо, на MVP означало бы ответ без источников.
    """
    backend = Orchestrator(_gateway())
    response = await backend.generate(_request())

    assert isinstance(response, GenerateResponse)


async def test_прямой_путь_не_ищет_контекст():
    """Отключения и регламентные ответы модели не отдают — искать им нечего."""
    kb = FakeKnowledgeBase()
    backend = Orchestrator(
        _gateway(completion=_explode), knowledge_base=kb, direct=(AlwaysAnswers(),)
    )
    await backend.generate(_request())

    assert kb.queries == []


# --- граница со шлюзом ----------------------------------------------------------- #


def test_шлюз_не_знает_про_прямые_ответы():
    """Ответчик жил в шлюзе, пока не было оркестратора, и слой был не тот.

    Возврат сюда означал бы, что предметная логика снова поехала в шлюз, а его
    описание («здесь нет ничего про водоканал») снова перестало быть правдой.
    """
    import app.gateway.llm_gateway as gateway_module

    assert not hasattr(gateway_module, "DirectResponder")
    assert not hasattr(gateway_module, "DirectAnswer")
    assert "direct" not in LlmGateway.__init__.__code__.co_varnames


def test_шлюз_не_знает_про_поиск():
    """Раздел 5.3 подписывает связь Backend → Vector DB, а не шлюз → Vector DB."""
    import app.gateway.llm_gateway as gateway_module

    source = gateway_module.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "KnowledgeBase" not in text
    assert "app.kb" not in text


@pytest.mark.parametrize("method", ["stream", "generate"])
def test_оркестратор_отдаёт_тот_же_контракт(method: str):
    """HTTP-слой не должен отличать Backend от шлюза: события те же."""
    assert hasattr(Orchestrator, method)
    assert hasattr(LlmGateway, method)
