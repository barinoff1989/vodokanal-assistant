"""Проверки Backend: выбор пути ответа, поиск контекста, лимиты.

По разделу 5.2 Backend отвечает за оркестрацию и сборку промпта, а шлюз — за
модель, лимиты и защиту. Эти проверки сторожат границу: предметные пути не
должны сползти обратно в шлюз, а поиск — оказаться там, где его не ждут.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from app.agents.triage import Triage
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
from app.taxonomy import Topic


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


class TopicOnlyResponder:
    """Отвечает, только если тема запроса — заданная. По образцу OutageResponder."""

    def __init__(self, topic: Topic, text: str = "ответ по теме") -> None:
        self._topic = topic
        self._answer = DirectAnswer(text=text)

    def answer(self, request: GenerateRequest, *, now: datetime) -> DirectAnswer | None:  # noqa: ARG002
        if request.metadata.topic is self._topic:
            return self._answer
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
    """«Не мой случай» не должно превращаться в отказ.

    База знаний здесь не настроена намеренно: без неё правило консультанта не
    применяется, и запрос идёт к модели. Проверка именно этого случая — что
    молчание ответчика не подменяется отказом.
    """
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


# --- прогрев ------------------------------------------------------------------ #


def test_оркестратор_прогревает_шлюз():
    """Прогрев доходит до шлюза через оркестратор, а не теряется по дороге."""
    warmed: list[str] = []

    class TrackingGateway:
        def warmup(self) -> None:
            warmed.append("да")

    orchestrator = Orchestrator(TrackingGateway())  # type: ignore[arg-type]
    orchestrator.warmup()

    assert warmed == ["да"]


def test_интерфейс_прогревает_настоящий_оркестратор():
    """**Проверка, которой не хватало, когда прогрев сломался.**

    Соседняя проверка в `test_llm_gateway.py` подставляет собственную заглушку с
    методом `warmup` и доказывает, что интерфейс вызывает его **у того, у кого он
    есть**. Что настоящий объект метод потерял, она увидеть не может — и не
    увидела: с появлением Backend внедрять стали оркестратор, у которого метода
    не было, `getattr` вернул `None`, и прогрев перестал выполняться вовсе.

    Здесь через интерфейс проходит **настоящий** `Orchestrator**, и заглушкой
    подменён только шлюз — то есть проверяется вся цепочка внедрения целиком.
    """
    from fastapi.testclient import TestClient

    from app.api import create_app

    warmed: list[str] = []

    class TrackingGateway:
        def warmup(self) -> None:
            warmed.append("да")

        async def generate(self, request: GenerateRequest) -> Any:
            return GenerateResponse(answer="", model="m", trace_id="t")

        async def stream(self, request: GenerateRequest) -> Any:
            yield DoneEvent(trace_id="t")

    orchestrator = Orchestrator(TrackingGateway())  # type: ignore[arg-type]
    with TestClient(create_app(orchestrator)):
        pass

    assert warmed == ["да"], "интерфейс не прогрел настоящий оркестратор при запуске"


# --- адрес берётся из Биллинга, а не из запроса ------------------------------- #


class _FakeBilling:
    """Биллинг-заглушка: знает адрес одного абонента."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def account(self, number: str) -> Any:
        addr = self._mapping.get(number)
        if addr is None:
            return None
        return type("Acc", (), {"address": addr})()


class _AddressSpy:
    """Прямой ответчик, запоминающий адрес, с которым его позвали."""

    def __init__(self) -> None:
        self.seen: list[str | None] = []

    def answer(self, request: GenerateRequest, *, now: datetime) -> DirectAnswer:  # noqa: ARG002
        self.seen.append(request.metadata.address)
        return DirectAnswer(text="ок")


async def test_адрес_подставляется_из_биллинга_а_не_из_запроса():
    spy = _AddressSpy()
    backend = Orchestrator(
        _gateway(),
        direct=(spy,),
        billing=_FakeBilling({"sub-1": "г. Воронеж, ул. Ленина, д. 1, кв. 5"}),
    )
    # клиент прислал чужой адрес — он игнорируется
    await backend.generate(
        _request(metadata=RequestMetadata(
            subscriber_id="sub-1", session_id="s", address="г. Москва, чужой, д. 9"
        ))
    )
    assert spy.seen == ["г. Воронеж, ул. Ленина, д. 1, кв. 5"]


async def test_неизвестный_абонент_адрес_сбрасывается():
    spy = _AddressSpy()
    backend = Orchestrator(_gateway(), direct=(spy,), billing=_FakeBilling({}))
    await backend.generate(
        _request(metadata=RequestMetadata(
            subscriber_id="sub-x", session_id="s", address="г. Где-то, д. 1"
        ))
    )
    assert spy.seen == [None]


async def test_без_биллинга_адрес_из_запроса_сохраняется():
    """Служебные вызовы и проверки задают адрес сами."""
    spy = _AddressSpy()
    backend = Orchestrator(_gateway(), direct=(spy,))
    await backend.generate(
        _request(metadata=RequestMetadata(
            subscriber_id="sub-1", session_id="s", address="г. Воронеж, д. 2"
        ))
    )
    assert spy.seen == ["г. Воронеж, д. 2"]


# --- запасной классификатор темы (app/agents/topic_fallback.py) --------------- #


class _EmptyKnowledgeBase:
    """Поиск, который никогда ничего не находит — включает отказ консультанта."""

    def search(self, query: str, **kwargs: Any) -> list[ContextChunk]:  # noqa: ARG002
        return []


class FakeTopicFallback:
    """Дубль запасного классификатора — отдаёт заданную тему без вызова модели."""

    def __init__(self, topic: Topic | None) -> None:
        self._topic = topic
        self.calls = 0

    async def classify(self, query: str) -> Topic | None:  # noqa: ARG002
        self.calls += 1
        return self._topic


async def test_правила_не_нашли_тему_но_модель_помогла():
    """Живой случай раздела 90 журнала: «отключение» без слова «вода» мимо
    маркеров outage — запасной классификатор находит тему, ответчик отвечает."""
    fallback = FakeTopicFallback(Topic.OUTAGE)
    triage = Triage(model_fallback=fallback)
    backend = Orchestrator(
        _gateway(completion=_explode),
        direct=(TopicOnlyResponder(Topic.OUTAGE),),
        triage=triage,
    )
    response = await backend.generate(_request(query="по адресу до скольки отключение?"))

    assert isinstance(response, GenerateResponse)
    assert response.answer == "ответ по теме"
    assert fallback.calls == 1


async def test_модель_не_нашла_тему_путь_как_раньше():
    """Запасной классификатор вернул None (модель тоже не помогла) — путь
    прежний: тема остаётся GENERAL, до модели не дошло только благодаря тому,
    что провайдер здесь заглушка, которая падает."""
    fallback = FakeTopicFallback(None)
    triage = Triage(model_fallback=fallback)
    backend = Orchestrator(
        _gateway(),
        direct=(NeverAnswers(),),
        triage=triage,
        knowledge_base=_EmptyKnowledgeBase(),
    )
    response = await backend.generate(_request(query="по адресу до скольки отключение?"))

    assert isinstance(response, GenerateResponse)
    assert fallback.calls == 1
    # Пустой поиск -> консультант отвечает без модели, как и без классификатора.
    assert "нет сведений" in response.answer.lower()


async def test_без_запасного_классификатора_поведение_прежнее():
    """Triage() по умолчанию — как до этой правки, модель не зовётся вовсе."""
    backend = Orchestrator(
        _gateway(),
        direct=(NeverAnswers(),),
        knowledge_base=_EmptyKnowledgeBase(),
    )
    response = await backend.generate(_request(query="по адресу до скольки отключение?"))
    assert isinstance(response, GenerateResponse)
    assert "нет сведений" in response.answer.lower()


async def test_тема_из_метаданных_классификатор_не_зовёт():
    """Тема пришла снаружи — ни правила, ни модель её не трогают (раздел 79)."""
    fallback = FakeTopicFallback(Topic.TARIFF)
    triage = Triage(model_fallback=fallback)
    backend = Orchestrator(
        _gateway(completion=_explode),
        direct=(TopicOnlyResponder(Topic.OUTAGE),),
        triage=triage,
    )
    response = await backend.generate(
        _request(
            query="что угодно",
            metadata=RequestMetadata(
                subscriber_id="sub-1", session_id="s", topic=Topic.OUTAGE
            ),
        )
    )
    assert isinstance(response, GenerateResponse)
    assert response.answer == "ответ по теме"
    assert fallback.calls == 0
