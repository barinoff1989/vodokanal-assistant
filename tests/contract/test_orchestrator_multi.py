"""Backend: несколько задач в одной реплике — ответ на каждую часть.

Раньше вторая задача молча терялась: «Какой у меня долг и когда поверка?» отвечало
только про поверку. Здесь сторожится обратное: каждая часть получает свой ответ
или прямое «не выполняю», а модель не зовётся там, где она не нужна.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from app.backend.orchestrator import NO_ACTIONS_RULE, DirectAnswer, Orchestrator
from app.config import Settings
from app.models import (
    DoneEvent,
    ErrorEvent,
    GenerateRequest,
    GenerationParameters,
    MetadataEvent,
    RequestMetadata,
    TokenEvent,
)
from app.taxonomy import Topic
from tests.contract.test_orchestrator import (
    FakeKnowledgeBase,
    NeverAnswers,
    TopicOnlyResponder,
    _explode,
    _gateway,
)


def _request(query: str, **meta: Any) -> GenerateRequest:
    return GenerateRequest(
        system="Ты помощник абонента водоканала.",
        query=query,
        parameters=GenerationParameters(stream=True),
        metadata=RequestMetadata(subscriber_id="sub-1", session_id="sess-1", **meta),
    )


async def _collect(backend: Orchestrator, request: GenerateRequest):
    events = [e async for e in backend.stream(request)]
    text = "".join(e.delta for e in events if isinstance(e, TokenEvent))
    return events, text


def _streaming(text: str, seen: list[dict[str, Any]] | None = None):
    """Потоковый ответ модели по кускам — как отдаёт провайдер."""

    async def call(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        if seen is not None:
            seen.extend(kwargs["messages"])

        async def gen():
            for piece in text.split(" "):
                yield {"choices": [{"delta": {"content": piece + " "}}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        return gen()

    return call


class _Account:
    address = "г. Воронеж, ул. Тестовая, д. 1"

    def __init__(self, debt: str) -> None:
        self.debt = Decimal(debt)

    @property
    def has_debt(self) -> bool:
        return self.debt > 0


class _Billing:
    def __init__(self, debt: str) -> None:
        self._account = _Account(debt)

    def account(self, number: str) -> _Account:  # noqa: ARG002
        return self._account


class _Nothing:
    """Ответчик, который никогда не отвечает: всё идёт к базе знаний и модели."""

    def answer(self, request: GenerateRequest, *, now: datetime) -> DirectAnswer | None:  # noqa: ARG002
        return None


async def test_обе_части_получают_ответ_без_модели():
    backend = Orchestrator(
        _gateway(completion=_explode),
        direct=(
            TopicOnlyResponder(Topic.OUTAGE, "отключений по адресу нет"),
            TopicOnlyResponder(Topic.TARIFF, "тариф 36,55"),
        ),
    )
    request = _request("Когда отключат воду на моей улице и сколько стоит кубометр воды?")

    events, text = await _collect(backend, request)

    assert "отключений по адресу нет" in text
    assert "тариф 36,55" in text
    assert "**1." in text and "**2." in text
    done = [e for e in events if isinstance(e, DoneEvent)]
    assert done[0].routing == {"path": "multi", "parts": 2}
    assert not any(isinstance(e, ErrorEvent) for e in events)


async def test_операция_без_выполнения_отказывает_прямо_и_не_зовёт_модель():
    backend = Orchestrator(_gateway(completion=_explode), direct=(NeverAnswers(),))

    _, text = await _collect(backend, _request("Хочу передать показания 1234"))

    assert "выполнить не может" in text
    assert "ничего не передавал" in text


async def test_условие_про_долг_выполняется_только_когда_долга_нет():
    query = "Скажите, какой у меня долг. Если долга нет, закажите поверку"
    without_debt = Orchestrator(
        _gateway(completion=_streaming("Заказ оформляется через кабинет.")),
        direct=(TopicOnlyResponder(Topic.ACCOUNT, "долг: нет"),),
        billing=_Billing("0"),
    )
    with_debt = Orchestrator(
        _gateway(completion=_explode),
        direct=(TopicOnlyResponder(Topic.ACCOUNT, "долг: 1250"),),
        billing=_Billing("1250"),
    )

    _, text_ok = await _collect(without_debt, _request(query))
    _, text_debt = await _collect(with_debt, _request(query))

    assert "не выполнено" not in text_ok
    assert "не выполнено: по счёту есть задолженность" in text_debt


async def test_непроверяемое_условие_не_выполняется_молча():
    backend = Orchestrator(
        _gateway(completion=_explode),
        direct=(TopicOnlyResponder(Topic.ACCOUNT, "поверка до 2029"),),
        billing=_Billing("0"),
    )
    query = "Когда поверка счётчика? Если скоро — закажите поверку"

    _, text = await _collect(backend, _request(query))

    assert "проверить не могу" in text
    assert "эту часть не выполняю" in text


async def test_к_модели_уходит_правило_не_писать_что_сделано():
    """Первый рубеж от ложного «сделал»: правило в системной части запроса."""
    seen: list[dict[str, Any]] = []
    backend = Orchestrator(
        _gateway(completion=_streaming("Справку можно получить в кабинете.", seen)),
        direct=(_Nothing(),),
        knowledge_base=FakeKnowledgeBase(),
    )
    await _collect(backend, _request("Как получить справку об отсутствии задолженности?"))

    system = [m["content"] for m in seen if m["role"] == "system"]
    assert system and NO_ACTIONS_RULE.strip() in system[0]


async def test_обычная_реплика_идёт_прежним_путём():
    backend = Orchestrator(
        _gateway(completion=_explode), direct=(TopicOnlyResponder(Topic.TARIFF, "тариф 36,55"),)
    )

    events, text = await _collect(backend, _request("Сколько стоит кубометр воды?"))

    assert text == "тариф 36,55"
    assert [e for e in events if isinstance(e, DoneEvent)][0].routing == {"path": "direct"}
    assert [e for e in events if isinstance(e, MetadataEvent)]


async def test_тема_из_метаданных_отключает_разбор():
    """Пришедшая снаружи тема считается более осведомлённой — реплику не делим."""
    backend = Orchestrator(
        _gateway(completion=_explode),
        direct=(TopicOnlyResponder(Topic.TARIFF, "тариф 36,55"),),
    )
    request = _request(
        "Какой у меня долг и когда поверка счётчика?", topic=Topic.TARIFF
    )

    events, text = await _collect(backend, request)

    assert text == "тариф 36,55"
    assert [e for e in events if isinstance(e, DoneEvent)][0].routing == {"path": "direct"}


def test_настройки_содержат_телефон_контакт_центра():
    """Отказ указывает, куда обратиться: без телефона он бесполезен."""
    assert Settings(_env_file=None).contact_center_phone
