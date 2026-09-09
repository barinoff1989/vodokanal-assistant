"""Проверки ответа о фактах лицевого счёта.

Работают на **настоящем примере** из `data_example/` — как и `test_billing_source`:
подменять данные значило бы не проверить путь, которым ассистент отвечает на
демонстрации.

Главное, что сторожат проверки, — **граница «факт / спор»**: на «сколько я
должен» ответчик отвечает суммой, на «почему такой долг» молчит и пускает
запрос в базу знаний.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from app.backend.orchestrator import Orchestrator
from app.billing.answer import AccountResponder, resolve_period
from app.billing.source import CsvBillingSource
from app.config import Settings
from app.gateway.guardrails import SyncGuardrails
from app.gateway.llm_gateway import LlmGateway
from app.gateway.pii_filter import PiiSanitizer
from app.models import (
    GenerateRequest,
    GenerateResponse,
    GenerationParameters,
    RequestMetadata,
)
from app.taxonomy import Topic

DATA = Path(__file__).resolve().parents[2] / "data_example"
NOW = datetime(2026, 9, 8, 10, 0)

DEBTOR = "2100202213"
"""Счёт с долгом и начислениями за все три периода (`test_billing_source`)."""


@pytest.fixture(scope="module")
def billing() -> CsvBillingSource:
    return CsvBillingSource(DATA)


@pytest.fixture(scope="module")
def responder(billing: CsvBillingSource) -> AccountResponder:
    return AccountResponder(billing)


@pytest.fixture(scope="module")
def no_debt_account(billing: CsvBillingSource) -> str:
    return next(a.number for a in billing._accounts.values() if not a.has_debt)  # noqa: SLF001


def ask(
    query: str,
    *,
    subscriber: str = DEBTOR,
    topic: Topic | None = Topic.ACCOUNT,
) -> GenerateRequest:
    return GenerateRequest(
        system="помощник абонента",
        query=query,
        parameters=GenerationParameters(stream=False),
        metadata=RequestMetadata(
            subscriber_id=subscriber, session_id="sess-1", topic=topic
        ),
    )


# --- граница «факт / спор» ------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "почему у меня такая задолженность, я же плачу",
        "на каком основании начислено 900 рублей",
        "прошу пересчитать задолженность по числу проживающих",
        "откуда долг за три месяца",
        "начисление в квитанции необоснованно завышено",
        "я не согласен с суммой",
    ],
)
def test_спор_о_сумме_ответчик_не_трогает(responder: AccountResponder, query: str):
    """На «почему» ответить «сколько» — хуже, чем промолчать: запрос уйдёт в
    базу знаний и к оператору, где такому вопросу и место."""
    assert responder.answer(ask(query), now=NOW) is None


def test_чужая_тема_молчание(responder: AccountResponder):
    assert responder.answer(ask("сколько я должен", topic=Topic.GENERAL), now=NOW) is None
    assert responder.answer(ask("сколько я должен", topic=None), now=NOW) is None


def test_неизвестный_абонент_молчание(responder: AccountResponder):
    assert responder.answer(ask("сколько я должен", subscriber="0000000000"), now=NOW) is None


def test_нераспознанный_подвопрос_молчание(responder: AccountResponder):
    """Тема account, но подвопрос не про долг/начисления/поверку/показания."""
    assert responder.answer(ask("здравствуйте"), now=NOW) is None


# --- задолженность -------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    ["сколько я должен?", "какая у меня задолженность", "есть ли долг", "какой долг"],
)
def test_долг_называет_сумму_и_разбивку(responder: AccountResponder, query: str):
    answer = responder.answer(ask(query), now=NOW)
    assert answer is not None
    assert "1 250,00 ₽" in answer.text
    assert "В том числе" in answer.text


def test_нет_долга_говорит_прямо(responder: AccountResponder, no_debt_account: str):
    answer = responder.answer(
        ask("есть ли задолженность", subscriber=no_debt_account), now=NOW
    )
    assert answer is not None
    assert "задолженности нет" in answer.text.lower()


# --- начисления --------------------------------------------------------- #


def test_начисления_за_период_из_вопроса(responder: AccountResponder):
    answer = responder.answer(ask("сколько начислено за 07.2026"), now=NOW)
    assert answer is not None
    assert "июль 2026" in answer.text
    assert "начислено" in answer.text
    assert "Всего начислено" in answer.text


def test_начисления_без_периода_берут_последний(responder: AccountResponder):
    answer = responder.answer(ask("сколько мне начислили"), now=NOW)
    assert answer is not None
    assert "август 2026" in answer.text


def test_начисления_за_отсутствующий_период_честны(responder: AccountResponder):
    answer = responder.answer(ask("сколько начислили за январь 2020"), now=NOW)
    assert answer is not None
    assert "в данных нет" in answer.text
    assert "Есть за" in answer.text


# --- поверка ----------------------------------------------------------- #


def test_поверка_различает_действующую_и_просроченную(responder: AccountResponder):
    answer = responder.answer(ask("когда поверка счётчиков и истёк ли срок"), now=NOW)
    assert answer is not None
    assert "истёк" in answer.text
    assert "по нормативу" in answer.text
    assert "действительна до" in answer.text


# --- показания ------------------------------------------------------- #


def test_показания_называют_значение_и_дату(responder: AccountResponder):
    answer = responder.answer(ask("какие у меня последние показания"), now=NOW)
    assert answer is not None
    assert "последнее показание" in answer.text
    assert "передано" in answer.text


# --- разбор периода ------------------------------------------------- #


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("сколько начислили за июнь", "2026-06"),
        ("начисления за июнь 2026", "2026-06"),
        ("сколько за 06.2026", "2026-06"),
        ("что начислено за 2026-07", "2026-07"),
        ("начислено в мае", "2026-05"),
        ("начисления за август", "2026-08"),
    ],
)
def test_период_из_вопроса(query: str, expected: str):
    available = ["2026-08", "2026-07", "2026-06"]
    assert resolve_period(query, available=available, today=NOW.date()) == expected


def test_период_не_назван_и_данных_нет(responder: AccountResponder):
    assert resolve_period("сколько начислено", available=[], today=NOW.date()) is None


# --- сквозь оркестратор ------------------------------------------------ #


async def _explode(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("провайдер вызван, хотя ответ есть в Биллинге")


@pytest.mark.asyncio
async def test_вопрос_о_долге_отвечается_из_биллинга_без_модели(
    billing: CsvBillingSource,
):
    """Сквозная проверка: Triage сам ставит тему, ответчик отвечает суммой,
    провайдер не вызывается. Метаданные тему не несут — классификация её и
    выводит из текста вопроса."""
    sanitizer = PiiSanitizer(analyzer=None)
    gateway = LlmGateway(
        sanitizer=sanitizer,
        guardrails=SyncGuardrails(sanitizer=sanitizer),
        completion=_explode,
        settings=Settings(_env_file=None, llm_provider="local-test"),
        quota=None,
    )
    backend = Orchestrator(gateway, direct=(AccountResponder(billing),))

    request = GenerateRequest(
        system="помощник абонента",
        query="сколько я должен?",
        parameters=GenerationParameters(stream=False),
        metadata=RequestMetadata(subscriber_id=DEBTOR, session_id="s-1"),
    )
    result = await backend.generate(request)

    assert isinstance(result, GenerateResponse)
    assert "1 250,00 ₽" in result.answer
    assert result.routing.get("path") == "direct"
