"""Охранитель «ложное сделал»: модель не должна писать, что выполнила операцию.

Помощник только консультирует. Ответ «Передал показания 1234. Проверил наличие
долга — его нет» — ложь, которой абонент поверит. Проверки идут в обе стороны:
и что ложь ловится, и что обычные ответы не блокируются.
"""

from __future__ import annotations

import pytest

from app.gateway.guardrails import (
    ACTION_CLAIM_FALLBACK,
    SAFE_FALLBACK,
    BlockReason,
    StreamGuard,
    SyncGuardrails,
)
from app.gateway.pii_filter import PiiSanitizer


def _guard() -> SyncGuardrails:
    return SyncGuardrails(sanitizer=PiiSanitizer(analyzer=None))


@pytest.mark.parametrize(
    "answer",
    [
        "Передал показания 1234. Проверил наличие долга — его нет. Заказую поверку.",
        "Я передала показания.",
        "Хорошо. Оформляю обращение.",
        "* Отправил заявление в контактный центр.",
        "Ответ ниже.\nПроверил долг: его нет.",
        "Уже зарегистрировал ваше обращение.",
    ],
)
def test_ложное_сделал_блокируется(answer):
    verdict = _guard().check(answer)
    assert not verdict.allowed
    assert verdict.reason is BlockReason.UNPERFORMED_ACTION
    assert verdict.text_for_subscriber == ACTION_CLAIM_FALLBACK


@pytest.mark.parametrize(
    "answer",
    [
        "Вы можете передать показания через личный кабинет.",
        "Специалист проверил счётчик и составил акт.",
        "Если вы передали показания, они учтены в расчёте.",
        "Передайте показания до 25 числа.",
        "Проверить срок поверки можно в личном кабинете.",
        "Заказать поверку можно по телефону контакт-центра.",
        "Могу оформить обращение: подтвердите кнопкой.",
        "Мастер отправил акт в офис.",
    ],
)
def test_обычные_ответы_не_блокируются(answer):
    """Ложная блокировка вредна: абонент теряет верный ответ."""
    assert _guard().check(answer).allowed


def test_текст_для_абонента_говорит_что_действие_не_выполнено():
    assert "не выполняю" in ACTION_CLAIM_FALLBACK
    assert ACTION_CLAIM_FALLBACK != SAFE_FALLBACK


async def test_поток_заменяется_текстом_про_невыполненное_действие():
    """Заглушка на потоке — та же, что в синхронной проверке, а не нейтральная."""

    async def chunks():
        for piece in ("Понятно. ", "Передал ", "показания 1234."):
            yield piece

    stream = StreamGuard(guardrails=_guard())
    out = [piece async for piece in stream.filter(chunks())]
    assert out[-1].strip() == ACTION_CLAIM_FALLBACK
    assert "показания 1234" not in "".join(out)
