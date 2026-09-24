"""Проверки реестра регламентных ответов (ADR-100).

Главное, что здесь сторожится, — **неизменность и происхождение текста**. Ошибка
в этом модуле не роняет сервис: она тихо отдаёт абоненту не тот текст под видом
регламентного, а регламентный текст на то и регламентный, что ему верят без
проверки.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.models import GenerateRequest, RequestMetadata
from app.regulated import (
    PROTOTYPE_REGISTRY,
    RegulatedAnswer,
    RegulatedResponder,
    UnapprovedAnswerError,
)
from app.taxonomy import Topic

PHONE = "8 (473) 206-77-06"
NOW = datetime(2026, 9, 4, 12, 0)


def request(topic: Topic | None, query: str = "какая у вас вода") -> GenerateRequest:
    return GenerateRequest(
        system="помощник абонента",
        query=query,
        metadata=RequestMetadata(subscriber_id="21000001", session_id="s-1", topic=topic),
    )


@pytest.fixture
def responder() -> RegulatedResponder:
    return RegulatedResponder(phone=PHONE, strict=False)


def test_на_свою_тему_отвечает_без_модели(responder):
    answer = responder.answer(request(Topic.WATER_QUALITY), now=NOW)
    assert answer is not None
    assert answer.text


def test_на_чужую_тему_молчит(responder):
    """Молчание означает «не мой случай», и запрос идёт обычным путём."""
    for topic in (Topic.OUTAGE, Topic.GENERAL):
        assert responder.answer(request(topic), now=NOW) is None


def test_без_темы_молчит(responder):
    """Тема не проставлена — классификация не дошла.

    Отвечать регламентным текстом наугад нельзя: он на то и регламентный, что
    отдаётся только там, где положено."""
    assert responder.answer(request(None), now=NOW) is None


def test_номер_подставляется_из_настроек(responder):
    """Второй экземпляр номера разошёлся бы с настройками при первой правке."""
    answer = responder.answer(request(Topic.WATER_QUALITY), now=NOW)
    assert answer is not None
    assert PHONE in answer.text
    assert "{phone}" not in answer.text


def test_текст_не_зависит_от_вопроса_и_времени(responder):
    """Дословность — свойство записи, а не результат сборки на лету."""
    first = responder.answer(request(Topic.WATER_QUALITY, "чем вы чистите воду"), now=NOW)
    second = responder.answer(
        request(Topic.WATER_QUALITY, "вода пахнет хлоркой"),
        now=datetime(2027, 1, 1),
    )
    assert first is not None and second is not None
    assert first.text == second.text


def test_неутверждённая_запись_роняет_строгую_загрузку():
    """На MVP строгий режим обязателен: наш текст не должен уйти как утверждённый.

    Ошибка, а не пропуск записи: молча выключенный регламентный ответ отправил бы
    вопрос на генерацию — туда, откуда ADR-100 его и убирал."""
    with pytest.raises(UnapprovedAnswerError):
        RegulatedResponder(phone=PHONE, strict=True)


def test_утверждённая_запись_строгую_загрузку_проходит():
    """Проверка сторожит сам механизм, а не текущее состояние реестра."""
    approved = RegulatedAnswer(
        topic=Topic.WATER_QUALITY,
        template="утверждённый текст, телефон {phone}",
        source="владелец",
        owner="владелец",
        approved=True,
    )
    responder = RegulatedResponder((approved,), phone=PHONE, strict=True)
    answer = responder.answer(request(Topic.WATER_QUALITY), now=NOW)
    assert answer is not None
    assert answer.text == f"утверждённый текст, телефон {PHONE}"


def test_запись_прототипа_помечена_неутверждённой():
    """Пока владелец не передал формулировку, поле обязано это показывать.

    Синтетический код, принятый за настоящий, — ошибка того же рода, что наш
    текст, принятый за утверждённый."""
    (water,) = PROTOTYPE_REGISTRY
    assert water.approved is False
    assert water.effective_from is None
    assert "не утверждена" in water.source


def test_текст_прототипа_ничего_не_утверждает_за_владельца():
    """Заявление «вода соответствует нормативам» — утверждение о безопасности
    водоснабжения, и делать его за владельца нельзя (ADR-100).

    Проверка грубая по построению: она не разбирает смысл, а сторожит, что
    формулировки-заявления не появятся при правке текста «чтобы звучало лучше»."""
    (water,) = PROTOTYPE_REGISTRY
    text = water.template.lower()
    for claim in ("соответствует", "безопасн", "пригодна", "отвечает требованиям"):
        assert claim not in text, f"текст утверждает за владельца: {claim!r}"


def test_у_каждой_темы_не_больше_одной_записи():
    """Две записи на тему означали бы, что дословность зависит от порядка."""
    topics = [answer.topic for answer in PROTOTYPE_REGISTRY]
    assert len(topics) == len(set(topics))
