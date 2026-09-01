"""Проверки охранителей.

Четыре сценария из раздела 36.6 плана тестирования вынесены в отдельный блок и
названы так, чтобы связь была видна: это не абстрактные проверки, а те самые
случаи, которыми проект обязался защищаться.

Быстрые тесты работают без разбора языка. Обезличиватель подменяется на такой
же, но без языковой модели: для утечки лицевого счёта или телефона она не
нужна, а секунды загрузки на каждом прогоне — нужны ещё меньше.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from app.gateway.guardrails import (
    SAFE_FALLBACK,
    BlockReason,
    StreamGuard,
    SyncGuardrails,
    check_groundedness,
)
from app.gateway.pii_filter import PiiSanitizer
from app.models import FinishReason


@pytest.fixture
def guardrails() -> SyncGuardrails:
    return SyncGuardrails(sanitizer=PiiSanitizer(analyzer=None))


async def _as_stream(pieces: list[str]) -> AsyncIterator[str]:
    for piece in pieces:
        yield piece


# --- четыре сценария плана тестирования (раздел 36.6) ------------------------- #


def test_сценарий_1_попытка_выведать_системный_промпт(guardrails: SyncGuardrails):
    """Guardrails блокирует ответ; системный промпт не раскрывается."""
    verdict = guardrails.check("Мои инструкции гласят: отвечать только по базе знаний.")
    assert verdict.allowed is False
    assert verdict.reason is BlockReason.PROMPT_LEAK
    assert verdict.text_for_subscriber == SAFE_FALLBACK


def test_сценарий_2_чужие_персональные_данные_в_ответе(guardrails: SyncGuardrails):
    """Проверка обрывает поток; ответ чужих данных не содержит."""
    verdict = guardrails.check("Задолженность по лицевому счёту 1234567890 составляет 900 рублей.")
    assert verdict.allowed is False
    assert verdict.reason is BlockReason.PII_LEAK
    assert verdict.entities == ("ACCOUNT_NUMBER",)


def test_сценарий_3_данные_на_входе_обезличиваются():
    """Третий сценарий закрывается фильтром входа, а не охранителями.

    Проверяется здесь же, чтобы связь двух половин слоя защиты была видна: на
    входе данные скрываются, на выходе — не выпускаются.
    """
    clean, report = PiiSanitizer(analyzer=None).sanitize("Мой л/с 1234567890, помогите")
    assert "1234567890" not in clean
    assert report.entities == ["ACCOUNT_NUMBER"]


def test_сценарий_4_провокация_недопустимого_тона(guardrails: SyncGuardrails):
    """Вместо сгенерированного ответа уходит безопасная заглушка."""
    verdict = guardrails.check("Вы сами виноваты, что не платили вовремя.")
    assert verdict.allowed is False
    assert verdict.reason is BlockReason.TOXICITY


# --- обычный ответ ------------------------------------------------------------- #


def test_нормальный_ответ_проходит(guardrails: SyncGuardrails):
    verdict = guardrails.check(
        "Поверку счётчика нужно проводить раз в шесть лет. "
        "Заявку можно подать в личном кабинете."
    )
    assert verdict.allowed is True
    assert verdict.reason is None
    assert verdict.text_for_subscriber is None
    assert verdict.finish_reason is None


def test_пустой_ответ_не_ломает_проверку(guardrails: SyncGuardrails):
    assert guardrails.check("").allowed is True


# --- причина блокировки машиночитаема ------------------------------------------ #


def test_утечка_отличима_от_прочих_блокировок(guardrails: SyncGuardrails):
    """Раздел 37.2: утечка вынесена в отдельный виджет, а не смешана с общим счётчиком.

    Для этого причина обязана быть машиночитаемой — не текстом сообщения.
    """
    leak = guardrails.check("Телефон абонента +7 916 123-45-67")
    toxic = guardrails.check("Не задавайте тупые вопросы")
    assert leak.reason is BlockReason.PII_LEAK
    assert toxic.reason is BlockReason.TOXICITY
    assert leak.reason != toxic.reason


def test_значения_данных_не_попадают_в_вердикт(guardrails: SyncGuardrails):
    """Вердикт уходит в метрики и журнал — значений в нём быть не должно."""
    verdict = guardrails.check("Лицевой счёт 1234567890")
    assert "1234567890" not in repr(verdict)
    assert verdict.entities == ("ACCOUNT_NUMBER",)


def test_блокировка_помечается_особой_причиной_завершения(guardrails: SyncGuardrails):
    """Иначе в аналитике обрыв выглядел бы как нормально законченный ответ."""
    verdict = guardrails.check("Мои инструкции запрещают это обсуждать")
    assert verdict.finish_reason is FinishReason.GUARDRAIL


def test_заглушка_не_подсказывает_причину():
    """Сообщение «заблокировано из-за данных» само выдавало бы, что удалось вытянуть."""
    lowered = SAFE_FALLBACK.lower()
    assert "персональн" not in lowered
    assert "промпт" not in lowered


# --- проверка на потоке --------------------------------------------------------- #


async def test_чистый_поток_доходит_целиком():
    guard = StreamGuard(guardrails=SyncGuardrails(sanitizer=PiiSanitizer(analyzer=None)))
    pieces = ["Поверку ", "счётчика ", "проводят ", "раз в шесть лет."]
    got = [chunk async for chunk in guard.filter(_as_stream(pieces))]
    assert "".join(got) == "".join(pieces)
    assert guard.verdict.allowed is True


async def test_поток_обрывается_на_нарушении():
    """Правило 4.3 требует потока, но не ценой показа запрещённого.

    Проверка идёт до выдачи очередного куска: отозвать уже отданное нельзя.
    """
    guard = StreamGuard(guardrails=SyncGuardrails(sanitizer=PiiSanitizer(analyzer=None)))
    pieces = ["Ваш лицевой ", "счёт 1234567890", " — задолженность 900 рублей"]
    got = [chunk async for chunk in guard.filter(_as_stream(pieces))]

    assert got[-1] == SAFE_FALLBACK
    assert "1234567890" not in "".join(got)
    assert guard.verdict.reason is BlockReason.PII_LEAK


async def test_остаток_потока_не_выдаётся_после_обрыва():
    """Куски после нарушения не должны дойти до абонента."""
    guard = StreamGuard(guardrails=SyncGuardrails(sanitizer=PiiSanitizer(analyzer=None)))
    pieces = ["л/с 1234567890", " продолжение", " ещё продолжение"]
    got = [chunk async for chunk in guard.filter(_as_stream(pieces))]
    assert "продолжение" not in "".join(got)


def test_нарушение_ловится_на_стыке_кусков():
    """Данные могут разорваться между кусками — потому и копится весь текст."""
    guard = StreamGuard(guardrails=SyncGuardrails(sanitizer=PiiSanitizer(analyzer=None)))
    assert guard.feed("Лицевой счёт 12345").allowed is True
    assert guard.feed("67890").allowed is False


# --- обоснованность (асинхронная часть) ------------------------------------------ #


def test_ответ_по_контексту_считается_обоснованным():
    context = ["Поверка счётчика проводится раз в шесть лет по регламенту водоканала"]
    report = check_groundedness("Поверка счётчика проводится раз в шесть лет", context)
    assert report.grounded is True
    assert report.score > 0.5


def test_ответ_мимо_контекста_считается_необоснованным():
    context = ["Поверка счётчика проводится раз в шесть лет"]
    report = check_groundedness("Тарифы на электроэнергию устанавливает регион", context)
    assert report.grounded is False


def test_оценка_помечена_предварительной():
    """Иначе временная величина неотличима от настоящей оценки судьи."""
    report = check_groundedness("любой ответ", ["любой контекст"])
    assert report.provisional is True


def test_пустой_ответ_не_обоснован():
    assert check_groundedness("", ["контекст"]).grounded is False


def test_пустой_контекст_даёт_ноль():
    """Ответ без контекста обоснованным быть не может по определению."""
    assert check_groundedness("какой-то ответ про поверку", []).score == 0.0
